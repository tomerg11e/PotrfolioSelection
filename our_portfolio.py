import numpy as np
import pandas as pd
from typing import Optional
import cvxpy as cp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

num_stocks = 503


class PAMRPortfolio:

    def __init__(self, pamr_type="1", epsilon=0.5, c=500):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.weights = np.ones((num_stocks,)) / num_stocks
        self.mean_weights = np.ones((num_stocks,)) / num_stocks
        assert pamr_type in ["0", "1", "2"]
        assert 0 <= epsilon <= 1
        self.pamr_type = pamr_type
        self.epsilon = epsilon
        self.c = c

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close'].to_numpy()
        for i in range(train_data.shape[0]):
            # get stock prices
            x_t = train_data[i]
            # calc loss
            loss = np.maximum(0.0, self.weights @ x_t - self.epsilon)
            # calc step size
            tau = self.get_tau(loss, x_t)
            # find best portfolio
            wanted_weights = self.weights - tau * (x_t - np.sum(x_t) * self.mean_weights)
            # normalize portfolio
            self.weights = self.normalize_weights(wanted_weights)
        return self.weights

    def get_tau(self, loss: np.ndarray, x_t: np.ndarray) -> np.ndarray:
        norm_part = np.sum((x_t - np.sum(x_t) * self.mean_weights) ** 2)
        if self.pamr_type == "0":
            return loss / norm_part
        elif self.pamr_type == "1":
            return np.minimum(self.c, loss / norm_part)
        else:
            return loss / (norm_part + 1 / (2 * self.c))

    def normalize_weights(self, wanted_weights: np.ndarray) -> np.ndarray:
        b = cp.Variable(num_stocks)  # TODO make it's sum equal to one exactly!
        objective = cp.Minimize(cp.sum_squares(b - wanted_weights))
        constraints = [b >= 0, cp.sum(b) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return b.value

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"PAMR,type=%s,epsilon=%s,c=%s" % (self.pamr_type, self.epsilon, self.c)


class EqualPortfolio:

    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        pass

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close']
        return np.ones(len(train_data.columns)) / len(train_data.columns)

    def __repr__(self):
        return "EP"


class DeepPortfolio(nn.Module):
    def __init__(self, num_layers=4, hidden_size=128, window_size=30, dropout=0):
        super().__init__()
        if num_layers == 1:
            dropout = 0
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.dropout = dropout
        self.lstm = nn.LSTM(input_size=num_stocks * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(num_layers * hidden_size, num_stocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (h_n, c_n) = self.lstm(x)
        h_n = h_n.transpose(0, 1).flatten(start_dim=1)
        output = self.linear(h_n)
        output = F.softmax(output, dim=-1)
        # output = F.softmax(output-torch.max(output, dim=-1)[0].detach(), dim=-1)
        while output.sum() > 1:
            output = output / output.sum()
        return output

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        self.eval()
        with torch.no_grad():
            data = self.preprocess_data(train_data)[None, :]
            output = self.forward(data).numpy()
            while output.sum() > 1:
                output = output / output.sum()
        return output

    def preprocess_data(self, train_data: pd.DataFrame) -> torch.Tensor:
        adj_close = train_data['Adj Close']
        relative_returns = adj_close.pct_change(1).iloc[1:, :]
        adj_close = torch.tensor(adj_close.iloc[-self.window_size:, :].values)
        relative_returns = torch.tensor(relative_returns.iloc[-self.window_size:, :].values)
        data = torch.zeros(self.window_size, 2 * num_stocks)
        data[:, ::2] = adj_close
        data[:, 1::2] = relative_returns
        return data

    def train_portfolio(self, full_train):
        self.train()
        torch.autograd.set_detect_anomaly(True)

        START_DATE = '2022-04-01'
        END_TRAIN_DATE = '2022-05-31'
        END_TEST_DATE = '2022-06-30'
        date_range = pd.date_range(END_TRAIN_DATE, END_TEST_DATE)

        optimizer = optim.Adam(self.parameters())
        returns = torch.zeros((1,), requires_grad=True)
        returns_squared = torch.zeros((1,), requires_grad=True)
        train_days = 0
        for test_date in tqdm(date_range, desc="training..."):
            if test_date not in full_train.index:
                continue
            train_days += 1
            train = full_train[full_train.index < test_date]
            temp_train_data = self.preprocess_data(train)[None, :]
            cur_portfolio = self.forward(temp_train_data)
            if cur_portfolio.sum() > 1:
                raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
            test_data = full_train['Adj Close'].loc[test_date].to_numpy()
            prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
            test_data = torch.tensor(test_data / prev_test_data).to(torch.float)
            cur_return = cur_portfolio @ test_data
            returns = returns + cur_return
            returns_squared = returns_squared + torch.square(cur_return)
            if train_days == 10:
                mean_returns = returns / train_days
                mean_returns_squared = returns_squared / train_days
                our_sharpe = -mean_returns / torch.sqrt(mean_returns_squared - torch.square(mean_returns))
                our_sharpe.backward()
                optimizer.step()

                returns = torch.zeros((1,))
                returns_squared = torch.zeros((1,))
                optimizer.zero_grad()
                train_days = 0
        if train_days > 2:
            mean_returns = returns / train_days
            mean_returns_squared = returns_squared / train_days
            our_sharpe = -mean_returns / torch.sqrt(mean_returns_squared - torch.square(mean_returns))
            our_sharpe.backward()
            optimizer.step()

    def __repr__(self):
        return f"%s, num_layers=%s, hidden_size=%s, window_size=%s, dropout=%s" % (
            self.__class__, self.num_layers, self.hidden_size, self.window_size, self.dropout)


class MinVarPortfolio:
    def __init__(self, tau: float = 0):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.tau = tau

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close']
        all_weekdays = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
        train_data = train_data.reindex(all_weekdays)
        train_data = train_data.fillna(method='ffill')
        train_data = train_data.pct_change(1).iloc[2:, :].fillna(0)

        cov = train_data.cov().to_numpy()
        n = cov.shape[0]
        x = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(x, cov) + self.tau * cp.norm(x, 1))
        constraints = [cp.sum(x) == 1]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return x.value

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"MVP,tau={self.tau}"


class BestBasketPortfolio:
    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close']
        all_weekdays = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
        train_data = train_data.reindex(all_weekdays)
        train_data = train_data.fillna(method='ffill')
        train_data = train_data.pct_change(1).iloc[2:, :].fillna(0)
        desc_close = train_data.describe().T[["mean", "std"]]

        r = desc_close["mean"]
        std = desc_close["std"]
        cov = train_data.cov().to_numpy()

        c_inv = np.linalg.inv(cov)
        e = np.ones_like(r)

        bbp = (c_inv @ r) / (e.T @ c_inv @ r)
        while not np.isclose(bbp.sum(), 1):
            bbp = bbp / (e.T @ bbp)
        return bbp

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"BBP"


class TangentPortfolio:
    def __init__(self):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close']
        all_weekdays = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
        train_data = train_data.reindex(all_weekdays)
        train_data = train_data.fillna(method='ffill')
        train_data = train_data.pct_change(1).iloc[2:, :].fillna(0)
        desc_close = train_data.describe().T[["mean", "std"]]
        cov = train_data.cov().to_numpy()
        r = desc_close["mean"]
        std = desc_close["std"]
        c_inv = np.linalg.inv(cov)
        e = np.ones_like(r)

        r_f = 0.05 / 250
        r_tilde = r - r_f
        tp = (c_inv @ r_tilde) / (e.T @ c_inv @ r_tilde)
        while not np.isclose(tp.sum(), 1):
            tp = tp / (e.T @ tp)
        return tp

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"TangentPortfolio"


class SingletonPortfolio:
    def __init__(self, stock_index: str = "AAPL"):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.stock_index = stock_index

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        train_data = train_data['Adj Close']
        loc = train_data.columns.get_loc(self.stock_index)
        weights = np.zeros(num_stocks)
        weights[loc] = 1
        return weights

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"stock_index={self.stock_index}"


class WeightedPortfolio:
    def __init__(self, relative: bool = False, mean_window: int = 3):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.relative = relative
        self.mean_window = mean_window

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """

        train_data = train_data['Adj Close']
        if self.relative:
            all_weekdays = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
            train_data = train_data.reindex(all_weekdays)
            train_data = train_data.fillna(method='ffill')
            train_data = train_data.pct_change(1).iloc[2:, :].fillna(0)
        weights = train_data.tail(self.mean_window).mean().values
        weights = weights / np.sum(weights)
        while not np.isclose(weights.sum(), 1):
            weights = weights / np.sum(weights)
        return weights

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"WP,relative=%s,window=%s" % (self.relative, self.mean_window)


class momentumPortfolio:
    def __init__(self, window_size: int = 5, backup=None):
        """
        The class should load the model weights, and prepare anything needed for the testing of the model.
        The training of the model should be done before submission, here it should only be loaded
        """
        self.window_size = window_size
        if backup is None:
            backup = WeightedPortfolio(relative=True, mean_window=window_size)
        self.backup = backup

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        changed_train_data = train_data['Adj Close']
        all_weekdays = pd.date_range(start=changed_train_data.index.min(), end=changed_train_data.index.max(), freq='B')
        changed_train_data = changed_train_data.reindex(all_weekdays)
        changed_train_data = changed_train_data.fillna(method='ffill')
        changed_train_data = changed_train_data.pct_change(1).iloc[2:, :].fillna(0)

        # get window size of relative returns
        changed_train_data = changed_train_data.tail(self.window_size)
        positive_stocks = np.all(changed_train_data > 0, axis=0)
        negative_stocks = np.all(changed_train_data < 0, axis=0)
        rest_stocks = ~(positive_stocks + negative_stocks)
        positive_weight = positive_stocks.values.sum()
        negative_weight = negative_stocks.values.sum()
        rest_weight = rest_stocks.values.sum()
        positive_stocks = np.where(positive_stocks, 1, 0)
        negative_stocks = np.where(negative_stocks, -1, 0)
        if positive_weight + negative_weight == 0:
            return self.backup.get_portfolio(train_data)
        elif positive_weight == 0:
            positive_stocks = np.zeros(num_stocks)
            negative_stocks = np.where(negative_stocks, -1 / negative_weight, 0)
            rest_stocks = np.where(rest_stocks, 2 / rest_weight, 0)
        elif negative_weight == 0:
            positive_stocks = np.where(positive_stocks, 1 / positive_weight, 0)
            negative_stocks = np.zeros(num_stocks)
            rest_stocks = np.zeros(num_stocks)
        else:
            positive_stocks = np.where(positive_stocks, 2 / positive_weight, 0)
            negative_stocks = np.where(negative_stocks, -1 / negative_weight, 0)
            rest_stocks = np.zeros(num_stocks)

        output = positive_stocks + negative_stocks + rest_stocks
        return output

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"momentum portfolio,ws={self.window_size}"
