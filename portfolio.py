import numpy as np
import pandas as pd
from typing import Optional
import cvxpy as cp
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

num_stocks = 503


# TODO: the use the class Portfolio, after finding the best, we need to call his class Portfolio
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
        constraints = [b >= 0, cp.sum(b) <= 0.9999]
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        return b.value

    def train(self, train_data: pd.DataFrame) -> Optional[np.ndarray]:
        return None

    def __repr__(self):
        return f"%s,PAMR type=%s,epsilon=%s,c=%s" % (self.__class__, self.pamr_type, self.epsilon, self.c)


class MarketPortfolio:

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
        return "%s" % self.__class__


class DeepPortfolio(nn.Module):
    def __init__(self, num_layers=1, hidden_size=64, window_size=30, dropout=0.2):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=num_stocks * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_size, num_stocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (h_n, c_n) = self.lstm(x)
        output = self.linear(h_n)
        output = F.softmax(output, dim=-1).squeeze(1)
        return output

    def get_portfolio(self, train_data: pd.DataFrame) -> np.ndarray:
        """
        The function used to get the model's portfolio for the next day
        :param train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history,
        with all the training data. The following day (the first that does not appear in the index) is the test day
        :return: a numpy array of shape num_stocks with the portfolio for the test day
        """
        data = self.preprocess_data(train_data)[None, :]
        return self.forward(data).numpy()

    def preprocess_data(self, train_data: pd.DataFrame) -> torch.Tensor:
        adj_close = train_data['Adj Close']
        relative_returns = adj_close.pct_change(1).iloc[1:, :]
        adj_close = torch.tensor(adj_close.iloc[-self.window_size:, :].values)
        relative_returns = torch.tensor(relative_returns.iloc[-self.window_size:, :].values)
        data = torch.zeros(self.window_size, 2 * num_stocks)
        data[:, ::2] = adj_close
        data[:, 1::2] = relative_returns
        return data
