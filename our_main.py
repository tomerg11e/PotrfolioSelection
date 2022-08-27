import pandas as pd
import yfinance as yf
from our_portfolio import PAMRPortfolio, EqualPortfolio, MinVarPortfolio, BestBasketPortfolio, \
    TangentPortfolio, SingletonPortfolio, WeightedPortfolio, momentumPortfolio, DeepPortfolio
from os.path import exists
import matplotlib.pyplot as plt
import numpy as np

SP_TICKER = "^GSPC"
START_DATE = '2017-05-01'
END_TRAIN_DATE = '2022-07-01'
END_TEST_DATE = '2022-08-27'

DATA_PATH = "data.pkl"

"""
https://www.sciencedirect.com/science/article/pii/S0377221720310407?fr=RR-2&ref=pdf_download&rr=7305a52f6d6c76d1
"""


def get_data(re_download=False):
    data = None
    if re_download:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        print("re-downloading data...\n")
        data = yf.download(tickers, START_DATE, END_TEST_DATE)
    else:
        if exists(DATA_PATH):
            print("using the existing data\n")
            data = pd.read_pickle(DATA_PATH)
        else:
            wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp_tickers = wiki_table[0]
            tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
            print("downloading data...\n")
            data = yf.download(tickers, START_DATE, END_TEST_DATE)
            data.to_pickle(DATA_PATH)
    return data


def test_portfolio(full_train, strategy):
    returns = []
    for i, test_date in enumerate(pd.date_range(END_TRAIN_DATE, END_TEST_DATE)):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    print(f"the strategy {strategy}, gave sharpe value of {sharpe}")
    return returns


def main():
    get_SP_adj_close(END_TRAIN_DATE, END_TEST_DATE)
    df = run_models()
    plotting_df_SP(df)


def get_SP_adj_close(start_date, end_date):
    data = yf.download([SP_TICKER], start_date, end_date)["Adj Close"]
    all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
    train_data = data.reindex(all_weekdays)
    train_data = train_data.fillna(method='ffill')
    train_data = train_data.pct_change(1).iloc[2:]
    return train_data


def plotting_df_SP(df):
    sp = get_SP_adj_close(END_TRAIN_DATE, END_TEST_DATE)
    ax = plt.gca()
    sp.plot.line(linestyle="--", ax=ax)
    df.plot.line(ax=ax, alpha=0.5)
    plt.legend()
    plt.title("relative return of different portfolios")
    plt.ylabel("relative return")
    plt.xlabel("date")
    plt.xticks([])
    plt.savefig("aaaa")


def run_models():
    full_train = get_data()

    model = DeepPortfolio()
    model.train_portfolio(full_train)
    models = [EqualPortfolio(), MinVarPortfolio(tau=0), SingletonPortfolio("AAPL"),
              WeightedPortfolio(), BestBasketPortfolio(), TangentPortfolio(), model,
              ]

    df = pd.DataFrame()
    for model in models:
        returns = test_portfolio(full_train, model)
        df[str(model)] = returns['return']
    return df


def plotting():
    full_train = get_data()['Adj Close']
    full_train.plot(legend=False)
    plt.title("adj close price")
    plt.show()

    to_plot = full_train.pct_change(1).iloc[2:, :]
    to_plot.plot(legend=False)
    plt.title("relative returns")

    plt.show()


if __name__ == '__main__':
    main()
