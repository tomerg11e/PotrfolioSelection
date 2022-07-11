import pandas as pd
import yfinance as yf
from portfolio import PAMRPortfolio, MarketPortfolio, DeepPortfolio
from os.path import exists
from tqdm import tqdm
import matplotlib.pyplot as plt

START_DATE = '2022-04-01'
END_TRAIN_DATE = '2022-05-31'
END_TEST_DATE = '2022-06-30'

DATA_PATH = "data.pkl"


def get_data(re_download=False):
    data = None
    if re_download:
        wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp_tickers = wiki_table[0]
        tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
        print("re-downloading data...")
        data = yf.download(tickers, START_DATE, END_TEST_DATE)
    else:
        if exists(DATA_PATH):
            print("using the existing data")
            data = pd.read_pickle(DATA_PATH)
        else:
            wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            sp_tickers = wiki_table[0]
            tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
            print("downloading data...")
            data = yf.download(tickers, START_DATE, END_TEST_DATE)
            data.to_pickle(DATA_PATH)
    return data


def test_portfolio(full_train, strategy):
    print(f"running with {strategy}\n")
    returns = []
    for test_date in tqdm(pd.date_range(END_TRAIN_DATE, END_TEST_DATE)):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if cur_portfolio.sum() > 1:
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    print(sharpe)


def main():
    full_train = get_data()
    test_portfolio(full_train, DeepPortfolio())
    test_portfolio(full_train, MarketPortfolio())
    test_portfolio(full_train, PAMRPortfolio(pamr_type="0"))
    test_portfolio(full_train, PAMRPortfolio(pamr_type="1"))
    test_portfolio(full_train, PAMRPortfolio(pamr_type="2"))


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
