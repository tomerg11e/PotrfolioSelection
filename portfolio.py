import numpy as np
import pandas as pd


class Portfolio:

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
