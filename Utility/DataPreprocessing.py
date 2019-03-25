import Pandas as pd
import numpy as np

class DataPreprocessing:

    def __init__(self):
        print()

    @staticmethod
    def complete_Replace(dataframe, original, new):
        return dataframe[original].replace(original, new, inplace=True)

    @staticmethod
    def one_hot_encoder(dataframe):
        return pd.get_dummies(dataframe)

    @staticmethod
    def fill_Na(dataframe, column, fill):
        dataframe[column] = dataframe[column].fillna(fill)
        return dataframe

    @staticmethod
    def feature_scaling(dataframe, column):
        dataframe[column] = np.divide(np.subtract(dataframe, dataframe[column].mean()),
                                      dataframe[column].std())
        return dataframe[column]

    @staticmethod
    def split(dataframe, numrows):
        test_data = dataframe.tail(numrows).reset_index(drop=True)
        # here we reset index as if the index remain same for accessing by index could be difficult
        return dataframe.head(len(dataframe)-numrows).reset_index(), test_data.reset_index()

    @staticmethod
    def sigmoid_Function(Z):
        gz= np.divide(1, 1+np.power(np.e,np.multiply(-1,Z)))
        return gz

    