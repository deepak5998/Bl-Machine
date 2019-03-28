import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessing:

    def __init__(self):
        print("Data PreProcessign object created successfully")

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
        dataframe[column] = np.divide(np.subtract(dataframe, dataframe[column].min()),
                                      np.subtract(dataframe[column].max(),dataframe[column].min()))
        return dataframe[column]

    @staticmethod
    def split(dataframe, numrows):
        test_data = dataframe.tail(numrows).reset_index(drop=True).reset_index()
        # here we reset index as if the index remain same for accessing by index could be difficult
        dataframe = dataframe.head(dataframe.shape[0]-numrows).reset_index()
        try:
            test_data.drop(['index'],axis=1)
        except Exception as e:
            print()
        try: 
            dataframe.drop(['index'],axis=1)
        except Exception as e:
            print()
        return dataframe, test_data

    @staticmethod
    def sigmoid_Function(Z):
        gz= np.divide(1, 1+np.power(np.e,np.multiply(-1,Z)))
        return gz

    @staticmethod
    def cube(x):
        print(x**3)
    
    @staticmethod
    def separate(dataframe,col):
        output = dataframe[col]
        return dataframe.drop(col,axis=1),output
    
    @staticmethod
    def remove_outlier(dataframe):
        dataframe = dataframe[np.abs(dataframe - dataframe.mean()) <= (3 * dataframe.std())]
        try:
            dataframe.drop('index')
        except Exception as e:
            pass
        return dataframe.reset_index()
    
    @staticmethod
    def main(dataframe):
        dataframe.hist(figsize=(10,10))
        plt.show()
        print(dataframe.columns,"\nPlease copy paste and enter the columns to be scaled")
        col = input().split(',')
        print(col)
        print(list(col))