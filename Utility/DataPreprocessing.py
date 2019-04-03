import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DataPreprocessing:

    def __init__(self):
        print("Data PreProcessign object created successfully")

    @staticmethod
    def complete_Replace(dataframe, original, new):
        dataframe =  dataframe[original].replace(original, new, inplace=True)
        return DataPreprocessing.remove_index(dataframe)
    
    @staticmethod
    def one_hot_encoder(dataframe):
        dataframe = pd.get_dummies(dataframe)
        return DataPreprocessing.remove_index(dataframe)
        
    @staticmethod
    def fill_Na(dataframe, column, fill):
        dataframe[column] = dataframe[column].fillna(fill)
        dataframe = DataPreprocessing.remove_index(dataframe)
        return dataframe

    @staticmethod
    def feature_scaling(dataframe, column):
        dataframe[column] = np.divide(np.subtract(dataframe, dataframe[column].min()),
                                      np.subtract(dataframe[column].max(),dataframe[column].min()))
        return dataframe[column]
    
    @staticmethod
    def normalise(dataframe):
        # using log
        return np.log(dataframe)
    
    @staticmethod
    def standardize(dataframe):
        Y = dataframe['y']
        dataframe = np.divide(np.subtract(dataframe,np.array(dataframe.mean(axis=1)).reshape(dataframe.shape[0],1))
                              ,np.array(dataframe.std(axis=1)).reshape(dataframe.shape[0],1))
        dataframe['y'] = Y
        return dataframe                                                                                    
        
    @staticmethod
    def split(dataframe, numrows):
        test_data = dataframe.tail(numrows).reset_index(drop=True)
        # here we reset index as if the index remain same for accessing by index could be difficult
        dataframe = dataframe.head(dataframe.shape[0]-numrows).reset_index(drop=True)
        dataframe = DataPreprocessing.remove_index(dataframe)
        test_data = DataPreprocessing.remove_index(test_data)
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
        dataframe = dataframe.drop(col,axis=1)
        dataframe = DataPreprocessing.remove_index(dataframe)
        output = DataPreprocessing.remove_index(output)
        return dataframe,output
    
    @staticmethod
    def remove_outlier(dataframe):
        dataframe = dataframe[np.abs(dataframe - dataframe.mean()) <= (3 * dataframe.std())]
        dataframe = DataPreprocessing.remove_index(dataframe)
        return dataframe
    
    @staticmethod
    def remove_index(dataframe):
        try:
            dataframe = dataframe.drop('index', axis=1)
        except Exception as e:
            pass
        return dataframe
    
    @staticmethod
    def duplicates_count(dataframe):
        return dataframe.duplicated().sum()
    
    @staticmethod
    def root_mean_square(predicted,original):
        error = np.sqrt(np.sum(np.square(np.divide(np.subtract(predicted,original),predicted.shape[0]))))
        return error
    
    def mean_absolute_error(predicted,original):
        error = np.divide(np.sum(np.abs(np.subtract(predicted,original))),predicted.shape[0])
        return error
        
    @staticmethod
    def main(dataframe):
        dataframe.hist(figsize=(10,10))
        plt.show()
        print(dataframe.columns,"\nPlease copy paste and enter the columns to be scaled")
        col = input().split(',')
        print(col)
        print(list(col))