# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import preprocessing
from metrics_lib import data_similarity
from plot_lib import plot_data

def read_excel(file_path, sheet_name, header = 0):
    '''
    args:
        file_path:

    return:
    '''
    file_path.replace(' ','')
    df = pd.read_excel(file_path, sheet_name = sheet_name, header = header)
    print()
    #
    return df

def normalization(X):
    #
##    if dtype(X) == "numpy.ndarray"
    if np.max(X) != 1 and np.min(X) != 0:
        scaler = preprocessing.MinMaxScaler(X, copy = False)
        data = scaler.fit_transform(X)
    else:
        data = X
    #
    return data

def inverse_normalization(X, scaler):
    #
    data = scaler.inverse_transform(X)
    #
    return data

def shuffle_data(data):
    pass


def sort_data(data):
    pass


def identical_distribution_split(target, n_splits = 3, train_size = 0.7,
                                 test_size = 0.3, cos_theta_lim = 0.7):
    #
    cos_theta_list = [0.0]
    #
    while min(cos_theta_list) < cos_theta_lim:
        #
        del cos_theta_list[:]
        index = []
        cv = model_selection.ShuffleSplit(n_splits = n_splits,
                                          train_size = train_size,
                                          test_size = test_size)
        #
        for index1, index2 in cv.split(target):
            #测试集必须与训练集、验证集分开归一化处理，不可合并归一化
            dataset1 = target[index1][:,np.newaxis]
            dataset2 = target[index2][:,np.newaxis]
            #
            cos_theta, _, _ =  data_similarity.hist_similarity(data_normalization(dataset1),
                                                               data_normalization(dataset2))
            cos_theta_list.append(cos_theta)
            index.extend([index1,index2])
    #
    return cv, index, cos_theta_list
        
def split_data(df_data):
    '''
    args:
        df_data:

    return:
    '''
    training_set = df_data.sample(frac=0.7,random_state=1)
    tmp_set = df_data.loc[~df_data.index.isin(training_set.index),:]
    validation_set = tmp_set.sample(frac=0.5,random_state=1)
    test_set = tmp_set.loc[~tmp_set.index.isin(validation_set.index),:]
    #

    #
    return training_set, validation_set, test_set


def write_excel(df_data, file_path, sheet_name = None,
                row_title = True, col_title = True):
    '''
    args:
        df_data:

    return:
    '''
    df_data.to_excel(file_path, sheet_name = sheet_name,
                     index = row_title, header = col_title)


if __name__ == "__main__":
    #
    in_file_path = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\src_data\sklearn_boston.xlsx"
    out_file_path1 = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\training_sklearn_boston.xlsx"
    out_file_path2 = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\validation_sklearn_boston.xlsx"
    out_file_path3 = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\test_sklearn_boston.xlsx"
    #
    df = read_excel(in_file_path, "Sheet", header = None)
    training_set, validation_set, test_set = split_data(df)
    #
    y_training = training_set.values[:,-1]
    y_val = validation_set.values[:,-1]
    y_test = test_set.values[:,-1]
    a = imitate_split(y_training,1,0.8,0.2)
##    #
##    write_excel(training_set, out_file_path1, sheet_name = "Sheet1",
##                row_title = False, col_title = False)
##    write_excel(validation_set, out_file_path2, sheet_name = "Sheet1",
##                row_title = False, col_title = False)
##    write_excel(test_set, out_file_path3, sheet_name = "Sheet1",
##                row_title = False, col_title = False)
##    #
##    print("Finish!")
