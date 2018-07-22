# coding: utf-8 -*-

import os
import numpy as np
import openpyxl as opxl
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import ensemble
from sklearn import model_selection
from sklearn import metrics


def read_xlsx_file(in_file_path, sheet_name, data_range):
    #
    data = []
    #
    wb = opxl.load_workbook(in_file_path)
    ws = wb[sheet_name]
    #
    for line_data in ws[data_range]:
        tmp_data = [cell.value for cell in line_data]
        data.append(tmp_data)
    
    return np.array(data).astype(float)

def view_data_distribution(data):
    #
    x = [i for i in range(1, data.shape[0] + 1)]
    y = data
    plt.scatter(y,x)
    plt.show()

def get_training_test_set(in_train_path, train_sheet_name, train_set_range, 
                          in_test_path, test_sheet_name, test_set_range):
    #
    training_set = read_xlsx_file(in_train_path, train_sheet_name, train_set_range)
    test_set = read_xlsx_file(in_test_path, test_sheet_name, test_set_range)
    #
    return training_set[:,:-1],test_set[:,:-1],\
           training_set[:,-1],test_set[:,-1]

##def get_training_test_set(in_file_path, sheet_name, data_range):
##    #
##    training_set = read_xlsx_file(in_file_path, sheet_name, data_range)
##    X = training_set[:,:-1]
##    y = list(training_set[:,-1])
##    print(y)
##    X_train, X_test, y_train, y_test = model_selection.train_test_split(
##                                                        X, y,
##                                                        test_size = 0.3,
##                                                        random_state = 0,
##                                                        stratify = y
##                                                    )
        
    #
    return X_train, X_test, y_train, y_test

def dim_reduction(samples, ndim):
    #
    pca = decomposition.PCA(n_components = ndim, svd_solver = "full")
    target_samples = pca.fit_transform(samples)
    #
    return target_samples

def run_model(training_samples, training_labels, n_estimators , max_features):
    #
    training_model_score = []
    #
    rf_model = ensemble.RandomForestRegressor(n_estimators = n_estimators,
                                              max_features = max_features,
                                              oob_score = True,
                                              random_state=42)
    #
    kf = model_selection.KFold(n_splits = 5, random_state=0)
    kf.get_n_splits(training_samples)
    for training_index, cv_index in kf.split(training_samples):
        training_X = training_samples[training_index]
        training_y = training_labels[training_index]
        #
        cv_X = training_samples[cv_index]
        cv_y = training_labels[cv_index]
        #
        rf_model.fit(training_X, training_y)
        #
        y_pred = rf_model.predict(cv_X)
        #
        r2 = rf_model.score(cv_X, cv_y)
        mse = metrics.mean_squared_error(cv_y, y_pred)
        oob_score = rf_model.oob_score_

        training_model_score.append([r2,mse,oob_score])

    return np.array(training_model_score), rf_model

def model_optimizer(model_score):
    #
    r2, = plt.plot(model_score[:,0])
##    mse, = plt.plot(model_score[:,1])
    oob_score, = plt.plot(model_score[:,2])
    #
    plt.legend(handles = [r2, oob_score], labels = ["r2", "oob_score"], loc = "best")
    plt.show()

def test_model(rf_model, test_samples, test_labels):
    #
    test_model_score = []
    #
    r2 = rf_model.score(test_samples, test_labels)
    y_pred = rf_model.predict(test_samples)
    mse = metrics.mean_squared_error(test_labels, y_pred)
    oob_score = rf_model.oob_score_
    test_model_score.append([r2,mse,oob_score])
    #
    return np.array(test_model_score)


if __name__ == "__main__":
    #
    in_file_dir = r'C:\Users\xrui\Desktop\test_sklearn_ml\data'
    in_file_name = ('Pb_training_set.xlsx', 'Pb_test_set.xlsx',
                    'Ag_training_set.xlsx', 'Ag_test_set.xlsx',
                    'As_training_set.xlsx', 'As_test_set.xlsx',
                    'Cr_training_set.xlsx', 'Cr_test_set.xlsx',
                    'Cu_training_set.xlsx', 'Cu_test_set.xlsx',
                    'Zn_training_set.xlsx', 'Zn_test_set.xlsx')

##    in_path = r"K:\test_sklearn_ml\RF_Data.xlsx"
##    training_samples, test_samples, training_labels, test_labels = get_training_test_set(
##                                                                            in_path,"Data","B2:BQK73")
##    #
##    view_data_distribution(training_labels)
##    view_data_distribution(test_labels)
    
    #
    for i in range(4,len(in_file_name),2):
        in_train_file_path = os.path.join(in_file_dir,in_file_name[i])
        in_test_file_path = os.path.join(in_file_dir,in_file_name[i + 1])
        training_samples, test_samples, training_labels, test_labels = get_training_test_set(
                                                                            in_train_file_path,
                                                                            "Sheet1","A1:BQJ54",
                                                                             in_test_file_path,
                                                                            "Sheet1","A1:BQJ18")
        #
##        print(training_labels.shape,test_labels.shape)
##        view_data_distribution(training_labels)
##        view_data_distribution(test_labels)
##        input('##:')
        #
        training_samples = dim_reduction(training_samples, 15)
        #
        n_estimators = 550
        max_features = "sqrt"                                                                                                                                                    
        training_model_score, rf_model = run_model(training_samples, training_labels, n_estimators, max_features)
        model_optimizer(training_model_score)
        input('...:')
        #
        test_samples = dim_reduction(test_samples, 20)
        #
        test_model_score = test_model(rf_model, test_samples, test_labels)
        model_optimizer(test_model_score)
        input('...:')
