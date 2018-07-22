# -*- coding: utf-8 -*-

import os
import openpyxl as opxl
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import model_selection
from sklearn import svm
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

def get_training_test_set(in_train_path, train_sheet_name, train_set_range, 
                          in_test_path, test_sheet_name, test_set_range):
    #
    training_set = read_xlsx_file(in_train_path, train_sheet_name, train_set_range)
    test_set = read_xlsx_file(in_test_path, test_sheet_name, test_set_range)
    #
    X_train, X_test = training_set[:,:-1], test_set[:,:-1]
    y_train, y_test = training_set[:,-1], test_set[:,-1]
    #
    return X_train, X_test, y_train, y_test

def dim_reduction(samples, ndim):
    #
    pca = decomposition.PCA(n_components = ndim, svd_solver = "full")
    target_samples = pca.fit_transform(samples)
    #
    return target_samples


def train_model(training_samples, training_labels):
    #
    training_model_score = []
    #
    init_svr_model = svm.SVR(kernel = "linear")
    #    
##    optimize_parameters = [{"kernel":["linear"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9]},
##                           {"kernel":["rbf"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9],
##                            "gamma":[0.001,0.005,0.1,0.5,1,5,10,15,20,25,30,35,40,50]}]
    #
    optimize_parameters = {"kernel":["rbf"],
                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9],
                            "gamma":[0.001,0.005,0.1,0.5,1,5,10,15,20,25,30,35,40,50]}
    #
##    optimize_parameters = {"kernel":["linear"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9]}
    #
    svr_model = model_selection.GridSearchCV(init_svr_model,
                                             optimize_parameters,
                                             cv = 5,
                                             refit = "r2",
                                             scoring=("r2","neg_mean_squared_error"),
                                             return_train_score = True)
    svr_model.fit(training_samples, training_labels)
    
    return svr_model

def model_optimizer(model_score, labels):
    #
    train_score, = plt.plot(model_score[0,:],'-',label=labels[0])
    test_score, = plt.plot(model_score[1,:],'-', label=labels[1])
    plt.legend(loc = "best")
    plt.show()


def test_model(svr_model, test_samples, test_labels):
    #
    test_model_score = []
    #
    y_pred = svr_model.predict(test_samples)
    r2 = metrics.r2_score(test_labels, y_pred)
    mse = metrics.mean_squared_error(test_labels, y_pred)
    test_model_score.extend([r2,mse])
    #
    return test_model_score

if __name__ == "__main__":
    #
    in_file_dir = r"C:\Users\xrui\Desktop\test_sklearn_ml\data"
    in_file_name = ('Pb_training_set.xlsx', 'Pb_test_set.xlsx',
                    'Ag_training_set.xlsx', 'Ag_test_set.xlsx',
                    'As_training_set.xlsx', 'As_test_set.xlsx',
                    'Cr_training_set.xlsx', 'Cr_test_set.xlsx',
                    'Cu_training_set.xlsx', 'Cu_test_set.xlsx',
                    'Zn_training_set.xlsx', 'Zn_test_set.xlsx')

    #
    for i in range(0,len(in_file_name),2):
        in_train_file_path = os.path.join(in_file_dir,in_file_name[i])
        in_test_file_path = os.path.join(in_file_dir,in_file_name[i + 1])
        X_train, X_test, y_train, y_test = get_training_test_set(in_train_file_path,
                                                                 "Sheet1","A1:BQJ54",
                                                                 in_test_file_path,
                                                                 "Sheet1","A1:BQJ18")
        #
##        X_train = dim_reduction(X_train, 15)
        #
        print(X_train.shape, y_train.shape)
        svr_model = train_model(X_train, y_train)
        #
        print("Best parameters set found:", svr_model.best_params_)

        mean_train_r2 = svr_model.cv_results_.get("mean_train_r2")[np.newaxis]
        mean_test_r2 = svr_model.cv_results_.get("mean_test_r2")[np.newaxis]
        mean_train_neg_mse = svr_model.cv_results_.get("mean_train_neg_mean_squared_error")[np.newaxis]
        mean_test_neg_mse = svr_model.cv_results_.get("mean_test_neg_mean_squared_error")[np.newaxis]
        #
        r2_score = np.concatenate((mean_train_r2, mean_test_r2), axis = 0)
        neg_mse_score = np.concatenate((mean_train_neg_mse, mean_test_neg_mse), axis = 0)
        model_optimizer(r2_score, ["train_r2","cv_r2"])
        model_optimizer(-neg_mse_score, ["train_mse","cv_mse"])
        #
        test_model_score = test_model(svr_model, X_test, y_test)
        print("r2_test:", test_model_score[0])
        print("mse_test:", test_model_score[1])
        #
        input("...:")


    
