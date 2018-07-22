# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from PreprocessData import PreprocessData_MNIST as mnist
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics


def train_model(training_samples, training_labels):
    #
    init_svr_model = svm.SVR()
    #    
##    optimize_parameters = [{"kernel":["linear"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9]},
##                           {"kernel":["rbf"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9],
##                            "gamma":[0.001,0.005,0.1,0.5,1,5,10,15,20,25,30,35,40,50]}]
    #
    # optimize_parameters = {"C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
    #                        "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9],
    #                        "gamma":[0.001,0.005,0.1,0.5,1,5,10,15,20,25,30,35,40,50]}
    #
    optimize_parameters = {"C":[1,30],
                           "epsilon":[0.001,0.9],
                           "gamma":[0.001,30]}
##    optimize_parameters = {"kernel":["linear"],
##                            "C":[1,2,5,7,10,12,15,17,20,22,25,30,40,45,50],
##                            "epsilon":[0.001,0.002,0.005,0.007,0.1,0.3,0.5,0.7,0.9]}
    #
    svr_model = model_selection.GridSearchCV(init_svr_model,
                                             optimize_parameters,
                                             cv = 3,
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
    y_pred = svr_model.predict(test_samples)
    r2 = metrics.r2_score(test_labels, y_pred)
    mse = metrics.mean_squared_error(test_labels, y_pred)
    #
    return r2, mse

if __name__ == "__main__":
    #
    train_X, train_y = mnist.train_samples, mnist.train_labels
    test_X, test_y = mnist.test_samples, mnist.test_labels
    #
    # print(train_X.shape, len(train_y),test_X.shape, len(test_y))
    # input('...')
    #
    svr_model = train_model(train_X, train_y)
    #
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
    r2, mse = test_model(svr_model, test_X, test_y)
    print("r2_test:", r2)
    print("mse_test:", mse)