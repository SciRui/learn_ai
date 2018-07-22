# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics

def get_train_test_set():
    #
    diabetes = datasets.load_diabetes()
    X, y = diabetes.data, diabetes.target
    #
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

def train_model(training_samples, training_labels):
    #
    training_model_score = []
    #
    training_samples = training_dataset[:,:-1]
    training_labels = training_dataset[:,-1]
    #
    init_model = svm.LinearSVR()
    #    
    optimize_parameters = {"C":[1.,5.,10.,20.,25,35.,40.,50.],
                           "epsilon":[0.001,0.005,0.1,0.3,0.5,0.7,0.9],
                           "loss":["epsilon_insensitive","squared_epsilon_insensitive"],
                           "max_iter":[500,1000,2000]}
    #
    model = model_selection.GridSearchCV(init_model,
                                         optimize_parameters,
                                         cv = 10,
                                         refit = "r2",
                                         scoring=("r2","neg_mean_squared_error"),
                                         return_train_score = True)
    model.fit(training_samples, training_labels)
    
##    scores = model_selection.cross_validate(svr_model,
##                                            training_samples,
##                                            training_labels,
##                                            cv = 10,
##                                            scoring=("r2","mean_squared_error"),
##                                            return_train_score = True)
##    print("train_r2:", svr_model.cv_results_['train_r2'].mean())
##    print("cv_r2:", svr_model.cv_results_['test_r2'].mean())
##    print("train_neg_mean_squared_error'scores", svr_model.cv_results_["train_neg_mean_squared_error"].mean())
##    print("cv_neg_mean_squared_error'scores", svr_model.cv_results_["test_neg_mean_squared_error"].mean())
##    #
##    kf = model_selection.KFold(n_splits = 10, random_state=0)
##    kf.get_n_splits(training_samples)
##    for training_index, cv_index in kf.split(training_samples):
##        training_X = training_samples[training_index]
##        training_y = training_labels[training_index]
##        #
##        cv_X = training_samples[cv_index]
##        cv_y = training_labels[cv_index]
##        #
##        svr_model.fit(training_X, training_y)
##        #
##        y_pred = svr_model.predict(cv_X)
##        #
##        r2 = svr_model.score(cv_X, cv_y)
##        mse = metrics.mean_squared_error(cv_y, y_pred)
##
##        training_model_score.append([r2,mse])

    return model

def model_optimizer(model_score, label):
    #
    train_score, = plt.plot(model_score[0,:],'o-',label=label[0])
    test_score, = plt.plot(model_score[1,:],'o-', label=label[0])
    plt.legend(loc = "best")
    plt.show()


def test_model(model, test_dataset):
    #
    test_model_score = []
    #
    test_samples = test_dataset[:,:-1]
    test_labels = test_dataset[:,-1]
    #
    y_pred = model.predict(test_samples)
    r2 = metrics.r2_score(test_labels, y_pred)
    mse = metrics.mean_squared_error(test_labels, y_pred)
    test_model_score.extend([r2,mse])
    #
    return test_model_score

if __name__ == "__main__":
    #
    X_train, X_test, y_train, y_test = get_train_test_set()
    #
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
    model_optimizer(neg_mse_score, ["train_mse","cv_mse"])
    #
    test_model_score = test_model(svr_model, X_test, y_test)
    print("r2_test:", test_model_score[0])
    print("mse_test:", test_model_score[1])
##    #
##    training_model_score, svr_model = run_model(X_train, y_train)
##    #
##    test_model_score = test_model(svr_model, X_test, y_test)
##    model_optimizer(training_model_score)
##    #
##    print("mean_r2_cv:", np.mean(training_model_score[:,0]), '\t',
##          'r2_test:', test_model_score[0])
##    print("mean_mse_cv:", np.mean(training_model_score[:,1]), '\t',
##          "mse_test:", test_model_score[1])

    
