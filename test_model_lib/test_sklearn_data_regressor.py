# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics

def get_train_test_set():
    #
    data, target = datasets.load_diabetes(return_X_y = True)
    #
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data, target, test_size = 0.3, random_state = 0)
    return X_train, X_test, y_train, y_test

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
        print(training_X.shape,cv_X.shape)
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
    X_train, X_test, y_train, y_test = get_train_test_set()
##    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #
    n_estimators = 550
    max_features = "sqrt" 
    training_model_score, rf_model = run_model(X_train, y_train, n_estimators, max_features)
    model_optimizer(training_model_score)
    #
##    test_model_score = test_model(rf_model, test_samples, test_labels)
##    model_optimizer(test_model_score)

    
