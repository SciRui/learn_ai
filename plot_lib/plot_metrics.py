# -*- coding:utf-8 -*-

import sys

sys.path.append("..\\")

import numpy as np
import matplotlib.pyplot as plt
from  metrics_lib import metrics_regression
from sklearn import metrics
from sklearn import model_selection

    
def plot_model_score(y_pred_train_val, train_val_scores,
                     y_pred_test = None,test_scores = None,
                     y_pred_all = None, all_scores = None,
                     tick_range = (0,50)):
    #
    train_val_scores = [np.max(train_val_scores[0]), np.max(train_val_scores[1]),
                        np.max(train_val_scores[2]), np.max(train_val_scores[3])]
    x = np.arange(tick_range[0],tick_range[1]) # one divide-by one line
    #
    fig = plt.figure(facecolor="lightgray", figsize = (6,6))
    ###
    #
    ax1 = fig.add_subplot(221)
    training_one_by_one_line, = ax1.plot(x,x, color = "gray", linestyle = ":")
    #
    train_slope, train_intercept = metrics_regression.fitting_line_params(y_pred_train_val[2], y_pred_train_val[4])
    pred_training_y = train_slope * x + train_intercept
    #
    training_func = "y=" + str(round(train_slope,2)) + "*x" + "+" + str(round(train_intercept,2))
    training_true_pred_data = ax1.scatter(y_pred_train_val[2],y_pred_train_val[4], color = "", edgecolors = "black", marker = "o")
    training_fitting_line, = ax1.plot(x,pred_training_y, color = "red", linestyle = "-")
    ax1.set_title("$Training:R^2=$" + str(round(train_val_scores[0],4)), fontsize=12)
    ax1.set_xlabel("Measured Values(x)")
    ax1.set_ylabel("Predicted Values(y)")
    ax1.set_xlim(0,50)
    ax1.set_ylim(0,50)
    ax1.legend(handles = [training_one_by_one_line, training_true_pred_data, training_fitting_line],
                labels = ["y = x", "Data", training_func],
                loc = "upper left",
                prop={"size":8})
    ###
    #
    ax2 = fig.add_subplot(222)
    val_one_by_one_line, = ax2.plot(x,x, color = "gray", linestyle = ":")
    #
    val_slope, val_intercept = metrics_regression.fitting_line_params(y_pred_train_val[3], y_pred_train_val[5])
    pred_val_y = val_slope * x + val_intercept
    #
    val_func = "y=" + str(round(val_slope,2)) + "*x" + "+" + str(round(val_intercept,2))
    val_true_pred_data = ax2.scatter(y_pred_train_val[3],y_pred_train_val[5], color = "", edgecolors = "black", marker = "o")
    val_fitting_line, = ax2.plot(x,pred_val_y, color = "green", linestyle = "-")
    ax2.set_title( "$Validation: R^2 = $"+ str(round(train_val_scores[1],4)),fontsize=12)
    ax2.set_xlabel("Measured Values(x)")
    ax2.set_ylabel("Predicted Values(y)")
    ax2.set_xlim(0,50)
    ax2.set_ylim(0,50)
    ax2.legend(handles = [val_one_by_one_line, val_true_pred_data, val_fitting_line],
               labels = ["y = x", "Data", val_func],
               loc = "upper left",
               prop={"size":8})
    ###
    #
    ax3 = fig.add_subplot(223)
    test_one_by_one_line, = ax3.plot(x,x, color = "gray", linestyle = ":")
    #
    if y_pred_test != None and test_scores != None:
        test_slope, test_intercept = metrics_regression.fitting_line_params(y_pred_test[1], y_pred_test[2])
        pred_test_y = test_slope * x + test_intercept
        test_func = "y=" + str(round(test_slope,2)) + "*x " + "+" + str(round(test_intercept,2))
        #
        test_true_pred_data = ax3.scatter(y_pred_test[1],y_pred_test[2], color = "", edgecolors = "black", marker = "o")
        test_fitting_line, = ax3.plot(x,pred_test_y, color = "blue", linestyle = "-")
        ax3.set_title("$Test: R^2 = $" + str(round(test_scores[0],4)), fontsize=12)
        ax3.set_xlabel("Measured Values(x)")
        ax3.set_ylabel("Predicted Values(y)")
        ax3.set_xlim(0,50)
        ax3.set_ylim(0,50)
        ax3.legend(handles = [test_one_by_one_line, test_true_pred_data, test_fitting_line],
                   labels = ["y = x", "Data", test_func],
                   loc = "upper left",
                   prop={"size":8})
    ###
    #
    ax4 = fig.add_subplot(224)
    all_one_by_one_line, = ax4.plot(x,x, color = "gray", linestyle = ":")
    #
    if y_pred_all != None and all_scores != None:
        all_slope, all_intercept = metrics_regression.fitting_line_params(y_pred_all[1], y_pred_all[2])
        pred_all_y = all_slope * x + all_intercept
        all_func = "y=" + str(round(all_slope, 2)) + "*x" + "+" + str(round(all_intercept,2))
        #
        all_true_pred_data = ax4.scatter(y_pred_all[1],y_pred_all[2],  color = "", edgecolors = "black", marker = "o")
        all_fitting_line, = ax4.plot(x,pred_all_y, color = "black", linestyle = "-")
        ax4.set_title("$All: R^2 = $"+ str(round(all_scores[0],4)), fontsize=12)
        ax4.set_xlabel("Measured Values(x)")
        ax4.set_ylabel("Predicted Values(y)")
        ax4.set_xlim(0,50)
        ax4.set_ylim(0,50)
        ax4.legend(handles = [all_one_by_one_line, all_true_pred_data, all_fitting_line],
                   labels = ["y = x", "Data", all_func],
                   loc = "upper left",
                   prop={"size":8})
    ###
    #
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X, y, cv, train_sizes, scoring = "neg_mean_squared_error"):
    #
    train_sizes, train_scores, val_scores = model_selection.learning_curve(model,X, y, cv,
                                                                           train_sizes = train_sizes,
                                                                           scoring = scoring,
                                                                           n_jobs = -1)
    #
    fig = plt.figure()
    #
    ax = fig.add_subplot(111)
    ax.set_title("Learning curve", fontsize=12)
    ax.set_xlabel("Training size")
    ax.set_ylabel("Learning score")
    train_scores_line, = ax.plot(train_sizes, np.mean(train_scores, axis = 1), "o-", color = "red")
    valid_scores_line, = ax.plot(train_sizes, np.mean(val_scores, axis = 1), "o-", color = "green")
    ax.legend(handles = [train_scores_line, valid_scores_line],
              labels = ["Training score", "Cross validation score"],
              loc = "upper right",
              prop={"size":8})
    ###
    #
    plt.tight_layout()
    plt.show()
    

def plot_learning_curve_1(model, X,  y, n_splits = 3, train_size = 0.8, test_size = 0.2, 
                         scoring = "neg_mean_squared_error", train_sizes=np.linspace(0.1, 1.0, 5)):

    train_scores = []
    cv_scores = []
    #
    subset_sizes = X.shape[0] * train_sizes
    subset_sizes = subset_sizes.astype(int)
    cv = model_selection.ShuffleSplit(n_splits = n_splits,
                                      train_size = train_size,
                                      test_size = test_size)
    
    for m in subset_sizes:
        X_train_cv = X[:m]
        y_train_cv = y[:m]
        #
        scores = model_selection.cross_validate(model,
                                                X_train_cv,
                                                y_train_cv,
                                                scoring=scoring,
                                                cv=cv,
                                                n_jobs=-1,
                                                return_train_score = True)
        #
        tain_score = scores.get("train_score")
        cv_score = scores.get("test_score")
        #
        train_scores.append(tain_score)
        cv_scores.append(cv_score)
    #
    train_scores = np.mean(np.array(train_scores), axis=1)
    cv_scores = np.mean(np.array(cv_scores), axis=1)
    #
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_xscale("log")
    ax.set_xlabel("Training size")
    ax.set_ylabel('Score')
    ax.set_title('Learning curve')
    training_score, = ax.plot(subset_sizes,train_scores, lw=2)
    cv_score, = ax.plot(subset_sizes,cv_scores, lw=2)
    ax.legend(handles = [training_score, cv_score],
              labels = ["Training score", "Cross validation score"],
              loc = "upper right",
              prop={"size":8})
    ###
    #
    plt.tight_layout()
    plt.show()


def plot_valdation_curve(model, X, y, cv, train_val_params, scoring = "neg_mean_squared_error"):
    #
    for key, value in train_val_params[0].items():
        train_scores, valid_scores = model_selection.validation_curve(model,X,y,cv = cv,
                                                                   param_name = key,
                                                                   param_range = value,
                                                                   scoring = scoring,
                                                                   n_jobs = -1)
        #
        fig = plt.figure()
        #
        ax = fig.add_subplot(111)
        ax.set_title("Validation curve", fontsize=12)
        ax.set_xlabel(key)
        ax.set_ylabel(scoring)
        train_scores_line, = ax.plot(value,np.mean(train_scores, axis = 1), "o-", color = "red")
        valid_scores_line, = ax.plot(value,np.mean(valid_scores, axis = 1), "o-", color = "green")
        ax.legend(handles = [train_scores_line, valid_scores_line],
                  labels = ["Training score", "Cross validation score"],
                  loc = "upper right",
                  prop={"size":8})
    ###
    #
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    #
    pass
