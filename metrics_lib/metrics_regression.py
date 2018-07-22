# -*- coding:utf-8 -*-


import numpy as np


def fitting_line_params(true_value, prediction_value):
    #
    x = true_value
    y = prediction_value
    n = x.shape[0]
    #
    slope = (np.sum(x * y) - (np.sum(x) * np.sum(y))/n)/ \
            (np.sum(x ** 2) - np.sum(x) ** 2/n)
    #
    intercept = np.mean(prediction_value) - slope * np.mean(prediction_value)
    #
    return slope, intercept


if __name__ == "__main__":
    pass
