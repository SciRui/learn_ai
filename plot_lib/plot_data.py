# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_histogram_data(data, title = "fig1"):
    #
    fig, ax = plt.subplots()
    plt.title(title)
    hist_info = ax.hist(data, bins = 50, edgecolor="black")
    #
    return hist_info



if __name__ == "__main__":
    #
    test_data = np.random.randint(12,58)
    print(test_data)
    #
    plt = plot_histogram_data(test_data)
    plt.show()
