# -*-coding:utf-8 -*-

import numpy as np


def cos_similarity(vec1, vec2):
    #
    cos_theta = np.sum(vec1 * vec2) / \
                (np.sqrt(np.sum(vec1 ** 2)) * \
                 np.sqrt(np.sum(vec2 ** 2)))
    #
    return cos_theta


def hist_similarity(X1, X2):
    '''
    To evaluate the similarity of X1 and X2 by calculating their histogram's  Cosine similarity


    Note：X1 and X2 must be in the same range, which means that
        the maximum and minimum values of X1 are the same as those of X1.
        You can normalize firstly X1 and X2.
    '''
    hist_info1 = np.histogram(X1, bins = 50)
    hist_info2 = np.histogram(X2, bins = 50)
    #
    cos_theta = cos_similarity(hist_info1[0], hist_info2[0])
    #
    return cos_theta, hist_info1, hist_info2


if __name__ == "__main__":
    #
    pass
