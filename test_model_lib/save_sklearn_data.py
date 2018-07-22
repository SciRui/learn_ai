# -*- coding:utf-8 -*-

import numpy as np
import openpyxl as opxl
from sklearn import datasets


def get_dataset(name_dataset):
    #
    dataset = None
    if name_dataset == "diabetes":
        tmp_dataset = datasets.load_diabetes()
        dataset_arr = np.hstack((tmp_dataset.data,
                                 tmp_dataset.target[np.newaxis].T))
        dataset = dataset_arr[dataset_arr[:,-1].argsort()]
    elif name_dataset == "boston":
        tmp_dataset = datasets.load_boston()
        dataset_arr = np.hstack((tmp_dataset.data,
                                 tmp_dataset.target[np.newaxis].T))
        dataset = dataset_arr[dataset_arr[:,-1].argsort()]
    else:
        raise("Dataset is not be found!")
    #
    return dataset

def write_excel(data, file_path):
    #
    wb = opxl.Workbook()
    ws = wb.active
    #
    for line_data in data:
        ws.append(tuple(line_data))
    #
    wb.save(file_path)


if __name__ == "__main__":
    #
    diabetes_dataset = get_dataset("diabetes")
    boston_dataset = get_dataset("boston")
    #
    out_file_path1 = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\sklearn_diabetes1.xlsx"
    out_file_path2 = r"E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\sklearn_boston1.xlsx"
    write_excel(diabetes_dataset,out_file_path1)
    write_excel(boston_dataset,out_file_path2)
    #
    print("Finish!")
