# -*- coding:utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import osr
from osgeo.gdalconst import *

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

def read_geotiff(in_file_Path):
    inDataset = gdal.Open(in_file_Path, GA_ReadOnly)
    x_dim = inDataset.RasterXSize
    y_dim = inDataset.RasterYSize
    band_count = inDataset.RasterCount
    if band_count == 1:
        band_data = inDataset.GetRasterBand(1)
        tiff_img = band_data.ReadAsArray(0, 0, x_dim, y_dim)
    else:
        tiff_img = -1
    return tiff_img, inDataset.GetGeoTransform(), inDataset.GetProjectionRef()

def write_geotiff(out_file_Path,img_data,geoTransform,spatialRef):
    geoTiffDriver = gdal.GetDriverByName("GTiff")
    outDataset = geoTiffDriver.Create(out_file_Path,img_data.shape[1],
                                      img_data.shape[0],1,gdal.GDT_Int16)
    outDataset.SetGeoTransform(geoTransform)
    outDataset.SetProjection(spatialRef.ExportToWkt())
    outDataset.GetRasterBand(1).WriteArray(img_data)

def get_samples(tiff_img):
    #
    samples = tiff_img.reshape(tiff_img.shape[0]*tiff_img.shape[1],1)
    return samples

def view_data(samples):
    #
    a = samples[:,-1] == 0
    label_1 = samples[samples[:,-1] == 0,:]
    label_2 = samples[samples[:,-1] == 1]
    plt.scatter(label_1[:,0],label_1[:,1], s=30, c='r', marker='o', label="non-water")
    plt.scatter(label_2[:,0],label_2[:,1], s=30, c='b', marker='x', label="water")
    plt.legend()
    # plt.xlim(0,255)
    # plt.ylim(0,255)
    plt.show()

def shuffle_data(data):
    #
    np.random.shuffle(data)
    X = data[:,:data.shape[1] - 1]
    y = data[:,data.shape[1] - 1:] # data[:,- 1]是一个一维数组，则程序将产生错误！
    return X, y

def sigmoid(z):
    return 1 / np.exp(-z)

def model(X, theta):
    return sigmoid(np.dot(X,theta.T))

def cost(X, y, theta):
    with np.errstate(divide = "ignore"):
        tmp_value = model(X, theta)
        left_value = np.where(tmp_value != 0, np.multiply(-y,np.log(tmp_value)), 0)
        right_value = np.where(tmp_value != 1, np.multiply(1-y,np.log(1 - tmp_value)),0)
    return np.sum(left_value - right_value) / len(X)

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X,theta) - y).ravel()
    for i in range(len(theta.ravel())):
        term = np.multiply(error,X[:,i])
        grad[0,i] = np.sum(term) / len(X)
    #
    return grad

def stop_criterion(stop_std, value, threshold):
    #
    if stop_std == "STOP_ITER":
        return value > threshold
    elif stop_std == "STOP_COST":
        return abs(value[-1]-value[-2]) < threshold
    else:
        return np.linalg.norm(value) < threshold

def descent(data, theta, batch_size, stop_std, thresh, alpha):
    #
    nrows = data.shape[0]
    X, y = shuffle_data(data)
    i = 0 # num iter 
    j = 0 #NO. Batch
    stop_iter = False
    grad = np.zeros(theta.shape)
    costs = [cost(X, y, theta)]
    #
    while not stop_iter:
        grad = gradient(X[j:j + batch_size], y[j:j + batch_size], theta)
        j += batch_size
        if j >= nrows:
            j = 0
            X, y = shuffle_data(data)
        theta -= alpha * grad
        costs.append(cost(X, y, theta))
        i += 1

        if stop_std == "STOP_ITER":
            value = 1
        elif stop_std == "STOP_COST":
            value = costs
        elif stop_std == "STOP_GRAD":
            value = grad
        #
        stop_iter = stop_criterion(stop_std, value, thresh)
        #
    return theta, i -1, costs, grad

def run_model0(data, theta, batch_size, stop_std, thresh, alpha):
    #
    theta, num_iter, costs, grad = descent(data, theta, batch_size, 
                                           stop_std, thresh, alpha)
    print(num_iter, costs, grad)


###
def create_feature_importance_plot(feature_name, feature_importance):
    #
    plt.bar(feature_name, feature_importance)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Importance of Features")
    plt.show()

def create_parameters_optimizer_plot(optimizer_range, parameters_list):
    #
    accuracy_plot, = plt.plot(optimizer_range, parameters_list[0])
    precision_plot, = plt.plot(optimizer_range, parameters_list[1])
    recall_plot, = plt.plot(optimizer_range, parameters_list[2])
    oob_score_plot, = plt.plot(optimizer_range, parameters_list[2])
    #
    plt.legend([accuracy_plot, precision_plot,recall_plot,  oob_score_plot],
               ['accuracy', 'precision','recall', "oob score"], loc = 'best')
    plt.show()


def run_model(train_samples, train_labels, test_samples, test_labels, n_estimators=10, min_samples_split=2):
    #
    rf_model = RandomForestClassifier(n_estimators= n_estimators,criterion="gini",
                                    max_features="auto",max_depth=None,
                                    min_samples_split=min_samples_split,min_samples_leaf=1,
                                    min_weight_fraction_leaf=0,
                                    max_leaf_nodes=None,
                                    min_impurity_decrease=0,bootstrap=True,
                                    oob_score=True,n_jobs=1,random_state=None,
                                    verbose=0,warm_start=False)
    rf_model.fit(train_samples,train_labels)
    y_pred = rf_model.predict(test_samples)
    accuracy = metrics.accuracy_score(test_labels,y_pred)
    precision = metrics.precision_score(test_labels,y_pred)
    recall = metrics.recall_score(test_labels,y_pred)
    #
    return accuracy, precision, recall, rf_model

###

water_x1 = get_samples(read_geotiff(r"D:\lr_tm_data\water_img1.tif")[0])       
water_x2 = get_samples(read_geotiff(r"D:\lr_tm_data\water_img2.tif")[0])          
water_x = np.unique(np.hstack((water_x1,water_x2)), axis=0)   
label1 = np.full([water_x.shape[0],1], 1)
x1 = np.hstack((water_x, label1))
#
non_water_x1 = get_samples(read_geotiff(r"D:\lr_tm_data\non_water_img1.tif")[0])       
non_water_x2 = get_samples(read_geotiff(r"D:\lr_tm_data\non_water_img2.tif")[0])  
non_water_x = np.unique(np.hstack((non_water_x1,non_water_x2)), axis=0)  
label2 = np.full([non_water_x.shape[0],1], 0)
tmp_x2 = np.hstack((non_water_x, label2))
np.random.shuffle(tmp_x2)
x2 = tmp_x2[:x1.shape[0] + 5,:]
samples = np.vstack((x1,x2))
np.random.shuffle(samples)           
               
# view_data(samples)
# samples = np.hstack((np.full([samples.shape[0],1],1),samples))
# theta = np.zeros([1,samples.shape[1] - 1])
# batch_size = 100
# run_model(samples, theta, batch_size, "STOP_ITER", thresh=5000, alpha=0.0001)

split_index = int(np.around(0.7*samples.shape[0]))
train_samples = samples[:split_index,:-1]
train_labels = samples[:split_index,-1]
test_samples = samples[split_index:,:-1]
test_labels = samples[split_index:,-1]
parameters_list = [[],[],[],[]]

accuracy, precision, recall, model = run_model(train_samples, train_labels, test_samples,  test_labels, 50, 2)
# optimizer_range = range(5,100,5)
# for i in optimizer_range:
#     accuracy, precision, recall, model = run_model(train_samples, train_labels, test_samples, test_labels, i,2)
#     parameters_list[0].append(accuracy)
#     parameters_list[1].append(precision)
#     parameters_list[2].append(recall)
#     parameters_list[3].append(model.oob_score_)

# create_parameters_optimizer_plot(optimizer_range, parameters_list)
# feature_name = ["Band2","Band6"]
# create_feature_importance_plot(feature_name, model.feature_importances_)

#
in_file_path = [r"E:\lr_tm_data\L5123032_03220110608_B10.TIF",
                r"E:\lr_tm_data\L5123032_03220110608_B60.TIF"]
b1,a,_ = read_geotiff(in_file_path[0])
b6,_,_ = read_geotiff(in_file_path[1])
row = b1.shape[0]
col = b1.shape[1]
X = None
b = osr.SpatialReference()
b.SetWellKnownGeogCS("WGS84")
if b1.shape == b6.shape:
    b1 = b1.reshape([row*col,1])
    b6 = b6.reshape([row*col,1])
    X = np.hstack((b1,b6))
    rt = model.predict(X)
    rt_img = rt.reshape([row,col])
    #
    print(rt_img.max,rt_img.min)
    #
    write_geotiff(r"C:\Users\xrui\Desktop\ab.tif",rt_img,a,b)
    rt_img = None