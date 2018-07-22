# -*- coding:utf-8 -*-

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
import time
from osgeo import gdal
from osgeo.gdalconst import *


def filling_scheme(file_path, sub_size = 1024):
    '''
    '''
    dataset = gdal.Open(file_path, GA_ReadOnly)
    img_width, img_height = dataset.RasterXSize, dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    #
    count_nonzero_upper_img = []
    count_nonzero_lower_img = []
    count_nonzero_left_img = []
    count_nonzero_right_img = []
    filling_scheme = []
    #
    row_col_index = [[],[]]
    #
    for j in range(0, img_width, sub_size):
        if img_width - j >= sub_size:
            upper_img = band.ReadAsArray(j, 0, sub_size, sub_size)
            lower_img = band.ReadAsArray(j, img_height - sub_size,
                                            sub_size, sub_size)
            count_nonzero_upper_img.append(np.count_nonzero(upper_img))
            count_nonzero_lower_img.append(np.count_nonzero(lower_img))
        else:
            right_missing_img1 = band.ReadAsArray(j, 0, img_width - j, sub_size)
            right_missing_img2 = band.ReadAsArray(j, img_height - sub_size,
                                                     img_width - j, sub_size)
            count_nonzero_upper_img.append(np.count_nonzero(right_missing_img1))
            count_nonzero_lower_img.append(np.count_nonzero(right_missing_img2))
            #
            break
        #
        row_col_index[0].append(j)
    #
    for i in range(0, img_height, sub_size):
        if img_height - i >= sub_size:
            left_img = band.ReadAsArray(0, i, sub_size, sub_size)
            right_img = band.ReadAsArray(img_width - sub_size, i,
                                            sub_size, sub_size)
            count_nonzero_left_img.append(np.count_nonzero(left_img))
            count_nonzero_right_img.append(np.count_nonzero(right_img))
        else:
            lower_missing_img1 = band.ReadAsArray(0, i, img_height - i, sub_size)
            lower_missing_img2 = band.ReadAsArray(img_width - sub_size, i,
                                                     sub_size, img_height - i)
            count_nonzero_left_img.append(np.count_nonzero(lower_missing_img1))
            count_nonzero_right_img.append(np.count_nonzero(lower_missing_img2))
            #
            break
        #
        row_col_index[1].append(i)
    #
    if sum(count_nonzero_upper_img) >= sum(count_nonzero_lower_img):
        filling_scheme.append("Lower")
    else:
        filling_scheme.append("Upper")
    #
    if sum(count_nonzero_left_img) >= sum(count_nonzero_right_img):
        filling_scheme.append("Right")
    else:
        filling_scheme.append("Left")
    #
    return filling_scheme, row_col_index

def img_clipping(file_path, out_dir, sub_size = 1024):
    '''
    '''
    #
    print("程序正在运行，这可能需要一些时间，请耐心等待......")
    #
    with rasterio.open(file_path,"r") as in_dataset:
        img_width, img_height = in_dataset.width, in_dataset.height
        #
##        res = 0.911183
##        transform1 = from_origin(117.10 - res / 2, 40.32 + res / 2, res, res)
        #
        surplus_size_x = img_width % sub_size
        surplus_size_y = img_height % sub_size
        #
        print(surplus_size_x, surplus_size_y)
        k = 0
        #
        for j in range(0, img_width, sub_size):
            for i in range(0, img_height, sub_size):
                #
                ### Extracted sub image by clipping image with a fifty percent overlapping
                #
                if img_width - j >= (sub_size * 3) / 2:
                    #
                    overlap_img = in_dataset.read(window=Window(j, i + sub_size / 2,
                                                                sub_size, sub_size)) # overlap:bgr 50%
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_overlap_col_" + str(k) + ".tif")
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = overlap_img.dtype) as out_dataset:
                        out_dataset.write(overlap_img) # bgr
                    #
                    k += 1
                #
                if img_height - i >= (sub_size * 3) / 2:
                    #
                    overlap_img = in_dataset.read(window=Window(j + sub_size / 2, i,
                                                                sub_size, sub_size)) # overlap:bgr 50%
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_overlap_row_" + str(k) + ".tif")
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = overlap_img.dtype) as out_dataset:
                        out_dataset.write(overlap_img) # bgr
                    #
                    k += 1
                #
                if img_width - j >= (sub_size * 3) / 2 and img_height - i >= (sub_size * 3) / 2:
                    #
                    overlap_img = in_dataset.read(window=Window(j + sub_size / 2,
                                                                i + sub_size / 2,
                                                                sub_size, sub_size)) # overlap:bgr 50%
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_overlap_col_row_" + str(k) + ".tif")
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = overlap_img.dtype) as out_dataset:
                        out_dataset.write(overlap_img) # bgr
                    #
                    k += 1
                #
                ### Extracted sub image by clipping image specified size
                #
                if img_width - j >= sub_size and img_height - i >= sub_size:
                    img = in_dataset.read(window=Window(j, i, sub_size, sub_size)) # bgr
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_" + str(k) + ".tif")

                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = img.dtype) as out_dataset:
                        #
                        out_dataset.write(img) # bgr

                    #
                    k += 1
                #
                ### Filled space pixel
                #
                if img_height - i < sub_size and img_width - j >= sub_size:
                    missing_img = in_dataset.read(window=Window(j, i,
                                                                sub_size, 
                                                                surplus_size_y))  #In the direction of x and y, the missing pixels is filled with 0
                    #
                    print(missing_img.shape)
                    filling_arr = np.zeros([missing_img.shape[0], sub_size - surplus_size_y, missing_img.shape[2]],
                                           dtype = missing_img.dtype)
                    missing_img = np.concatenate((missing_img, filling_arr), axis = 1)
                    print(missing_img.shape)
                    #
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_missing_row_" + str(k) + ".tif")
                    print(out_path)
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = missing_img.dtype) as out_dataset:
                        out_dataset.write(missing_img) # bgr
                    #
                    k += 1
                    #
                if img_height - i >= sub_size and img_width - j < sub_size:
                    missing_img = in_dataset.read(window=Window(j, i,
                                                                surplus_size_x, 
                                                                sub_size))  #In the direction of  x, the missing pixels is filled with 0
                    #
                    filling_arr = np.zeros([missing_img.shape[0], missing_img.shape[1], sub_size - surplus_size_x],
                                           dtype = missing_img.dtype)
                    missing_img = np.concatenate((missing_img, filling_arr), axis = 2)
                    #
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_missing_col_" + str(k) + ".tif")
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = missing_img.dtype) as out_dataset:
                        out_dataset.write(missing_img) # bgr
                    #
                    k += 1
                    #
                if img_height - i < sub_size  and img_width - j < sub_size:
                    missing_img = in_dataset.read(window=Window(j, i,
                                                                surplus_size_x,
                                                                surplus_size_y))  #In the direction of y, the missing pixels is filled with 0
                    #
                    filling_arr1 = np.zeros([missing_img.shape[0], sub_size - surplus_size_y, missing_img.shape[2]],
                                            dtype = missing_img.dtype)
                    missing_img = np.concatenate((missing_img, filling_arr1), axis = 1)
                    filling_arr2 = np.zeros([missing_img.shape[0], missing_img.shape[1], sub_size - surplus_size_x],
                                            dtype = missing_img.dtype)
                    missing_img = np.concatenate((missing_img, filling_arr2), axis = 2)
                    #
                    out_path = os.path.join(out_dir,
                                            str(i) + "_" + str(j) + "_img_missing_col_row" + str(k) + ".tif")
                    #
                    with rasterio.open(out_path, 'w',
                                       driver="GTiff",
                                       height = sub_size,
                                       width = sub_size,
                                       count = 3,
                                       dtype = missing_img.dtype) as out_dataset:
                        out_dataset.write(missing_img) # bgr
                    #
                    k += 1
                    #
    #
    print("程序运行结束！成功分割图片：" + file_path + "为" + str(k) + "个" +
          str(sub_size) + "*" + str(sub_size) + "大小的子图片")

if __name__ == "__main__":
    #
    start_time = time.clock()
    #
    file_path = r"D:\weitu\download\盘锦市\Level17\盘锦市.tif"
    img_clipping(file_path, r"D:\Data\test_ssd_img_clipping2", sub_size=8192)
    #
##    file_path1 = r"D:\重污染企业特征样例图\熊儿寨乡\Level17\熊儿寨乡.tif"
##    img_clipping(file_path1, r"D:\Data\test_ssd_img_clipping_small", sub_size=2048)
    #
    end_time = time.clock()
    print("程序耗时（单位：s）：", end_time - start_time)
    #
##    img_dir = r"G:\Data\Data_Liao"
##    sub_img_dir = r"G:\Data\Data_Liao_sub_img"
##    city_names_tuple = ("鞍山市","阜新市","锦州市","辽阳市","盘锦市","沈阳市","铁岭市","营口市")
##    img_level = "Level17"
##    #
##    sub_size = 8192  # 512  1024  2048  4096  8192
##    #
##    for city_name in city_names_tuple:
##        #
##        ## ---记录程序开始运行时间---
##        start_time = time.clock()
##        #
##        for img_path in os.listdir(os.path.join(img_dir, city_name, img_level)):
##            if img_path.endswith(".tif"):
##                #  
##                sub_img_path = os.path.join(sub_img_dir, city_name)
####                os.mkdir(sub_img_path)
####                img_clipping(img_path, sub_img_path, sub_size=sub_size)
##            else:
##                continue
##        #
##        end_time = time.clock()
##        print("程序耗时（单位：S）：", end_time - start_time)
