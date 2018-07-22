# -*-coding:utf-8 -*-

import numpy as np
import openpyxl as opxl
from openpyxl import Workbook
from osgeo import gdal
from osgeo.gdalconst import *

def len_to_angle(length, upper_lat, lower_lat, left_lon, right_lon):
    #
    
    #
    angle = length/((2*np.pi*6371004)/360)
    #
    return angle

def read_tiff(in_file_path):
    #
    in_dataset = gdal.Open(in_file_path, GA_ReadOnly)
    band_data = in_dataset.GetRasterBand(1)
    img = band_data.ReadAsArray(0, 0,
                                in_dataset.RasterXSize,
                                in_dataset.RasterYSize)
    #
    return img

def coordinate_to_xy(coordinate_range, spatial_resolution, coordinates_arr):
    #
    pass

def get_pixel_value_xy(img, row_col_arr, save_path):
    #
    line_data = []
    #
    wb = Workbook(guess_types = True)
    ws = wb["Sheet1"]
    #
    for row_col in row_col_arr:
        for i in range(0,len(row_col),2):
            line_data.append(img[i,i+1])
        ws.append(line_data)
    #
    wb.save(save_path)
    
if __name__ == "__main__":
    #
    upper_lat = 41.2912408
    lower_lat = 39.3354545
    left_lon = 115.2386412
    right_lon = 118.1262263
    #
    angle = len_to_angle(30, upper_lat, lower_lat, left_lon, right_lon)
    print(angle,'\n', (upper_lat - lower_lat)/angle,'\n', (right_lon - left_lon)/angle)
