import pandas as pd
import numpy as np 
import os
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
import pymap3d
import re
import cv2

############### anti-clock rorate point(x, y) w.r.t center(x, y) ###########################
def Nrotate(angleDeg, pointsx, pointsy, centerx, centery):
    angle = np.deg2rad(angleDeg)
    nRotatex = (pointsx-centerx)*math.cos(angle) - (pointsy-centery)*math.sin(angle) + centerx
    nRotatey = (pointsx-centerx)*math.sin(angle) + (pointsy-centery)*math.cos(angle) + centery
    return nRotatex, nRotatey



##################### return list of one column in .csv file ################################
def csv_reader(csv_path, column_name):
    df = pd.read_csv(csv_path)
    col_list = df[column_name]
    col_array = np.array(col_list)
    return col_array


##################### return dict from given .csv file ################################
def csv2dict(csv_path, keys_list):
    new_dict = {}
    if keys_list == 'all':
        df = pd.read_csv(csv_path)
        keys_list = list(df.columns)

    for col_name in keys_list:
        new_dict[col_name] = csv_reader(csv_path, col_name)
    return new_dict

##################### add aer items into dgps_dict ################################
def LLA2AER(dgps_dict):
        latitudeDeg_list = dgps_dict['HunterPosLat']
        longitudeDeg_list = dgps_dict['HunterPosLon']
        altitude_list = dgps_dict['nc_altitude']
        
        azimuth_list = []
        elevation_list = []
        srange_list = []

        # calculate azimuth & srange list
        n = len(latitudeDeg_list)

        for i in range(n):
            if i != n-1:
                azi, el, srange = pymap3d.geodetic2aer(lat=latitudeDeg_list[i+1], lon=longitudeDeg_list[i+1], h=altitude_list[i+1],
                                                        lat0=latitudeDeg_list[i], lon0=longitudeDeg_list[i], h0=altitude_list[i])
                azimuth_list.append(azi)
                elevation_list.append(el)
                srange_list.append(srange)
            else:
                azimuth_list.append(azi)
                elevation_list.append(el)
                srange_list.append(srange)

        dgps_dict['azimuth'] = azimuth_list
        dgps_dict['elevation'] = elevation_list
        dgps_dict['srange'] = srange_list

        return dgps_dict


############### interpolate y1_array by x1_array according to x0_array, y0_array #############
def interpolation(x0_arr, y0_arr, x1_arr, method='quadratic'):
    f=interpolate.interp1d(x0_arr,y0_arr,kind=method)
    y1_arr = f(x1_arr)
    return y1_arr


############## delete all points which are out of map range ################
def map_limit_filter(self, Idx_row, Idx_col, MapShape):

    min_row, min_col = 0, 0
    max_row, max_col = MapShape

    Idx_row = Idx_row[Idx_row < max_row]
    Idx_row = Idx_row[Idx_row > min_row]
    Idx_col = Idx_col[Idx_col < max_col]
    Idx_col = Idx_col[Idx_col > min_col]

    if len(Idx_row) > len(Idx_col):
        Idx_row = Idx_row[0:len(Idx_col)]
    elif len(Idx_row) < len(Idx_col):
        Idx_col = Idx_col[0:len(Idx_row)]

    return Idx_row, Idx_col



########## make dicts of Radar and DGPS measurement synchronization in time stamp ############
def Time_Stamp_Synchronization(long_dict, short_dict):
    '''
    here: 
    long_dict = dgps_dict
    short_dict = radar_dict
    '''

    long_time_list = long_dict['meta_time']
    short_time_list = short_dict['meta_time']

    new_index = []
    short_idx = 0

    #find nearest time stamp index in long_time_list corresponding to short_time_list
    for long_idx in range(len(long_time_list)):
        if short_idx == len(short_time_list):
            break
        
        if long_time_list[long_idx] <= short_time_list[short_idx] and \
        long_time_list[long_idx+1] > short_time_list[short_idx]:

            if short_time_list[short_idx] - long_time_list[long_idx] <= long_time_list[long_idx+1] > short_time_list[short_idx]:
                new_index.append(long_idx)
            else:
                new_index.append(long_idx+1)

            short_idx += 1
    new_long_dict = {}
    for key in long_dict.keys():
        new_long_dict[key] = long_dict[key][new_index]

    new_long_dict.update(short_dict)

    return new_long_dict, short_dict


############## find original row-index of car-center in rotated image ################
def find_line(img1, img2):

    ref_line = img2[-1, :, :]

    loss = np.zeros(img1.shape[0])

    for i in range(img1.shape[0]):

        loss[i] = np.sum((img1[i, :, :] - ref_line)**2)

    return np.argmin(loss)


############## delete special char in meta_time_abs ################

def del_char(str1):
    return re.sub(r"[^a-zA-Z0-9]","",str1)





############## move right image ################
# 将二维数组中值为1的元素向右移动15个单位
def move_right(img):
    # 复制原始数组，防止原地修改
    result = img.copy()
    # 找到所有值为1的像素点的坐标
    x, y = np.where(img == 255)
    # 将这些像素点向右移动15个单位
    y_new = y + 15
    # 找到越界的像素点，不进行移动
    valid_indices = np.where(y_new < img.shape[1])
    x = x[valid_indices]
    y = y[valid_indices]
    y_new = y_new[valid_indices]
    # 修改像素点的位置
    result[x, y] = 0
    result[x, y_new] = 255
    return result


################# segmented image post-processing ###################

def post_process(seg_img):

    img = cv2.convertScaleAbs(seg_img)

    # 找到所有白色连通域
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 找到图像中心点
    center_x, center_y = int(img.shape[1] / 2), int(img.shape[0] / 2)

    # 找到包含中心点的连通域
    center_contour = None
    for contour in contours:
        if cv2.pointPolygonTest(contour, (center_x, center_y), False) > 0:
            center_contour = contour

    # 创建一个与原始图像大小相同的黑色图像
    mask = np.zeros_like(img)

    # 在掩模上画出中心连通域的白色区域
    cv2.drawContours(mask, [center_contour], 0, 255, -1)

    # 将掩模应用于原始图像，并只保留中心连通域
    result = cv2.bitwise_and(img, mask)

    '''# 补洞
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #seg_img = cv2.morphologyEx(seg_img, cv2.MORPH_CLOSE, kernel1, iterations=10)
    #seg_img = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, kernel1, iterations=10)
    
    
    # 进行连通域分析，获取所有连通域的信息
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(seg_img)

    # 遍历所有连通域，找到面积小于10的黑色和白色连通域，分别进行填充
    for i in range(1, nlabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50000:
            if seg_img[int(centroids[i][1]), int(centroids[i][0])] == 0:
                seg_img[labels == i] = 255
            else:
                seg_img[labels == i] = 0


    #blur = cv2.GaussianBlur(seg_img, (5, 5), sigmaX=3, sigmaY=3)
    #edge = cv2.Canny(blur, 100, 255)

    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    #edge = cv2.dilate(edge, kernel, iterations=1)'''
    return result
