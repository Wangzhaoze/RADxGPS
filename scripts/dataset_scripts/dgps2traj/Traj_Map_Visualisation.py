
import numpy as np 
import os
from utils import *
import pymap3d
from TrajMap import DGPSGridMap
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import trange
import sys 
sys.path.append("..") 

from dataset_scripts.dgps2seg.DGPS2Sat import dgps2sat
path = "./dgps.csv"


###########################################################################
########################### parameters ####################################
###########################################################################

latitudeDeg_list = csv_reader(path, 'nc_latitudeDeg')
longitudeDeg_list = csv_reader(path, 'nc_longitudeDeg')
altitude_list = csv_reader(path, 'nc_altitude')
metatime_list = csv_reader(path, 'meta_time')

azimuth_list = []
theta_list = []
elevation_list = []
srange_list = []
delta_x = []
delta_y = []



###########################################################################
########## csv data procecssing & mapping from LLA/geodetic to aer ########
###########################################################################

data_length = len(metatime_list)
sample_duration = 10
for i in range(0, data_length-10, 10):

    azi, el, srange = pymap3d.geodetic2aer(lat=latitudeDeg_list[i+10], 
                                            lon=longitudeDeg_list[i+10], 
                                            h=altitude_list[i+10],
                                            lat0=latitudeDeg_list[i], 
                                            lon0=longitudeDeg_list[i], 
                                            h0=altitude_list[i])
    azimuth_list.append(azi)
    theta = np.deg2rad(-(azi-90))
    theta_list.append(theta)   
    elevation_list.append(el)
    srange_list.append(srange)
    delta_x.append(srange*np.cos(theta))
    delta_y.append(srange*np.sin(theta))




###########################################################################
######################## Grid Map Processing ##############################
###########################################################################

Grid_Map_Album = []
num_maps = 10

tra_map = DGPSGridMap()
for i in trange(num_maps):
    #算后800点
    delta_x_array = np.array(delta_x[i:i+1000])
    delta_y_array = np.array(delta_y[i:i+1000])

    #累加delta, 得绝对坐标
    delta_x_array = np.cumsum(delta_x_array)
    delta_x_array = delta_x_array + 500
    delta_y_array = np.cumsum(delta_y_array)

    #转至正向
    points_x_array, points_y_array = Nrotate(azimuth_list[i], 
                                            delta_x_array, 
                                            delta_y_array,
                                            500,
                                            0)


    #转到图像坐标系(500, 0)---->(1000, 500)
    Idx_y = points_x_array
    Idx_x = 1000 - points_y_array

    #加上初始点
    Idx_x = np.append(1000-1, Idx_x)
    Idx_y = np.append(500-1, Idx_y)

    #坐标插值化整化整
    Idx_row = np.arange(0, 1000)
    Idx_col = interpolation(Idx_x, Idx_y, Idx_row)
    
    #draw map
    tra_map.map_update(Idx_row, Idx_col)
    
    #save map
    Grid_Map_Album.append(tra_map.Map)



###########################################################################
################## Grid Map Visualization and Save ########################
###########################################################################

show_sat_map = True
if show_sat_map:
    sat = dgps2sat(zoom=20, size=640, scale=2, img_num=5, map_type='satellite')
    sat.img_num = num_maps
    sat.DownloadMapsFromCsv('./dgps.csv')
    sat_album = sat.satmap_album

    plt.ion()
    fig_MAP = plt.figure('MAP')
    ax1 = fig_MAP.add_subplot(1, 1, 1)
    ax1.axis('off')
    for i, MAP in enumerate(sat_album):
            ax1.imshow(MAP)
            plt.pause(0.01)
            plt.show()


show_grid_map = True
if show_grid_map:
    plt.ion()
    fig_MAP = plt.figure('MAP')

    for i, MAP in enumerate(Grid_Map_Album):
        ax = fig_MAP.add_subplot(1, 1, 1)
        plt.rcParams['image.cmap']='jet'
        imagemat=ax.imshow(MAP)
        imagemat.set_clim(0.0,1.0) 
        cbar=plt.colorbar(imagemat,ax=ax)
        cbar.set_label('Likelihood of Trajectory',size=18)
        plt.pause(0.01)
        plt.show()

plt.ioff()




