
import numpy as np 
import os
from scripts.dataset_scripts.dgps2traj.utils import *
import pymap3d
from scripts.dataset_scripts.dgps2traj.TrajMap import TrajGridMap
from matplotlib import cm
import matplotlib.pyplot as plt
from tqdm import trange
import sys 
sys.path.append("..") 





class dgps2traj:

    def __init__(self):
        self.save_path = None
        self.interval = 20
        pass

    def dgps_dict2traj(self, dgps_dict):
        
        ###########################################################################
        ########################### parameters ####################################
        ###########################################################################

        dgps_dict = LLA2AER(dgps_dict)

        metatime_list = dgps_dict['meta_time_abs']
        latitudeDeg_list = dgps_dict['HunterPosLat']
        longitudeDeg_list = dgps_dict['HunterPosLon']
        altitude_list = dgps_dict['nc_altitude']


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
        for i in range(0, data_length-1, 1):

            azi, el, srange = pymap3d.geodetic2aer(lat=latitudeDeg_list[i+1], 
                                                    lon=longitudeDeg_list[i+1], 
                                                    h=altitude_list[i+1],
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
        num_maps = len(metatime_list)

        traj_map = TrajGridMap()
        for i in trange(0, num_maps-200, self.interval):
            #算后800点
            delta_x_array = np.array(delta_x[i:i+200])
            delta_y_array = np.array(delta_y[i:i+200])

            #累加delta, 得绝对坐标
            delta_x_array = np.cumsum(delta_x_array)
            delta_x_array = delta_x_array + 50
            delta_y_array = np.cumsum(delta_y_array)

            #转至正向
            points_x_array, points_y_array = Nrotate(azimuth_list[i], 
                                                    delta_x_array, 
                                                    delta_y_array,
                                                    50,
                                                    0)


            #转到图像坐标系(500, 0)---->(1000, 500)
            Idx_y = points_x_array
            Idx_x = 100 - points_y_array

            Idx_x = Idx_x / 0.1
            Idx_y = Idx_y / 0.1

            #加上初始点
            Idx_x = np.append(1000-1, Idx_x)
            Idx_y = np.append(500-1, Idx_y)


            #坐标插值化整化整
            Idx_row = np.arange(0, 1000)
            try:
                Idx_col = interpolation(Idx_x, Idx_y, Idx_row)
            except Exception:
                pass
            else:
                #draw map
                traj_map.map_update(Idx_row, Idx_col)
                
                #save map
                img_name = os.path.join(self.save_path, del_char(metatime_list[i]) + '.npy')
                np.save(img_name, traj_map.Map)
        pass

if __name__ == '__main__':
    trajmap1 = dgps2traj()

    radar_keys = ['meta_time', 'meta_time_abs']
    dgps_keys = ['meta_time', 'meta_time_abs', 'HunterPosLat','HunterPosLon', 'nc_altitude']
    radar_dict = csv2dict('./dataloader/Seg_scripts/radar_detect.csv', radar_keys)
    dgps_dict = csv2dict('./dataloader/Seg_scripts/rtrange.csv', dgps_keys)

    dgps_dict, radar_dict = Time_Stamp_Synchronization(dgps_dict, radar_dict)

    trajmap1.save_path = './label/Traj_label'

    trajmap1.dgps_dict2traj(dgps_dict)
