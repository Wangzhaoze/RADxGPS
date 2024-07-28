
import os
from dgps2seg.utils import *
from dgps2seg.DGPS2Sat import dgps2sat
from dgps2seg.DGPS2Seg import dgps2seg
from dgps2traj.DGPS2Traj import dgps2traj

for i in range(0, 12):
    print(i)
    print()
    print('########################################################')
    # file path
    batch = i
    csv_path = os.path.join('csv_data','batch'+str(batch))
    radar_csv = os.path.join(csv_path, 'radar.csv')
    dgps_csv = os.path.join(csv_path, 'dgps.csv')


    # key names
    radar_keys = ['meta_time', 'meta_time_abs']
    dgps_keys = ['meta_time', 'meta_time_abs', 'HunterPosLat','HunterPosLon', 'nc_altitude']


    # radar & dgps dict
    print('----- loading csv file -----')
    radar_dict = csv2dict(radar_csv, radar_keys)
    dgps_dict = csv2dict(dgps_csv, dgps_keys)
    print('----- load csv file : done -----')


    # time stamp synchronization
    print('----- time stamp synchronizing -----')
    dgps_dict, radar_dict = Time_Stamp_Synchronization(dgps_dict, radar_dict)
    dgps_dict['meta_time_abs'] = del_char_list(dgps_dict['meta_time_abs'])
    radar_dict['meta_time_abs'] = del_char_list(radar_dict['meta_time_abs'])
    print('----- time stamp synchronization : done -----')


    load_seg = False
    load_traj = True
    load_ogm = False
    load_sat = False


    # dgps2segmentation
    if load_seg:
        print('----------------- Segmentation Label Loading --------------------')

        loader_dgps2seg = dgps2seg()
        loader_dgps2seg.save_path = './dataset/ogm2seg_label'
        loader_dgps2seg.dgps_dict2seg(dgps_dict)


    # dgps2traj

    if load_traj:
        print('----------------- Trajectory Label Loading --------------------')
        loader_dgps2traj = dgps2traj()
        loader_dgps2traj.save_path = './dataset/traj_label_all'
        loader_dgps2traj.dgps_dict2traj(dgps_dict)



    # radar2ogm rename
    if load_ogm:
        ogm_save_path = './dataset/det0/'
        namelist_ogm = os.listdir(ogm_save_path)

        namelist_seg = dgps_dict['meta_time_abs']

        for i in range(len(namelist_seg)):
            namelist_seg[i] = namelist_seg[i] + '.npy'


        namelist_ogm.sort(key = lambda x:int(x[:-4]))
        #namelist_seg.sort(key = lambda x:int(x[:-4]))

        for i in range(len(namelist_ogm)):
            os.rename(os.path.join(ogm_save_path, namelist_ogm[i]), os.path.join(ogm_save_path, namelist_seg[i]))



    #dgps2sat image
    if load_sat:
        print('----------------- Satellite Image Loading --------------------')
        loader_dgps2sat = dgps2sat()
        loader_dgps2sat.save_path = './dataset/sat20'
        loader_dgps2sat.dgps_dict2sat(dgps_dict)

