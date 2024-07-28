import sys
import os
sys.path.append(os.getcwd())
import imutils
from tqdm import trange
from skimage import io
import matplotlib.pyplot as plt
import pymap3d
from scripts.dataset_scripts.dgps2seg.utils import *

import cv2



# 72dpi = 28.346pixels/1cm
# zoom = 20 --> 0.27m / 1 pixel
# zoom = 19 --> 0.54m / 1 pixel

# 96dpi = 37.7592pixels/1cm = 38pixels/1cm
# zoom = 20 --> 0.2118689m ~ 0.21052632 / 1 pixel
# zoom = 19 --> 0.39725418 ~ 0.39473684 / 1 pixel

# 100dpi = 39.37pixels/1cm = 40pixels/1cm
# zoom = 20 --> 0.20015011m ~ 0.2 / 1 pixel


class dgps2sat:

    def __init__(self):
        """
        ### parameters:
        zoom:
        size:
        scale:
        img_num:
        may_type:
        """
        self.zoom = 19
        self.size = 640
        self.scale = 2
        self.delta = 20
        self.map_type = 'satellite'
        self.save_path = './'
        self.add_map_to_dict = False
        #self.save_path = './sat_{}.png'.format(metatime)


        self.dgps_keys = ['meta_time', 'meta_time_abs', 'HunterPosLat','HunterPosLon', 'nc_altitude']
        pass


    ###################combinate google-url from given latitudeDeg & longitudeDeg ##############################
    def image_url(self, latDeg, longDeg):

        head_url = 'https://maps.googleapis.com/maps/api/staticmap?'
        api_key = '<apply and paste your Google Map Token here>'
        param_url = f'center={latDeg},{longDeg}&zoom={self.zoom}&scale={self.scale}&size={self.size}x{self.size}&maptype={self.map_type}&'

        # combine urls
        img_url = head_url + param_url + api_key

        return img_url

    #################### download one satellite image from Url of Google-API ##########################
    def url2sat(self, latDeg, longDeg):
        img_url = self.image_url(latDeg, longDeg)
        image = io.imread(img_url)
        return image

    #################### add aer information into dgps_dict ##########################
    def LLA2AER(self, dgps_dict):
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


        dgps_dict['azimuth'] = azimuth_list
        dgps_dict['elevation'] = elevation_list
        dgps_dict['srange'] = srange_list

        return dgps_dict


################### download segmentated images automaticlly from time_stamp synchronized dgps_dict ######################
    def dgps_dict2sat(self, dgps_dict):
        

        dgps_dict = self.LLA2AER(dgps_dict)

        metatime_list = dgps_dict['meta_time_abs']
        latitudeDeg_list = dgps_dict['HunterPosLat']
        longitudeDeg_list = dgps_dict['HunterPosLon']
        altitude_list = dgps_dict['nc_altitude']
        azimuth_list = dgps_dict['azimuth']
        elevation_list = dgps_dict['elevation']

        n = len(metatime_list)

        for i in trange(0, n, self.delta):
            lat = latitudeDeg_list[i]
            long = longitudeDeg_list[i]
            alt = altitude_list[i]
            metatime = metatime_list[i]
            azimuth = azimuth_list[i]
            elevation = elevation_list[i]

            img_name: str = os.path.join(self.save_path, del_char(metatime) + '.png')

            # transformation
            lat_shift, long_shift, _ = pymap3d.aer2geodetic(az=azimuth, el=elevation, srange=500*0.1, lat0=lat, lon0=long, h0=alt)

            img_shift = self.url2sat(lat_shift, long_shift)
            img = self.url2sat(lat, long)
            cv2.imwrite(f'./{lat}_{long}_sat.png', img)

            img_shift = imutils.rotate(img_shift, azimuth)
            img = imutils.rotate(img, azimuth)
            img = img[0:640, :, :]

            Idx = find_line(img_shift[:, 320:960, :], img[:, 320:960, :])
            sat_image = img_shift[Idx-1000:Idx, 140:1140, 0:3]

            if sat_image.shape != (1000, 1000, 3):
              sat_image = np.random.randint((1000, 1000, 3))
            cv2.imwrite(img_name, sat_image)


        pass

if __name__ == "__main__":


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


        load_seg = True
        load_traj = True
        load_ogm = False
        load_sat = False


        # dgps2segmentation
        if load_seg:
            print('----------------- Segmentation Label Loading --------------------')

            loader_dgps2seg = dgps2sat()
            loader_dgps2seg.save_path = './dataset/ogm2seg_label'
            loader_dgps2seg.dgps_dict2sat(dgps_dict)