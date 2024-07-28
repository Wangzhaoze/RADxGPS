import cv2
import imutils
from tqdm import trange
from skimage import io
import matplotlib.pyplot as plt
import pymap3d
import numpy as np
from scripts.dataset_scripts.dgps2seg.utils import *
import os
from scripts.dataset_scripts.dgps2seg.sat2seg_pred import sat2seg


# 72dpi = 28.346pixels/1cm
# zoom = 20 --> 0.27m / 1 pixel
# zoom = 19 --> 0.54m / 1 pixel

# 96dpi = 37.7592pixels/1cm = 38pixels/1cm
# zoom = 20 --> 0.2118689m ~ 0.21052632 / 1 pixel
# zoom = 19 --> 0.39725418 ~ 0.39473684 / 1 pixel

# 100dpi = 39.37pixels/1cm = 40pixels/1cm
# zoom = 20 --> 0.20015011m ~ 0.2 / 1 pixel


class dgps2seg:

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
        self.img_num = 100
        self.map_type = 'satellite'
        self.save_path = None
        self.add_map_to_dict = False
        self.interval = 20

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
            else:
                azimuth_list.append(azi)
                elevation_list.append(el)
                srange_list.append(srange)

        dgps_dict['azimuth'] = azimuth_list
        dgps_dict['elevation'] = elevation_list
        dgps_dict['srange'] = srange_list

        return dgps_dict

    ################### download satellite images automaticlly from time_stamp synchronized dgps_dict ######################
    def dgps_dict2seg(self, dgps_dict):
        

        dgps_dict = self.LLA2AER(dgps_dict)

        metatime_list = dgps_dict['meta_time_abs']
        latitudeDeg_list = dgps_dict['HunterPosLat']
        longitudeDeg_list = dgps_dict['HunterPosLon']
        altitude_list = dgps_dict['nc_altitude']
        azimuth_list = dgps_dict['azimuth']
        elevation_list = dgps_dict['elevation']

        n = len(metatime_list)

        for i in trange(0, n, self.interval):
            lat = latitudeDeg_list[i]
            long = longitudeDeg_list[i]
            alt = altitude_list[i]
            metatime = metatime_list[i]
            azimuth = azimuth_list[i]
            elevation = elevation_list[i]

            img_name = os.path.join(self.save_path, del_char(metatime) + '.png')

            # transformation
            lat_shift, long_shift, _ = pymap3d.aer2geodetic(az=azimuth, el=elevation, srange=500*0.1, lat0=lat, lon0=long, h0=alt)

            img_shift = self.url2sat(lat_shift, long_shift)
            img = self.url2sat(lat, long)

            img_shift = imutils.rotate(img_shift, azimuth)
            img = imutils.rotate(img, azimuth)
            img = img[0:640, :, :]

            Idx = find_line(img_shift[:, 320:960, :], img[:, 320:960, :])
            sat_image = img_shift[Idx-1000:Idx, 140:1140, 0:3]
            
            if sat_image.shape == (1000, 1000, 3):               
                seg_image = sat2seg(sat_image)
            else:
                ##google error
                seg_image = np.random.randint((1000, 1000, 1))

            plt.rcParams['image.cmap']='gray'
            
            cv2.imwrite(img_name, seg_image)


        pass



        pass

