import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np


class Data_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        #self.ogm_path = glob.glob(os.path.join(data_path, 'ogm/*.npy'))
        self.seg_label_path = glob.glob(os.path.join(data_path, 'seg_label/*.png'))
        self.traj_label_path = glob.glob(os.path.join(data_path, 'traj_label/*.npy'))


    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        seg_label_path = self.seg_label_path[index]
        traj_label_path = self.traj_label_path[index]


        input_data = 'ogm'

        if input_data == 'ogm':
            ogm_path = seg_label_path.replace('seg_label', 'ogm_train')
            image_path = ogm_path.replace('png', 'npy')
            image = np.load(image_path)
        
        elif input_data == 'pcl':
            image_path = seg_label_path.replace('seg_label', 'pvl_train')
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)


        
        seg_label = cv2.imread(seg_label_path, cv2.IMREAD_GRAYSCALE)
        traj_label = np.load(traj_label_path)

        image = image.reshape(1, image.shape[0], image.shape[1])
        seg_label = seg_label.reshape(1, seg_label.shape[0], seg_label.shape[1])
        seg_label[seg_label==255] = 1.0
        traj_label = traj_label.reshape(1, traj_label.shape[0], traj_label.shape[1])

        label = np.concatenate((seg_label, traj_label), axis=0)

        image = image[:, :, 300:700]
        label = label[:, :, 300:700]


        # 随机进行数据增强，为2时不做处理
        # flipCode = random.choice([-1, 0, 1, 2])
        # if flipCode != 2:
        #     image = self.augment(image, flipCode)
        #     label = self.augment(label, flipCode)

        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.seg_label_path)

    
if __name__ == "__main__":
    isbi_dataset = Data_Loader("./dataset/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)