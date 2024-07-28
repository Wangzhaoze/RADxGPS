import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random
import numpy as np


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        #self.ogm_path = glob.glob(os.path.join(data_path, 'ogm/*.npy'))
        self.label_path = glob.glob(os.path.join(data_path, 'seg_label/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        label_path = self.label_path[index]
        pcl_path = label_path.replace('seg_label', 'pcl_train')
        image_path = pcl_path
        #.replace('png', 'npy')

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        image = image[:, 300:700]
        label = label[:, 300:700]

        # 处理标签，将像素值为255的改为1
        label[label==255] = 1.0

        # 随机进行数据增强，为2时不做处理
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.label_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("./data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)