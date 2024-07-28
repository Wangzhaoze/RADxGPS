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
        self.imgs_path = glob.glob(os.path.join(data_path, 'pcl_train/*.png'))

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('pcl_train', 'traj_label').replace('png', 'npy')
        #label_path = label_path.replace('.png', '.png') # todo 更新标签文件的逻辑

        # 读取训练图片和标签图片
        # print(image_path)
        # print(label_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = np.load(label_path)
        image = image[:, 300:700]
        label = label[:, 300:700]

        # 将数据转为3通道的图片
        image = np.repeat(image[np.newaxis, ...], 3, axis=0)

        #label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255

        # 随机进行数据增强，为2时不做处理
        flipCode = 2
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        
        label = np.argmax(label, axis=1)
        label = label.reshape(1, label.shape[0])
        return image, label

    def __len__(self):
        # 返回训练集大小
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)