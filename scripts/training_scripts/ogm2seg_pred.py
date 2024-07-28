import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

from models.ogm2seg_model import ogm2seg_UNet
from utils import *


if __name__ == "__main__":


    pth_path = './pth/best_model_ogm2seg_k7.pth'
    
    tests_path = glob.glob('dataset/ogm_eval/*.npy')


    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载网络，图片单通道，分类为1。
    net = ogm2seg_UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load(pth_path, map_location=device))

    # 测试模式
    net.eval()

    # 遍历素有图片
    for test_path in tests_path:

        # 保存结果地址
        pred_path = test_path.split('.')[0] + '_pred.png'
        ogm_path = test_path.split('.')[0] + '_ogm.png'

        pred_path = pred_path.replace('ogm_eval', 'ogm2seg_pred')
        ogm_path = ogm_path.replace('ogm_eval', 'ogm2seg_pred')

        # 读取图片
        ogm_img = np.load(test_path)

        # 切片
        ogm_img = ogm_img[:, 300:700]
        origin_shape = ogm_img.shape
        print(origin_shape)
        
        # 保存ogm图片
        plt.rcParams['image.cmap']='jet'
        #plt.imsave(ogm_path, ogm_img)

        # 转为batch为1，通道为1，大小为512*512的数组
        ogm_img = ogm_img.reshape(1, 1, ogm_img.shape[0], ogm_img.shape[1])

        # 转为tensor
        ogm_img_tensor = torch.from_numpy(ogm_img)

        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        ogm_img_tensor = ogm_img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        pred = net(ogm_img_tensor)

        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]*255

        # 处理结果
        pred[pred >= 100] = 255
        pred[pred < 100] = 0
        
        # 保存图片
        plt.rcParams['image.cmap']='gray'
        pred = post_process(pred)
        plt.imsave(pred_path, pred)
