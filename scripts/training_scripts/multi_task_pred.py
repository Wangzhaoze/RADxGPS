import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.getcwd())

from models.multi_task_model import multi_task_UNet
from utils import *


if __name__ == "__main__":


    pth_path = './pth/best_model_multi_task.pth'
    
    tests_path = glob.glob('dataset/ogm_eval/*.npy')


    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载网络，图片单通道，分类为1。
    net = multi_task_UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load(pth_path, map_location=device))

    # 测试模式
    net.eval()

    # 遍历素有图片
    for test_path in tests_path:

        seg_pred_path = (test_path.split('.')[0] + '_seg_pred.png').replace('ogm_eval', 'multi_task_seg_pred')
        traj_pred_path = (test_path.split('.')[0] + '_traj_pred.npy').replace('ogm_eval', 'multi_task_traj_pred')


        # 读取图片
        ogm = np.load(test_path)

        # 切片
        ogm = ogm[:, 300:700]
        origin_shape = ogm.shape
        print(origin_shape)
        
        # 保存ogm图片
        plt.rcParams['image.cmap']='jet'
        #plt.imsave(ogm_path, ogm_img)

        # 转为batch为1，通道为1，大小为512*512的数组
        ogm = ogm.reshape(1, 1, ogm.shape[0], ogm.shape[1])

        # 转为tensor
        ogm_img_tensor = torch.from_numpy(ogm)

        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        ogm_img_tensor = ogm_img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        pred = net(ogm_img_tensor)

        seg_pred, traj_pred = torch.chunk(pred, chunks=2, dim=1)

        # seg
        seg_pred = np.array(seg_pred.data.cpu()[0])[0]*255
        seg_pred[seg_pred >= 100] = 255
        seg_pred[seg_pred < 100] = 0
        print(test_path)
        seg_pred = post_process(seg_pred)
        plt.rcParams['image.cmap']='gray'
        #pred = post_process(pred)
        plt.imsave(seg_pred_path, seg_pred)

        # traj
        traj_pred = np.array(traj_pred.data.cpu()[0])[0]
        print(traj_pred.shape)
        np.save(traj_pred_path, traj_pred)





