import glob
import numpy as np
import torch
import os
import cv2
import sys
import matplotlib.pyplot as plt
import os
sys.path.append(os.getcwd())

from models.ogm2traj_model import ogm2traj_UNet




if __name__ == "__main__":

    
    pth_path = './pth/best_model_pcl2traj_k7_softmax.pth'
    
    tests_path = glob.glob('dataset/pcl_eval/*.png')

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 加载网络，图片单通道，分类为1。
    net = ogm2traj_UNet(n_channels=1, n_classes=1)

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 加载模型参数
    net.load_state_dict(torch.load(pth_path, map_location=device))

    # 测试模式
    net.eval()

    # 遍历素有图片
    for test_path in tests_path:

        # 保存结果地址

        pred_path = test_path.split('.')[0] + '_pred_traj.png'
 

        pred_path = pred_path.replace('pcl_eval', 'pcl2traj_pred')


        # 读取图片
        img1 = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

        # qiepian
        img = img1[:, 300:700]
        origin_shape = img.shape
        print(origin_shape)

        # 转为灰度图
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        

        # 转为batch为1，通道为1，大小为512*512的数组
        img = img.reshape(1, 1, img.shape[0], img.shape[1])

        # 转为tensor
        img_tensor = torch.from_numpy(img)

        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)

        # 预测
        pred = net(img_tensor)

        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]

        # 处理结果
        #pred[pred < 0.3] = 0
        
        # 保存图片
        np.save(pred_path, pred)
        plt.rcParams['image.cmap']='jet'
        plt.imsave(pred_path.replace('npy', 'png'), pred)

        
