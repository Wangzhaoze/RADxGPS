import sys
import os
sys.path.append(os.getcwd())

from models.ogm2traj_model import ogm2traj_UNet
from ogm2traj_dataloader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter


def train_net(net, device, data_path, epochs=30, batch_size=1, lr=0.00001):

    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,batch_size=batch_size,shuffle=True)

    # 定义RMSprop优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    
    # 定义Loss
    criterion = nn.MSELoss()

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 创建TensorBoard的SummaryWriter对象
    log_dir = "./logs/pcl2traj_densenet201"  # 指定保存日志的目录
    writer = SummaryWriter(log_dir=log_dir)

    # 训练epochs次
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):

            # 训练模式
            net.train()

            # 按照batch_size开始训练
            for batch_index, (image, label) in enumerate(train_loader):

                # 梯度清零
                optimizer.zero_grad()

                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # 使用网络参数，输出结果
                output = net(image)

                label = label[0]

                # 计算loss
                loss = criterion(output, label)

                # 写入损失值到TensorBoard
                writer.add_scalar("Loss/train", loss.item(), epoch * per_epoch_num + batch_index)

                # 保存loss值最小的网络参数
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    torch.save(net.state_dict(), './pth/best_model_pcl2traj_k7_densenet.pth')

                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)

    # 关闭SummaryWriter对象
    writer.close()


if __name__ == "__main__":

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片单通道1，分类为1。
    from models.densenet import DenseNet
    net = DenseNet(num_classes=1, input_size=(1000, 400))  

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    data_path = "./dataset" #本地的数据集位置

    print("------training------")
    train_net(net, device, data_path, epochs=40, batch_size=1)
