import sys
import os
sys.path.append(os.getcwd())

from models.ogm2seg_model import ogm2seg_UNet
from scripts.training_scripts.pcl2seg_dataloader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_net(net, device, data_path, epochs=30, batch_size=1, lr=0.00001):

    # 加载训练集
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,batch_size=batch_size,shuffle=True)

    # 定义Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)

    # 定义Loss
    criterion = nn.BCEWithLogitsLoss()

    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 创建TensorBoard的SummaryWriter对象
    log_dir = "./logs/pcl2traj"  # 指定保存日志的目录
    writer = SummaryWriter(log_dir=log_dir)

    # 训练epochs次
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):

            # 训练模式
            net.train()

            # 按照batch_size开始训练
            for i, (image, label) in enumerate(train_loader):

                # 梯度清零
                optimizer.zero_grad()

                # 将数据拷贝到device中
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # 使用网络参数，输出结果
                output = net(image)

                # 计算loss
                loss = criterion(output, label)

                # 使用写入器记录训练损失
                writer.add_scalar('Loss/train', loss.item(), epoch * per_epoch_num + i)

                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss

                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)

    # 关闭TensorBoard写入器
    writer.close()


if __name__ == "__main__":

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片单通道1，分类为1。
    net = ogm2seg_UNet(n_channels=1, n_classes=1)  

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    data_path = "./dataset" #本地的数据集位置

    print("------training------")
    train_net(net, device, data_path, epochs=40, batch_size=1)