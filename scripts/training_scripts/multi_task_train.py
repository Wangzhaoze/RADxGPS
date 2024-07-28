import sys
import os
sys.path.append(os.getcwd())

from models.multi_task_model import multi_task_UNet
from scripts.training_scripts.multi_task_dataloader import Data_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
from loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter


def train_net(net, device, data_path, epochs=30, batch_size=1, lr=0.00001):

    # 加载训练集
    isbi_dataset = Data_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,batch_size=batch_size,shuffle=True)

    # 定义Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    # optimizer = optim.Adam([
    # {'params': net.shared_layers.parameters(), 'lr': 1e-5},
    # {'params': net.seg_specific_layers.parameters(), 'lr': 1e-5},
    # {'params': net.traj_specific_layers.parameters(), 'lr': 1e-4}], weight_decay=1e-8)

    # 定义Loss
    criterion1 = nn.BCEWithLogitsLoss()
    criterion2 = FocalLoss()


    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 创建TensorBoard的SummaryWriter对象
    log_dir = "./log/logsmulti"  # 指定保存日志的目录
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

                seg_output, traj_output = torch.chunk(output, chunks=2, dim=1)
                seg_label, traj_label = torch.chunk(label, chunks=2, dim=1)


                # 计算loss
                loss_seg = criterion1(seg_output, seg_label)
                loss_traj = criterion2(traj_output, traj_label)

                loss = 0.75 * loss_seg + 0.25 * loss_traj


                #print('{}/{}:Loss/train'.format(epoch + 1, epochs), loss.item())
                # 保存loss值最小的网络参数
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), './pth/best_model_multi_task.pth')
                    
                # 写入损失值到TensorBoard
                writer.add_scalar("loss_seg/train", loss_seg.item(), epoch * per_epoch_num + batch_index)
                writer.add_scalar("loss_traj/train", loss_traj.item(), epoch * per_epoch_num + batch_index)
                writer.add_scalar("Loss_total/train", loss.item(), epoch * per_epoch_num + batch_index)
                    

                # 更新参数
                loss.backward()
                optimizer.step()
                pbar.update(1)


if __name__ == "__main__":

    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载网络，图片单通道1，分类为1。
    net = multi_task_UNet(n_channels=1, n_classes=1)  

    # 将网络拷贝到deivce中
    net.to(device=device)

    # 指定训练集地址，开始训练
    data_path = "./dataset" #本地的数据集位置

    print("------training------")
    train_net(net, device, data_path, epochs=70, batch_size=1)
