import numpy as np
from func import *
import os
import glob

data_path = './dataset'

ogm2traj_label_path = glob.glob(os.path.join(data_path, 'traj_eval/*.npy'))
ogm2traj_pred_path = glob.glob(os.path.join(data_path, 'multi_task_traj_pred/*.npy'))

n = len(ogm2traj_label_path)
made = 0
mfde = 0
res1_25 = 0
res1_50 = 0
res1_75 = 0
for i in range(n):

    label = np.load(ogm2traj_label_path[i])
    label = label[:, 300:700]

    pred = np.load(ogm2traj_pred_path[i])


    pred_traj = np.zeros((1000, 2))
    label_traj = np.zeros((1000, 2))

    pred_traj[:, 0] = np.arange(0, 1000)
    pred_traj[:, 1] = np.argmax(pred, axis=1)

    label_traj[:, 0] = np.arange(0, 1000)
    label_traj[:, 1] = np.where(label==1)[1]


    made += compute_ade(pred_traj, label_traj)
    mfde += compute_fde(pred_traj, label_traj) 


    delta = (pred_traj[:, 1] - label_traj[:, 1]) * 0.1 
    if delta[250] < 2:
        res1_25 += delta[250]**2 / 145
        res1_50 += delta[500]**2 / 145
        res1_75 += delta[750]**2 / 145


made = made / n
mfde = mfde / n

print('mean od ADE:', made)
print('mean od FDE:', mfde)

res1 = (np.sqrt(res1_25) + np.sqrt(res1_50) + np.sqrt(res1_75)) / 3
res2 = np.sqrt(res1_75)
print(res1)
print(res2)
