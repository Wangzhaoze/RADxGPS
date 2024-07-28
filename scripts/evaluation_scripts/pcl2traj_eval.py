import numpy as np
import os
import glob
import torch.nn as nn

data_path = './dataset'

pcl2traj_label_path = glob.glob(os.path.join(data_path, 'traj_eval/*.npy'))
pcl2traj_pred_path = glob.glob(os.path.join(data_path, 'ogm2traj_pred/*.npy'))

n = min(len(pcl2traj_label_path), len(pcl2traj_pred_path))

res1_25 = 0
res1_50 = 0
res1_75 = 0

softmax = nn.Softmax(dim=1)

def weighted_average_x(pred_matrix):
    """
    Calculate the weighted average of x-coordinates based on softmax probabilities.

    Parameters:
    - pred_matrix: Predicted probability matrix with each row containing a probability distribution.

    Returns:
    - estimated_x: Estimated x-coordinates based on weighted average.
    """
    # Apply softmax along each row
    softmax_probs = np.exp(pred_matrix) / np.sum(np.exp(pred_matrix), axis=1, keepdims=True)

    # Calculate weighted average x-coordinate estimate
    estimated_x = np.sum(softmax_probs * np.arange(pred_matrix.shape[1]), axis=1)

    return estimated_x

for i in range(n):

    try: 

        label = np.load(pcl2traj_label_path[i])[:, 300:700]

        pred = np.load(pcl2traj_pred_path[i].replace('ogm2traj_pred', 'traj_eval').replace('_ogm2traj.npy', '.npy'))

        pred_traj = np.zeros((1000, 2))
        label_traj = np.zeros((1000, 2))

        pred_traj[:, 0] = np.arange(0, 1000)
        pred_traj[:, 1] = weighted_average_x(pred)

        label_traj[:, 0] = np.arange(0, 1000)
        label_traj[:, 1] = np.argmax(label, axis=1)

        delta = pred_traj[:, 1] - label_traj[:, 1] 
        res1_25 += delta[250]**2 / n
        res1_50 += delta[500]**2 / n
        res1_75 += delta[750]**2 / n

    except: 
        continue

res1 = (np.sqrt(res1_25) + np.sqrt(res1_50) + np.sqrt(res1_75)) / 3
res2 = np.sqrt(res1_25)
print(res1)
print(res2)
