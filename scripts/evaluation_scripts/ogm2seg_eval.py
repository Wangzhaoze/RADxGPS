import cv2
from func import *
import os
import glob

data_path = './dataset'
#ogm_path = glob.glob(os.path.join(data_path, 'ogm/*.npy'))
ogm2seg_label_path = glob.glob(os.path.join(data_path, 'seg_eval/*.png'))
ogm2seg_pred_path = glob.glob(os.path.join(data_path, 'multi_task_seg_pred/*.png'))

n = len(ogm2seg_pred_path)
miou = 0
acc = 0
pre = 0
recall = 0
f1score = 0
for i in range(n):

    pred = cv2.imread(ogm2seg_pred_path[i], cv2.IMREAD_GRAYSCALE)
    pred[pred==255] = 1

    label = cv2.imread(ogm2seg_label_path[i], cv2.IMREAD_GRAYSCALE)
    if label.shape[0] == 3:
        break
    label[label==255] = 1

    miou += compute_miou(pred, label)
    acc += confusion_matrix(pred, label)[0]
    pre += confusion_matrix(pred, label)[1]
    recall += confusion_matrix(pred, label)[2]
    f1score += confusion_matrix(pred, label)[3]

miou = miou / n
acc = acc / n
pre = pre / n
recall = recall / n
f1score = f1score / n

print(f'miou of {n} pics:', miou)
print([acc, pre, recall, f1score])