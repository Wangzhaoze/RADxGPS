import cv2
from func import *
import os
import glob

data_path = './dataset'
#pcl_path = glob.glob(os.path.join(data_path, 'pcl/*.npy'))
pcl2seg_label_path = glob.glob(os.path.join(data_path, 'seg_eval/*.png'))
pcl2seg_pred_path = glob.glob(os.path.join(data_path, 'snr2seg_pred/*.png'))

n = len(pcl2seg_pred_path)
miou = 0
acc = 0
pre = 0
recall = 0
f1score = 0
for i in range(n):

    pred = cv2.imread(pcl2seg_pred_path[i], cv2.IMREAD_GRAYSCALE)
    pred[pred==255] = 1

    label = cv2.imread(pcl2seg_label_path[i], cv2.IMREAD_GRAYSCALE)
    if label.shape[0] == 3:
        break
    label[label==255] = 1
    if compute_miou(pred, label) < 0.8:
        print(pcl2seg_pred_path[i])
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