import numpy as np
import cv2


# Define function to calculate mean intersection over union (mIoU)
def compute_miou(prediction, label):
    intersection = np.logical_and(prediction, label)
    union = np.logical_or(prediction, label)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Define function to calculate average displacement error (ADE)
def compute_ade(prediction, label):
    prediction = prediction
    label = label
    error = np.linalg.norm(prediction - label, axis=1)
    ade_score = np.mean(error)
    return ade_score

# Define function to calculate final displacement error (FDE)
def compute_fde(prediction, label):
    error = np.linalg.norm(prediction[-1] - label[-1])
    return error





# Define function to calculate confusion matrix and acc, pre, recall & f1 score

def confusion_matrix(predicted_labels, true_labels):
    """
    计算混淆矩阵和各项参数
    """
    tp = np.sum((true_labels == 1) & (predicted_labels == 1))
    tn = np.sum((true_labels == 0) & (predicted_labels == 0))
    fp = np.sum((true_labels == 0) & (predicted_labels == 1))
    fn = np.sum((true_labels == 1) & (predicted_labels == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    
    #return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn, 'accuracy': accuracy,'precision': precision, 'recall': recall, 'f1_score': f1_score}
    return accuracy, precision, recall, f1_score


def compute_formular1(prediction, label):
    delta = prediction - label
    return delta[250, 1]**2, delta[500, 1]**2, delta[750, 1]**2

    
