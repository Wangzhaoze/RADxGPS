import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def f_score(inputs, target, beta=1, smooth=1e-5, threshold=0.5):
    """
    Compute the F-score, a weighted average of precision and recall.

    Args:
    - inputs (torch.Tensor): Predicted segmentation masks.
    - target (torch.Tensor): Ground truth segmentation masks.
    - beta (float): Weight parameter for F-score calculation (default is 1).
    - smooth (float): Smoothing factor to avoid division by zero.
    - threshold (float): Threshold value for binarizing the input masks.

    Returns:
    - score (torch.Tensor): F-score value.
    """
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    temp_inputs = torch.gt(temp_inputs, threshold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def fast_hist(a, b, n):
    """
    Compute a histogram for evaluating segmentation results.

    Args:
    - a (numpy.ndarray): Ground truth labels.
    - b (numpy.ndarray): Predicted labels.
    - n (int): Number of classes.

    Returns:
    - hist (numpy.ndarray): Histogram matrix.
    """
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    """
    Compute per-class Intersection over Union (IoU) values.

    Args:
    - hist (numpy.ndarray): Histogram matrix.

    Returns:
    - iu (numpy.ndarray): IoU values.
    """
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    """
    Compute per-class Pixel Accuracy (PA) and Recall values.

    Args:
    - hist (numpy.ndarray): Histogram matrix.

    Returns:
    - pa_recall (numpy.ndarray): PA and Recall values.
    """
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    """
    Compute per-class Precision values.

    Args:
    - hist (numpy.ndarray): Histogram matrix.

    Returns:
    - precision (numpy.ndarray): Precision values.
    """
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    """
    Compute per-class Accuracy.

    Args:
    - hist (numpy.ndarray): Histogram matrix.

    Returns:
    - accuracy (float): Accuracy value.
    """
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    """
    Compute mean IoU (mIoU), mean PA, and Accuracy over all classes given ground truth and predicted segmentation masks.

    Args:
    - gt_dir (str): Directory containing ground truth segmentation masks.
    - pred_dir (str): Directory containing predicted segmentation masks.
    - png_name_list (list): List of image names.
    - num_classes (int): Number of classes.
    - name_classes (list): List of class names.

    Returns:
    - hist (numpy.ndarray): Confusion matrix.
    - IoUs (numpy.ndarray): Per-class IoU values.
    - PA_Recall (numpy.ndarray): Per-class PA and Recall values.
    - Precision (numpy.ndarray): Per-class Precision values.
    """
    print('Num classes', num_classes)
    hist = np.zeros((num_classes, num_classes))

    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))

        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        label = np.array([int(x) for x in label.flatten()])
        label[label == 255] = 1

        pred = np.array([int(x) for x in pred.flatten()])
        pred[pred == 255] = 1

        hist += fast_hist(label, pred, num_classes)

        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
              + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
            round(Precision[ind_class] * 100, 2)))

    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))
    return np.array(hist, np.int), IoUs, PA_Recall, Precision


def adjust_axes(r, t, fig, axes):
    """
    Adjust the axes of a plot to accommodate text labels.

    Args:
    - r: Renderer object.
    - t: Text object.
    - fig: Figure object.
    - axes: Axes object.
    """
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    """
    Draw a horizontal bar plot for evaluation metrics.

    Args:
    - values (list): List of metric values.
    - name_classes (list): List of class names.
    - plot_title (str): Title of the plot.
    - x_label (str): Label for the x-axis.
    - output_path (str): Output path for saving the plot.
    - tick_font_size (int): Font size for ticks (default is 12).
    - plt_show (bool): Whether to display the plot (default is True).
    """
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()


def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size=12):
    """
    Display and save evaluation results including mIoU, mPA, Recall, Precision, and the confusion matrix.

    Args:
    - miou_out_path (str): Output path for saving evaluation results.
    - hist (numpy.ndarray): Confusion matrix.
    - IoUs (numpy.ndarray): Per-class IoU values.
    - PA_Recall (numpy.ndarray): Per-class PA and Recall values.
    - Precision (numpy.ndarray): Per-class Precision values.
    - name_classes (list): List of class names.
    - tick_font_size (int): Font size for ticks (default is 12).
    """
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs) * 100), "Intersection over Union", \
                   os.path.join(miou_out_path, "mIoU.png"), tick_font_size=tick_font_size, plt_show=True)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Pixel Accuracy", \
                   os.path.join(miou_out_path, "mPA.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))

    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall) * 100), "Recall", \
                   os.path.join(miou_out_path, "Recall.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100), "Precision", \
                   os.path.join(miou_out_path, "Precision.png"), tick_font_size=tick_font_size, plt_show=False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
