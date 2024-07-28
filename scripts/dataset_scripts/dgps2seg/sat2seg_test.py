# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
import cv2
import sys
import os

# Add current directory to the system path to import local modules
sys.path.append(os.getcwd())

from models.sat2seg_model import sat2seg_UNet

def cal_miou(test_dir="./trainset/val_set",
             pred_dir="./results", gt_dir="trainset/val_label"):
    # Mode for calculating mIoU
    # miou_mode 0: complete mIoU calculation process, including getting predictions and calculating mIoU.
    # miou_mode 1: only get predictions.
    # miou_mode 2: only calculate mIoU.
    miou_mode = 0

    # Number of classes +1, e.g., 2+1
    num_classes = 2

    # Class names, same as in json_to_dataset
    name_classes = ["background", "nidus"]

    # Calculation of results and comparison with ground truth

    # Load the model

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load the network, single channel for image, 1 class for classification
        net = sat2seg_UNet(n_channels=1, n_classes=1)
        # Copy the network to the device
        net.to(device=device)
        # Load model parameters
        net.load_state_dict(torch.load('best_model_skin.pth', map_location=device)) # todo
        # Set to evaluation mode
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
            # Convert to a batch of size 1, with 1 channel, and size of 512*512
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # Convert to tensor
            img_tensor = torch.from_numpy(img)
            # Copy the tensor to the device, if using CPU, copy to CPU, if using CUDA, copy to CUDA
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # Predict
            pred = net(img_tensor)
            # Extract result
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get mIoU.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  # Execute the function to calculate mIoU
        print("Get mIoU done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()
