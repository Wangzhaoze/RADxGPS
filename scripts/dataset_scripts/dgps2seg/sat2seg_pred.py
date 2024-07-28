import glob
import numpy as np
import torch
import cv2
import sys
import os

# Add current directory to the system path to import local modules
sys.path.append(os.getcwd())

from models.sat2seg_model import sat2seg_UNet

def sat2seg(sat_image):
    # Choose device, use cuda if available, otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load network, image with single channel, and classification into 1 class
    net = sat2seg_UNet(n_channels=1, n_classes=1)

    # Copy the network to the specified device
    net.to(device=device)

    # Load model parameters
    net.load_state_dict(torch.load('./pth/best_model_sat2seg_k3k7.pth', map_location=device))

    # Set to evaluation mode
    net.eval()

    origin_shape = sat_image.shape

    # Convert to grayscale
    img = cv2.cvtColor(sat_image, cv2.COLOR_RGB2GRAY)

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

    # Process result
    pred[pred >= 0.5] = 255
    pred[pred < 0.5] = 0
    
    return pred

if __name__ == '__main__':
    val_images_path = './dataset/sat/'
    save_path = './dataset/sat2seg_pred/'

    img_paths = os.listdir(val_images_path)
    for img_path in img_paths:
        sat = cv2.imread(val_images_path+img_path)
        pred = sat2seg(sat)
        cv2.imwrite(save_path+img_path, pred)
