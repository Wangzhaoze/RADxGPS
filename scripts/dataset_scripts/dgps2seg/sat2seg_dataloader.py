import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random

class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # Initialization function, read all images under data_path
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'sat/*.png'))

    def augment(self, image, flipCode):
        # Perform data augmentation using cv2.flip, flipCode 1 for horizontal flip, 0 for vertical flip, -1 for both
        flip = cv2.flip(image, flipCode)
        return flip
        
    def __getitem__(self, index):
        # Read images based on index
        image_path = self.imgs_path[index]
        # Generate label_path based on image_path
        label_path = image_path.replace('sat', 'sat2seg_label')
        #label_path = label_path.replace('.png', '.png') # todo Update the logic for label file

        # Read training images and label images
        # print(image_path)
        # print(label_path)
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)

        # Convert data to single-channel images
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

        # Process labels, change pixel values to 1 if they are 255
        if label.max() > 1:
            label = label / 255
        # Perform random data augmentation, no processing when flipCode is 2
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])
        return image, label

    def __len__(self):
        # Return the size of the training set
        return len(self.imgs_path)

    
if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("data/train/")
    print("Number of data:", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
