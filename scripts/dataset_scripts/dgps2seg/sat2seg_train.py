import sys
import os
sys.path.append(os.getcwd())

from models.sat2seg_model import sat2seg_UNet
from scripts.dataset_scripts.dgps2seg.sat2seg_dataloader import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):

    # Load the training dataset
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,batch_size=batch_size,shuffle=True)

    # Define the RMSprop optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Initialize best_loss to positive infinity
    best_loss = float('inf')

    # Train for 'epochs' epochs
    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):

            # Set to training mode
            net.train()

            # Train for each batch
            for image, label in train_loader:

                # Clear gradients
                optimizer.zero_grad()

                # Copy data to the device
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                # Use network parameters to produce output
                output = net(image)

                # Calculate loss
                loss = criterion(output, label)

                print('{}/{}:Loss/train'.format(epoch + 1, epochs), loss.item())
                # Save the parameters of the network with the smallest loss
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), './pth/best_model_sat2seg_k5.pth')

                # Update parameters
                loss.backward()
                optimizer.step()
                pbar.update(1)


if __name__ == "__main__":

    # Choose device, use cuda if available, otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the network, with 1 channel for image, and 1 class for segmentation
    net = sat2seg_UNet(n_channels=1, n_classes=1)  

    # Copy the network to the specified device
    net.to(device=device)

    # Specify the training dataset directory and start training
    data_path = "./dataset" # The local dataset location

    print("------training------")
    train_net(net, device, data_path, epochs=40, batch_size=1)
