import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, num_classes, input_size):
        """
        Initialize the DenseNet model.

        Args:
            num_classes (int): Number of output classes.
            input_size (tuple): Input size (height, width).
        """
        super(DenseNet, self).__init__()

        # Load the pre-trained DenseNet model
        densenet = models.densenet201()
        self.densenet = models.densenet201()
        
        # Get the feature extraction part of DenseNet
        feature_extractor = nn.Sequential(*list(densenet.children())[:-1])
        
        # Get the output channels of the feature extraction part
        in_channels = densenet.classifier.in_features
        
        # Define a custom segmentation head with 1 output channel
        segmentation_head = SegmentationHead(in_channels, num_classes)

        self.feature_extractor = feature_extractor
        self.segmentation_head = segmentation_head
        
        # Define the upsampling layer to increase the size of the feature map to the input size
        self.upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(1, 1, kernel_size=(1000, 400))

    def forward(self, x):
        """
        Forward pass of the DenseNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        output = self.densenet(x)

        return output


# Define a custom segmentation head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Initialize the segmentation head.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(SegmentationHead, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the segmentation head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        return x


if __name__ == '__main__':

    input_size = (1000, 400)
    # Create a model instance and train
    model = DenseNet(1, input_size)

    # Create a random input
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1])

    # Pass the input through the model for forward propagation
    densenet = models.densenet201()
    output_tensor = model(input_tensor)

    # Print the output size
    print(output_tensor.size())
