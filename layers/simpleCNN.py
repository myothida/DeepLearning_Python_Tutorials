import torch
import torch.nn as nn
from layers.convolution import Convolution
from layers.pooling import Pooling
from layers.relu import ReLU
from layers.linear import Linear


class SimpleCNN(nn.Module):
    """
    Custom CNN model built using the Convolution, Pooling, ReLU, and Linear layers.
    """
    def __init__(self, input_channels, num_classes):
        super(SimpleCNN, self).__init__()

        # Define the architecture
        self.conv1 = Convolution(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.pool1 = Pooling(kernel_size=2, stride=2, pooling_type="max")

        self.conv2 = Convolution(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool2 = Pooling(kernel_size=2, stride=2, pooling_type="max")

        self.conv3 = Convolution(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU()
        self.pool3 = Pooling(kernel_size=2, stride=2, pooling_type="max")

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = Linear(64 * 4 * 4, 128)  # Adjust based on input size
        self.relu_fc = ReLU()
        self.fc2 = Linear(128, num_classes)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Third convolutional block
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.fc2(x)

        return x
