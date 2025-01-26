import torch
import torch.nn as nn

class ReLU(nn.Module):
    """
    Custom implementation of a ReLU layer to demonstrate how ReLU activation works.
    """
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        output = torch.maximum(x, torch.zeros_like(x))
        return output