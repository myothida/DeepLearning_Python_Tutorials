import torch
import torch.nn as nn


class Linear(nn.Module):
    """
    Custom implementation of a Linear layer to demonstrate how fully connected layers work.
    """
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weights = torch.randn(out_features, in_features, requires_grad=True)
        self.bias = torch.randn(out_features, requires_grad=True)

    def forward(self, x):
        return x @ self.weights.T + self.bias

