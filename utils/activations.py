"""
@staticmethod is used because these activation methods do not require access to any instance-specific or class-specific data. 
They are utility functions that operate solely based on their inputs and do not modify or rely on the state of the class or its instances.
"""
import numpy as np
import torch
class ActivationFunctionsNumpy:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=-1, keepdims=True)

class ActivationFunctionsPyTorch:
    @staticmethod
    def sigmoid(x):
        return torch.sigmoid(x)

    @staticmethod
    def tanh(x):
        return torch.tanh(x)

    @staticmethod
    def relu(x):
        return torch.relu(x)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return torch.nn.functional.leaky_relu(x, negative_slope=alpha)

    @staticmethod
    def softmax(x, dim=-1):
        return torch.nn.functional.softmax(x, dim=dim)

## sample
# NumPy-based activation
result = ActivationFunctionsNumpy.sigmoid(np.array([1.0, 2.0, 3.0]))

# PyTorch-based activation
tensor = torch.tensor([1.0, 2.0, 3.0])
result = ActivationFunctionsPyTorch.relu(tensor)
