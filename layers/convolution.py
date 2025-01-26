import torch
import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):
    """
    Custom implementation of a convolutional layer to demonstrate how convolution works.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Convolution, self).__init__()
        self.weights = torch.randn(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
        self.bias = torch.randn(out_channels, requires_grad=True)
        self.stride = stride
        self.padding = padding

    def forward(self, x):  
        batch_size, in_channels, height, width = x.shape
        out_channels, _, kernel_h, kernel_w = self.weights.shape

        out_height = (height + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (width + 2 * self.padding - kernel_w) // self.stride + 1

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))

        output = torch.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for oc in range(out_channels):
                for i in range(0, out_height):
                    for j in range(0, out_width):
                        for ic in range(in_channels):
                            h_start = i * self.stride
                            h_end = h_start + kernel_h
                            w_start = j * self.stride
                            w_end = w_start + kernel_w
                            output[b, oc, i, j] += torch.sum(
                                x[b, ic, h_start:h_end, w_start:w_end] * self.weights[oc, ic]
                            )
                output[b, oc] += self.bias[oc]
        return output





