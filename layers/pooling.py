import torch
import torch.nn as nn

class Pooling(nn.Module):
    """
    Custom implementation of a pooling layer to demonstrate how pooling works.
    """
    def __init__(self, kernel_size, stride=None, pooling_type="max"):
        super(Pooling, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.pooling_type = pooling_type

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        kernel_h, kernel_w = self.kernel_size, self.kernel_size
        stride = self.stride

        out_height = (height - kernel_h) // stride + 1
        out_width = (width - kernel_w) // stride + 1


        output = torch.zeros((batch_size, channels, out_height, out_width))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * stride
                        h_end = h_start + kernel_h
                        w_start = j * stride
                        w_end = w_start + kernel_w

                        region = x[b, c, h_start:h_end, w_start:w_end]

                        if self.pooling_type == "max":
                            output[b, c, i, j] = torch.max(region)
                        elif self.pooling_type == "avg":
                            output[b, c, i, j] = torch.mean(region)
                        else:
                            raise ValueError("Unsupported pooling type. Use 'max' or 'avg'.")
        return output
