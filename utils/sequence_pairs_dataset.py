from torch.utils.data import Dataset

class SequencePairsDataset(Dataset):
    """
    A PyTorch Dataset for generating input-output sequence pairs using a 
    sliding window approach over sequential data.

    Args:
        data (list or array-like): The dataset (e.g., tokenized IDs or numerical series).
        block_size (int): The size of the input and output sequences.
    """
    def __init__(self, data, block_size, step_size = 1):
        self.data = data
        self.block_size = block_size
        self.step_size = step_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, pos):
        assert pos < (len(self.data) - self.block_size) // self.step_size

        # Calculate the actual position of the window
        start = pos * self.step_size
        end = start + self.block_size

        x = self.data[start:end]  # input
        y = self.data[start + 1:start + 1 + self.block_size]  # output
        return x, y
