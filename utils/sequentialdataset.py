from torch.utils.data import Dataset

"""
SequentialDataset is more flexible and can be used for tasks where the output sequence length may differ from the input,
such as forecasting, time series prediction, or certain NLP tasks where the output sequence may not be exactly the same length as the input.

TokenIdsDataset is designed for tasks like language modeling where the model is expected to predict the next token in the sequence,
making it ideal for training on text data where the input and output are of equal size.
"""

class SequentialDataset(Dataset):
    def __init__(self, data, seq_len, label_len=5):
        """
        A custom dataset class for handling sequential data. It creates input-output pairs based on sliding windows.
        
        Arguments:
        - data: The dataset (list or tensor) containing the sequences.
        - seq_len: Length of the input sequence.
        - label_len: Length of the output (label) sequence. Default is 5.
        """
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.label_len
    
    def __getitem__(self, index):
        assert index + self.seq_len + self.label_len <= len(self.data), \
            f"Index {index} is out of bounds for data with length {len(self.data)}."

        # Create sequences from the data
        seq_x = self.data[index:index + self.seq_len]
        seq_y = self.data[index + self.seq_len:index + self.seq_len + self.label_len]
        
        return seq_x, seq_y


class TokenIdsDataset(Dataset):
    def __init__(self, data, block_size, pad_token = 0):
        """
        A custom dataset class for handling tokenized data. It creates input-output pairs for language modeling.

        Arguments:
        - data: The tokenized dataset (list or tensor) containing token IDs.
        - block_size: The length of the input-output sequences.
        """
        self.data = data
        self.block_size = block_size
        self.pad_token = pad_token

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, pos):
        """
        assert pos < len(self.data) - self.block_size

        x = self.data[pos:pos + self.block_size]
        y = self.data[pos + 1:pos + 1 + self.block_size]
        
        if len(x)< self.block_size:
            x = x + [self.pad_token] * (self.block_size - len(x))
            y = y + [self.pad_token] * (self.block_size - len(y))
        """
        
        if pos + self.block_size > len(self.data):
            # If we're near the end, pad the remaining tokens to make a full block
            x = self.data[pos:] + [self.pad_token] * (self.block_size - (len(self.data) - pos))
            y = self.data[pos+1:] + [self.pad_token] * (self.block_size - (len(self.data) - pos - 1))
        else:
            # Normal case: no need for padding
            x = self.data[pos:pos + self.block_size]
            y = self.data[pos + 1:pos + 1 + self.block_size]
        
        return x, y