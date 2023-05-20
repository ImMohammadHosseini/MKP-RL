
"""

"""
import torch
from torch import nn
import numpy as np

class PositionalEncoding(nn.Module):
    """
    this implementation is inspired by: 
        https://wingedsheep.com/building-a-language-model/
    """
    def __init__(self, embed_dimension, max_length, device):
        super().__init__()
        self.embed_dimension = embed_dimension
        self.max_length = max_length
        self.positionalEncoding = self.build_positional_encoding()
        self.device = device
        
    def build_positional_encoding (self):
        positional_encoding = np.zeros((self.max_length, self.embed_dimension))
        for pos in range(self.max_length):
            for i in range(0, self.embed_dimension, 2):
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.embed_dimension)))
                positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dimension)))
        return torch.from_numpy(positional_encoding).float()
    
    def forward (self, x):
        return x + self.positionalEncoding[:x.size(1), :].to(self.device)