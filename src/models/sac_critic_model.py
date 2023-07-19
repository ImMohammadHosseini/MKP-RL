
"""

"""
from torch import nn
from typing import List

class SACCriticNetwork (nn.Module):
    def __init__ (
        self, 
        max_length: int,
        input_dim: int, 
        hidden_dims: List = [512, 128, 16]
    ):
        super().__init__()
        self.name = 'sac_critic_network' 
        
        
    
    
    def forward (
        self,
        external,
        actions,
    ):
        