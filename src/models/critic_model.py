
"""

"""

import torch
from torch import nn


class CriticNetwork (nn.Module):
    def __init__ (self, input_dim, alpha, hidden_dims = []):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        
        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_channels=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
            
        self.critic = nn.Sequential(*modules)
    
    def forward(self, state):
        return self.critic(state)
    
    