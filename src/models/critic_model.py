
"""

"""

import torch
from torch import nn
from typing import  List

class CriticNetwork (nn.Module):
    def __init__ (self, ext_input_dim, int_input_dim, 
                  hidden_dims: List = [512, 128, 16]):
        super().__init__()
        
        self.name = 'mlp_cretic'
        self.flatten = nn.Flatten()
        self.ext_input = nn.Sequential(
            nn.Linear(ext_input_dim, 512),
            nn.ReLU())
        self.int_input = nn.Sequential(
            nn.Linear(int_input_dim, 512),
            nn.ReLU())
        
        modules = []
        input_dim = 2*512
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic = nn.Sequential(*modules)
    
    def forward(self, external, internal):
        ext_part = self.ext_input(self.flatten(external))
        int_part = self.int_input(self.flatten(internal))
        
        return self.critic(torch.cat([ext_part, int_part], 1))
    
class ExternalCriticNetwork (nn.Module):
    def __init__ (
        self, 
        ext_input_dim: int, 
        hidden_dims: List = [512, 128, 16]
    ):
        super().__init__()
        self.name = 'mlp_cretic' 
        
        modules = []
        input_dim = ext_input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic = nn.Sequential(*modules)
    
    def forward(self, external):
        return self.critic(external)
    
    
    
    
    
    