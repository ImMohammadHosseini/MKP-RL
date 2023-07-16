
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
        self.critic_eo = nn.Sequential(*modules)
        
        
        modules = []
        input_dim = int_input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_io = nn.Sequential(*modules)
    
    def forward(self, external, internal):
        #ext_part = self.critic_eo(self.flatten(external))
        int_part = self.critic_io(self.flatten(internal))
        
        return int_part#(0.5*ext_part) + (0.5*int_part)
    
class ExternalCriticNetwork (nn.Module):
    def __init__ (
        self, 
        ext_input_dim: int, 
        hidden_dims: List = [512, 128, 16]
    ):
        super().__init__()
        self.name = 'mlp_external_cretic' 
        
        self.flatten = nn.Flatten()
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
        return self.critic(self.flatten(external))
    
    
    
    
    
    