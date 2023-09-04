
"""

"""

import torch
from torch import nn
from typing import List

class CriticNetwork (nn.Module):
    def __init__ (self, external_max_length, ext_input_dim, internal_max_length,
                  int_input_dim, 
                  hidden_dims: List = [512, 128, 16],
                  name = 'mlp_cretic'):
        super().__init__()
        
        self.name = name
        self.flatten = nn.Flatten()
        
        '''modules = []
        input_dim = ext_input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_eo = nn.Sequential(*modules)'''
        self.lstm_eo = nn.LSTM(input_size=ext_input_dim, hidden_size=2*ext_input_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        #sd = self.lstm.state_dict()
        self.flatten = nn.Flatten()
        modules = []
        input_dim = ext_input_dim*external_max_length*4
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_eo = nn.Sequential(*modules)
        
        
        '''modules = []
        input_dim = int_input_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_io = nn.Sequential(*modules)'''
        self.lstm_io = nn.LSTM(input_size=int_input_dim, hidden_size=2*int_input_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        #sd = self.lstm.state_dict()
        modules = []
        input_dim = int_input_dim*internal_max_length*4
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_io = nn.Sequential(*modules)
        
    
    def forward(self, external, internal):
        #ext_part = self.critic_eo(self.flatten(external))
        #int_part = self.critic_io(self.flatten(internal))
        lstm, _ = self.lstm_eo(external)
        ext_part = self.critic_eo(self.flatten(lstm))
        
        lstm, _ = self.lstm_io(internal)
        int_part = self.critic_io(self.flatten(lstm))
        return ext_part + int_part
    
class ExternalCriticNetwork (nn.Module):
    def __init__ (
        self, 
        max_length: int,
        input_dim: int, 
        hidden_dims: List = [256, 128, 128, 64, 16],
        name = 'mlp_external_cretic',
    ):
        super().__init__()
        self.name = name 
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=2*input_dim, 
                            num_layers=1, batch_first=True, bidirectional=True)
        #sd = self.lstm.state_dict()
        self.flatten = nn.Flatten()
        modules = []
        input_dim = input_dim*max_length
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic = nn.Sequential(*modules)
    
    def forward(self, external):
        #print(external.size())
        #lstm, _ = self.lstm(external)
        #print(self.flatten(lstm).size())
        return self.critic(self.flatten(external))
    
    
    
    
    
    