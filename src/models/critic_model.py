
"""

"""

import torch
from torch import nn
from typing import List, Optional 

class CriticNetwork1 (nn.Module):
    def __init__ (self, external_max_length, ext_input_dim, internal_max_length,
                  int_input_dim, device,
                  hidden_dims: List = [512, 256, 128, 128, 64, 16],
                  name = 'mlp_cretic'):
        super().__init__()
        
        self.name = name
        self.flatten = nn.Flatten()
        self.device =device
        
        #self.lstm_eo = nn.LSTM(input_size=ext_input_dim, hidden_size=2*ext_input_dim, 
        #                    num_layers=1, batch_first=True, bidirectional=True)
        #sd = self.lstm.state_dict()
        modules = []
        input_dim = (ext_input_dim*external_max_length)+(int_input_dim*internal_max_length)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, 1)))    
        self.critic_eo = nn.Sequential(*modules).to(device)
        
        '''self.lstm_io = nn.LSTM(input_size=int_input_dim, hidden_size=2*int_input_dim, 
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
        self.critic_io = nn.Sequential(*modules)'''
        
    
    def forward(self, external, internal):
        #ext_part = self.critic_eo(self.flatten(external))
        #int_part = self.critic_io(self.flatten(internal))
        #lstm, _ = self.lstm_eo(external)
        input_tensor = torch.cat([external.flatten(start_dim=1),internal.flatten(start_dim=1)],1)
        #ext_part = self.critic_eo(self.flatten(lstm))
        
        #lstm, _ = self.lstm_io(internal)
        #int_part = self.critic_io(self.flatten(lstm))
        #return ext_part + int_part
        return self.critic_eo(input_tensor.to(self.device))

    
class CriticNetwork2 (nn.Module):
    def __init__ (
        self, 
        max_length: int,
        input_dim: int, 
        device,
        hidden_dims: Optional[List] = None,
        out_put_dim:int = 1,
        name = 'mlp_cretic',
    ):
        super().__init__()
        self.name = name 
        self.device = device
        self.to(device)
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=2*input_dim, 
                            num_layers=2, batch_first=True, bidirectional=True).to(self.device)
        self.flatten = nn.Flatten().to(device)
        modules = []
        input_dim = input_dim*max_length*4
        if hidden_dims == None:
            main_size = 2*input_dim
            hidden_dims = []
            last_layer = 8 if out_put_dim == 1 else out_put_dim
            while main_size > last_layer: 
                hidden_dims.append(int(main_size))
                main_size = int(main_size/2)
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.ReLU())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, out_put_dim)))    
        self.critic = nn.Sequential(*modules).to(device)
    
    def forward(self, external, *args):
        encod, _ = self.lstm(external)

        return self.critic(self.flatten(encod))
    
    
    
    
    
    