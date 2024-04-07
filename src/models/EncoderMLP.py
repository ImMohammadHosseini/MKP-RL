
"""

"""
import torch
import numpy as np
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
)
from .positional_encoding import PositionalEncoding
from typing import List, Optional


class EncoderMLPKnapsack (nn.Module):
    def __init__(
        self,
        config,
        output_type,
        device: torch.device = torch.device("cpu"),
        hidden_dims: Optional[List] = None,
        name = 'transformerEncoderMLP',
    ):
        super().__init__()
        self.config = config
        self.output_type = output_type
        self.name = name
        self.device = device
        self.generate_link_number = self.config.link_number
    
        
        self.en_embed = nn.Linear(self.config.input_encode_dim, self.config.output_dim).to(self.device)
        
        self.en_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     self.config.max_length,
                                                     self.device).to(self.device)
        encoder_layers = TransformerEncoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, self.config.num_encoder_layers
        ).to(self.device)
        
        self.flatten = nn.Flatten().to(self.device)
        
        if hidden_dims == None:
            main_size = self.config.output_dim*self.config.max_length
            hidden_dims = []
            while main_size > self.config.inst_obs_size*self.config.knapsack_obs_size: 
                hidden_dims.append(int(main_size))
                main_size = int(main_size/2)
        modules = []
        input_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        #modules.append(nn.Sequential(nn.Linear(input_dim, self.config.output_dim)))    
        self.mlp = nn.Sequential(*modules).to(self.device)
        
        if self.output_type == 'type2':  
            self.instance_outer = nn.Linear(input_dim//2, self.config.inst_obs_size, device=self.device)
            self.knapsack_outer = nn.Linear(input_dim//2, self.config.knapsack_obs_size, device=self.device)
        elif self.output_type == 'type3':
            self.outer = nn.Linear(input_dim, 
                                   self.config.inst_obs_size*self.config.knapsack_obs_size, device=self.device)
            
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
    def generateOneStep (
        self,
        external_obs: torch.tensor,
        step: int,
        promp_tensor: Optional[torch.tensor] = None,   
    ):
        SOD = [1]*(self.config.input_encode_dim)
        EOD = [2]*(self.config.input_encode_dim)
        PAD = [0]*(self.config.input_encode_dim)
        
        encoder_padding_mask = torch.ones_like(external_obs[:,:,0], device=self.device)
        encoder_padding_mask[torch.all(external_obs.to(self.device) == torch.tensor(PAD, device=self.device), 
                                       dim=2)] = 0
        encoder_padding_mask = encoder_padding_mask.float() 
        encoder_padding_mask = encoder_padding_mask.masked_fill(encoder_padding_mask == 0, float('-inf'))
        encoder_padding_mask = encoder_padding_mask.masked_fill(encoder_padding_mask == 1, float(0.0))
        
        '''encoder_mask = torch.ones_like(external_obs[:,:,0])
        encoder_mask[torch.all(external_obs == torch.tensor(SOD, device=self.device), 
                               dim=2)] = 0
        encoder_mask[torch.all(external_obs == torch.tensor(EOD, device=self.device), 
                               dim=2)] = 0
        encoder_mask = encoder_mask.float() 
        encoder_mask = encoder_mask.masked_fill(encoder_mask == 0, float('-inf'))
        encoder_mask = encoder_mask.masked_fill(encoder_mask == 1, float(0.0))
        
        encoder_mask = torch.cat([encoder_mask]*self.config.nhead , 0)
        
        encoder_mask_sqr = torch.matmul(encoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                        encoder_mask.to(torch.device('cpu')).unsqueeze(1))'''    
        external_obs = external_obs.to(self.device)
        #encoder_mask_sqr = encoder_mask_sqr.to(self.device)
        encoder_padding_mask = encoder_padding_mask.to(self.device)

        firstGenerat, secondGenerat = self.forward(external_obs, encoder_padding_mask=encoder_padding_mask)
        
        return firstGenerat, secondGenerat, None
    
    def forward (
        self,
        external_obs: torch.tensor,
        encoder_mask: Optional[torch.tensor] = None,
        encoder_padding_mask: Optional[torch.tensor] = None,
    ):
        #print(external_obs)
        external_obs = external_obs.to(torch.float32)
        ext_embedding = self.en_embed(external_obs)
        encod = self.encoder(self.en_position_encode(ext_embedding),
                             mask=encoder_mask, 
                             src_key_padding_mask=encoder_padding_mask)
        
        flat = self.flatten(encod)
        flat = self.mlp(flat)

        if self.output_type == 'type2':
            flat_dim = self.config.max_length * self.config.input_encode_dim
            return self.softmax(self.instance_outer(flat[:,:flat_dim//2])), \
                self.softmax(self.knapsack_outer(flat[:,flat_dim//2:]))
                
        elif self.output_type == 'type3':
            return self.softmax(self.outer(flat)), None
        


class RNNMLPKnapsack (nn.Module):
    def __init__(
        self,
        config,
        device: torch.device = torch.device("cpu"),
        hidden_dims: List = [254, 128, 32],
        name = 'transformerEncoderMLP',
    ):
        #torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.name = name
        self.config = config
        self.device = device
        
        #self.en_embed = nn.Linear(self.config.input_dim, self.config.output_dim).to(self.device)
        
        self.lstm = nn.LSTM(input_size=self.config.input_encode_dim, hidden_size=2*self.config.input_encode_dim, 
                            num_layers=2, batch_first=True, bidirectional=True).to(self.device)
        
        self.flatten = nn.Flatten().to(self.device)
        
        modules = []
        input_dim = self.config.max_length * self.config.input_encode_dim * 4
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, self.config.output_dim)))    
        self.mlp = nn.Sequential(*modules).to(self.device)
        
        self.outer = nn.Linear(self.config.output_dim, 
                               self.config.inst_obs_size*self.config.knapsack_obs_size, device=self.device)
        
                
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
    def generateOneStep (
        self,
        external_obs: torch.tensor,
        mode = 'actor',
    ):
        external_obs = external_obs.to(self.device)
        
        generated = self.forward(external_obs)
        return generated
    
    def forward (
        self,
        external_obs: torch.tensor,
        encoder_mask: Optional[torch.tensor] = None,
        encoder_padding_mask: Optional[torch.tensor] = None,
    ):
        external_obs = external_obs.to(torch.float32)
        #ext_embedding = self.en_embed(external_obs)
        encod, _ = self.lstm(external_obs)
        flat = self.flatten(encod)
        
        mlp = self.mlp(flat)
        
        #print('max',self.softmax(self.instance_outer(mlp)).max())
        #print('min',self.softmax(self.instance_outer(mlp)).min())

        return  self.softmax(self.outer(mlp))