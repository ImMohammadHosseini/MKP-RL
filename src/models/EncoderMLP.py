
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
        device: torch.device = torch.device("cpu"),
        hidden_dims: List = [4096, 1024, 512],
    ):
        #torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.name = 'transformerEncoderMLP'
        self.config = config
        self.device = device
        
        self.en_embed = nn.Linear(self.config.input_dim, self.config.output_dim).to(self.device)
        
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
        
        modules = []
        input_dim = self.config.max_length * self.config.output_dim
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, out_features=h_dim),
                    nn.Tanh())
            )
            input_dim = h_dim
        modules.append(nn.Sequential(nn.Linear(input_dim, self.config.output_dim)))    
        self.mlp = nn.Sequential(*modules).to(self.device)
        
        self.instance_outer = nn.Linear(self.config.output_dim, self.config.inst_obs_size).to(self.device)
        self.knapsack_outer = nn.Linear(self.config.output_dim, self.config.knapsack_obs_size).to(self.device)
        
        #self.value_out = nn.Linear(self.config.output_dim, 1)
        
        self.softmax = nn.Softmax(dim=1).to(self.device)
        
    def generateOneStep (
        self,
        external_obs: torch.tensor,
        generat_link_number: int,
        promp_tensor: Optional[torch.tensor] = None,   
        mode = 'actor',
    ):
        PAD1 = [0]*12#self.config.output_dim
        if promp_tensor == None:
            start_tokens = [[PAD1]]*external_obs.size(0)#[[SOD1]]*external_obs.size(0)
            
            
        else: 
            start_tokens = promp_tensor.tolist()
            
        promp_tensor = torch.tensor(
            self.pad_left(
                sequence=start_tokens,
                final_length=
                2*generat_link_number, 
                padding_token=PAD1
                ),
            dtype=torch.float
        )
        promp_tensor = promp_tensor.to(self.device)
        external_obs = external_obs.to(self.device)
        next_instance, next_ks = self.forward(external_obs)
        return next_instance.unsqueeze(1), next_ks.unsqueeze(1), promp_tensor
    
    def forward (
        self,
        external_obs:torch.tensor,
    ):
        external_obs = external_obs.to(torch.float32)
        ext_embedding = self.en_embed(external_obs)
        encod = self.encoder(self.en_position_encode(ext_embedding))
        flat = self.flatten(encod)
        mlp = self.mlp(flat)
        
        #print('max',self.softmax(self.instance_outer(mlp)).max())
        #print('min',self.softmax(self.instance_outer(mlp)).min())

        return  self.softmax(self.instance_outer(mlp)), \
                self.softmax(self.knapsack_outer(mlp))
                
        
    def pad_left(self, sequence, final_length, padding_token):
        pads = [[padding_token] * (final_length - len(sequence[:][0]))] * len(sequence)
        return [pads[i]+sequence[i] for i in range (len(pads))]
        
        
        