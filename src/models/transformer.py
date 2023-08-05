
"""

"""
import torch
import math
import numpy as np
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from .positional_encoding import PositionalEncoding
from typing import Optional


class TransformerKnapsack (nn.Module):
    def __init__(
        self,
        config,
        generate_link_number: int,
        device: torch.device = torch.device("cpu"),
        name = 'transformer',
    ):
        #torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.name = name
        self.config = config
        self.device = device
        self.generate_link_number = generate_link_number
        
        self.en_embed = nn.Linear(self.config.input_encode_dim, self.config.output_dim)
        self.de_embed = nn.Linear(self.config.input_decode_dim, self.config.output_dim)

        self.en_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     self.config.max_length,
                                                     self.device)
        self.de_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     generate_link_number+1,
                                                     self.device)
        
        encoder_layers = TransformerEncoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, self.config.num_encoder_layers
        )
        decoder_layers = TransformerDecoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.decoder = TransformerDecoder(
            decoder_layers, self.config.num_decoder_layers
        )
        
        self.instance_outer = nn.Linear(self.config.output_dim//2, self.config.inst_obs_size)
        self.knapsack_outer = nn.Linear(self.config.output_dim//2, self.config.knapsack_obs_size)
        
        
        self.softmax = nn.Softmax(dim=1)
    
    def generateOneStep (
        self,
        step: int,
        external_obs: torch.tensor,
        promp_tensor: Optional[torch.tensor] = None,   
        mode = 'actor',
    ):
        SOD1 = [1]*self.config.input_decode_dim
        PAD1 = [0]*self.config.input_decode_dim

        
        if promp_tensor == None:
            start_tokens = [[SOD1]]*external_obs.size(0)
            
        else: 
            start_tokens = promp_tensor.tolist()
        
        promp_tensor = torch.tensor(
            self.padding(
                sequence=start_tokens,
                final_length=
                self.generate_link_number+1, 
                padding_token=PAD1
                ),
            dtype=torch.float,
            device=self.device
        )
        internal_obs = promp_tensor
        
        
        tgt_mask = torch.tril(torch.ones(self.config.nhead*internal_obs.size(0), 
                                            self.generate_link_number+1,
                                            self.generate_link_number+1) == 1) 
        tgt_mask = tgt_mask.float() 
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.device)
        
        decoder_padding_mask = torch.ones_like(internal_obs[:,:,0])
        decoder_padding_mask[torch.all(internal_obs == torch.tensor(PAD1, 
                                                                    device=self.device), 
                                       dim=2)] = 0
        decoder_padding_mask = decoder_padding_mask.float() 
        decoder_padding_mask = decoder_padding_mask.masked_fill(decoder_padding_mask == 0, float('-inf'))
        decoder_padding_mask = decoder_padding_mask.masked_fill(decoder_padding_mask == 1, float(0.0))
        
        external_obs = external_obs.to(self.device)
        internal_obs = internal_obs.to(self.device)
        
        if mode == 'actor':
            next_instance, next_ks = self.forward(external_obs,
                                                  internal_obs, tgt_mask, 
                                                  decoder_padding_mask,
                                                  step)
        
            return next_instance.unsqueeze(1), next_ks.unsqueeze(1), promp_tensor
        '''elif mode == 'ref':
            return self.forward(external_obs, encoder_mask_sqr, encoder_padding_mask,
                                internal_obs, decoder_mask, decoder_padding_mask, 
                                memory_mask, mode)'''
    def forward (
        self,
        external_obs:torch.tensor,
        internal_obs:torch.tensor,
        tgt_mask:torch.tensor,
        decoder_padding_mask:torch.tensor,
        step: Optional[int] = None,
        mode = 'RL_train',
    ):
        external_obs = external_obs.to(torch.float32)
        internal_obs = internal_obs.to(torch.float32)
        #decoder_mask = decoder_mask.to(torch.bool)
        self.en_embed=self.en_embed.to(self.device)
        self.de_embed=self.de_embed.to(self.device)

        ext_embedding = self.en_embed(external_obs)* math.sqrt(self.config.output_dim)
        int_embedding = self.de_embed(internal_obs)* math.sqrt(self.config.output_dim)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        encod = self.encoder(self.en_position_encode(ext_embedding))

        transformer_out = self.decoder(self.de_position_encode(int_embedding), 
                                       encod, tgt_mask=tgt_mask, 
                                       tgt_key_padding_mask=decoder_padding_mask)
        self.instance_outer = self.instance_outer.to(self.device)
        self.knapsack_outer = self.knapsack_outer.to(self.device)
        
        if mode == 'RL_train':
            pos = torch.cat([step.unsqueeze(0)]*self.config.output_dim,0).T.unsqueeze(1).to(self.device)
            out = transformer_out.gather(1,pos).squeeze(1)
            
            return self.softmax(self.instance_outer(out[:,:self.config.output_dim//2])), \
                    self.softmax(self.knapsack_outer(out[:,self.config.output_dim//2:]))
        
        
        elif mode == 'transformer_train':
            return self.instance_outer(transformer_out[:,1:,:self.config.output_dim//2]), \
                    self.knapsack_outer(transformer_out[:,1:,self.config.output_dim//2:])
        
    def padding(self, sequence, final_length, padding_token):
        pads = [[padding_token] * (final_length - len(sequence[:][0]))] * len(sequence)
        return [sequence[i]+pads[i] for i in range (len(pads))]
    
