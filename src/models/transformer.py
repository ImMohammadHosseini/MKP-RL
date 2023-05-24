
"""

"""
import torch
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from .positional_encoding import PositionalEncoding

class TransformerKnapsack (nn.Module):
    def __init__(
        self,
        config,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.name = 'transformer'
        self.config = config
        self.device = device
        
        self.embed = nn.Linear(self.config.input_dim, self.config.output_dim)
        self.position_encode = PositionalEncoding(self.config.output_dim, 
                                                  self.config.max_length,
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
    
    def generate (
        self,
        external_obs:torch.tensor,
        max_len_generate:int,
        
    ):
        pass
        
     
    def forward (
        self,
        external_obs:torch.tensor,
        encoder_mask:torch.tensor,
        internal_obs:torch.tensor,
        decoder_mask:torch.tensor,
        memory_mask:torch.tensor,
    ):
        external_obs = external_obs.to(torch.float32)
        internal_obs = internal_obs.to(torch.float32)
        encoder_mask = encoder_mask.to(torch.bool)
        decoder_mask = decoder_mask.to(torch.bool)
        memory_mask = memory_mask.to(torch.bool)
        
        self.embed.to(self.device)
        obs_embeding = self.embed(external_obs)
        positional = self.position_encode(obs_embeding)  
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        return self.decoder(internal_obs, self.encoder(positional, encoder_mask), 
                            decoder_mask, memory_mask)
        
    def pad_left(self, sequence, final_length, padding_token):
        return [padding_token] * (final_length - len(sequence)) + sequence
    
    
