
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
    ):
        super().__init__()
        self.config = config
        self.embed = nn.Linear(self.config.input_dim, self.config.output_dim)
        self.position_encode = PositionalEncoding(self.config.output_dim, 
                                                  self.config.max_length)
        
        encoder_layers = TransformerEncoderLayer(
            d_model=self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, self.config.num_encoder_layers
        )
        decoder_layers = TransformerDecoderLayer(
            d_model=self.config.output_dim,
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
        encoder_ACT = [0]*self.config.input_dim
        encoder_mask = torch.ones_like(external_obs[:,:,0])
        encoder_mask[torch.all(external_obs == torch.tensor(encoder_ACT), dim=2)] = 0

        decoder_ACT = [0]*self.config.output_dim
        start_tokens = [decoder_ACT]
        prompt_tensor = torch.tensor(
            self.pad_left(
                sequence=start_tokens,
                final_length=max_len_generate + 1,
                padding_token=decoder_ACT
            ),
            dtype=torch.long
        )
        prompt_tensor = prompt_tensor.unsqueeze(dim=0)
        
        out = prompt_tensor
        
        for _ in range(max_len_generate):
            internal_obs = out[:,-(max_len_generate+1):,:]
            
            decoder_mask = torch.ones_like(internal_obs[:,:,0])
            decoder_mask[torch.all(internal_obs == torch.tensor(decoder_ACT), dim=2)] = 0

            memory_mask = torch.matmul(decoder_mask.T.long(), encoder_mask.long())
            encoder_mask_sqr = torch.matmul(encoder_mask.T, encoder_mask)
            decoder_mask = torch.matmul(decoder_mask.T, decoder_mask)

            next_ = self.forward(external_obs, encoder_mask_sqr, internal_obs, 
                                 decoder_mask, memory_mask)
            out = torch.cat([out, torch.mean(next_, 1).unsqueeze(dim=0)], dim=1)
        out = out[0][max_len_generate+1:]
        #TODO check transformer encoder and change in mean
        return out
     
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
        
        obs_embeding = self.embed(external_obs)
        positional = self.position_encode(obs_embeding)
        return self.decoder(internal_obs, self.encoder(positional, encoder_mask), 
                            decoder_mask, memory_mask)
        
    def pad_left(self, sequence, final_length, padding_token):
        return [padding_token] * (final_length - len(sequence)) + sequence
    
    
