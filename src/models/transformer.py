
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
        obs_tensor:torch.tensor,
        encoder_mask:torch.tensor,
        max_len_generate:int,
        
    ):
        ACT = [0]*self.config.input_dim
        obs_tensor = obs_tensor.unsqueeze(dim=0)
        start_tokens = [ACT]
        prompt_tensor = torch.tensor(
            self.pad_left(
                sequence=start_tokens,
                final_length=self.config.max_length,
                padding_token=ACT
            ),
            dtype=torch.long
        )
        prompt_tensor = prompt_tensor.unsqueeze(dim=0)
        
        out = prompt_tensor
        
        for _ in range(max_len_generate):
            x = out[:,-(self.config.max_length-1):,:]
            
            decoder_mask = torch.ones_like(x[:,:,0])
            decoder_mask[torch.all(x == torch.tensor(ACT), dim=2)] = 0
            
            next_ = self.forward(obs_tensor, encoder_mask, x, 
                                        decoder_mask)
            out = torch.cat([out, next_], dim=1)
            
        return out[0]
     
    def forward (
        self,
        obs_tensor:torch.tensor,
        encoder_mask:torch.tensor,
        prompt:torch.tensor,
        decoder_mask:torch.tensor,
    ):
        obs_embeding = self.embed(obs_tensor)
        positional = self.position_encode(obs_embeding)
        return self.decoder(prompt, self.encoder(positional, encoder_mask), 
                            decoder_mask, encoder_mask)
        
    def pad_left(self, sequence, final_length, padding_token):
        return [padding_token] * (final_length - len(sequence)) + sequence
    
    '''def forward ( 
        self,
        obs_tensor,
        number_of_pairs
    ):
        size= self.config.inst_obs_size + self.config.knapsack_obs_size + 2
        generate = torch.tensor(torch.zeros((1,self.config.output_dim)))
        
        for i in range(number_of_pairs + 1):
            model_input = torch.cat((generate,
                                    torch.ones(size-i,self.config.output_dim) * 
                                    float('-inf')),0)
            out = self.decoder(model_input.unsqueeze(dim=0), self.encoder(
                self.embed(obs_tensor.unsqueeze(dim=0))))
            
            generate = torch.cat((generate, out[0][:i+1].mean(0).unsqueeze(dim=0)),0)
        return generate'''
    
