
"""

"""
import torch
from torch import nn
from transformers import PreTrainedModel
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from transformers.configuration_utils import PretrainedConfig

class TransformerKnapsack (nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.embed = nn.Linear(self.config.input_dim, self.config.output_dim)
        
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
    
    def forward (
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
        return generate
    
