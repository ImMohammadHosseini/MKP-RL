
"""

"""
from torch import nn
from transformers import PreTrainedModel
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from transformers.configuration_utils import PretrainedConfig

class TransformerKnapsack (PreTrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
    ):
        super().__init__(config)
        
        #TODO
        encoder_layers = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
            batch_first=config.batch_first,
        )
        self.pick_encoder = TransformerEncoder(
            encoder_layers, config.num_encoder_layers
        )
        decoder_layers = TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.d_hid,
            dropout=config.dropout,
            batch_first=config.batch_first,
        )
        self.place_decoder = TransformerDecoder(
            decoder_layers, config.num_decoder_layers
        )
    
    def forward (
        self,
        instances,
        knapsack,
        src_key_padding_mask,
        device,
    ):
        pass