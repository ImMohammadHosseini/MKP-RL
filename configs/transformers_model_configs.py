
"""

"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class TransformerKnapsackConfig (object):
    model_type = "TransformerKnapsack"

    def __init__(
        self,
        inst_obs_size: int,
        knapsack_obs_size: int,
        vector_dim: int,
        device,
        link_number:int,
        output_dim: int,
        nhead: int = 4,
        d_hid: Optional[int] = None,
        num_encoder_layers: Optional[int] = None,
        num_decoder_layers: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        self.inst_obs_size = inst_obs_size
        self.knapsack_obs_size = knapsack_obs_size
        self.vector_dim = vector_dim
        self.device = device
        self.link_number = link_number
        self.input_encode_dim = vector_dim
        self.input_decode_dim = 2 * self.input_encode_dim
        self.output_dim = 16*(vector_dim) #15*(vector_dim) if vector_dim%2==0 else 15*(vector_dim)+1 #
        self.max_length = inst_obs_size+knapsack_obs_size+3
        self.nhead = nhead 
        self.batch_first = batch_first
        self.d_hid = 2*self.output_dim if d_hid == None else d_hid	
        #self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = 10+self.vector_dim if num_encoder_layers == None else num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        super(TransformerKnapsackConfig, self).__init__(**kwargs)