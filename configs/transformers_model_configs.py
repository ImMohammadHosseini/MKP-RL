
"""

"""

from dataclasses import dataclass

@dataclass
class TransformerKnapsackConfig (object):
    model_type = "TransformerKnapsack"

    def __init__(
        self,
        inst_obs_size: int,
        knapsack_obs_size: int,
        problem_dim: int,
        device,
        link_number:int,
        output_dim: int,
        nhead: int = 1,
        d_hid: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 8,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        self.inst_obs_size = inst_obs_size
        self.knapsack_obs_size = knapsack_obs_size
        self.problem_dim = problem_dim
        self.device = device
        self.link_number = link_number
        self.input_encode_dim = problem_dim+1 #2*(problem_dim+1)#problem_dim*(1+knapsack_obs_size)+1
        self.input_decode_dim = 2 * self.input_encode_dim
        self.output_dim = output_dim
        self.max_length = inst_obs_size+knapsack_obs_size+3#2*(inst_obs_size*knapsack_obs_size)+4#self.inst_obs_size + self.knapsack_obs_size + 3
        self.nhead = nhead
        self.batch_first = batch_first
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        super(TransformerKnapsackConfig, self).__init__(**kwargs)