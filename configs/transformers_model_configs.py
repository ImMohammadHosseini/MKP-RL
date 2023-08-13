
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
        output_dim: int = 32,
        nhead: int = 4,
        d_hid: int = 16,
        num_encoder_layers: int = 10,
        num_decoder_layers: int = 12,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        self.inst_obs_size = inst_obs_size
        self.knapsack_obs_size = knapsack_obs_size
        self.problem_dim = problem_dim
        self.input_encode_dim = problem_dim + 1
        self.input_decode_dim = 2 * self.input_encode_dim
        self.output_dim = output_dim
        self.max_length = self.inst_obs_size + self.knapsack_obs_size + 3
        self.nhead = nhead
        self.batch_first = batch_first
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        super(TransformerKnapsackConfig, self).__init__(**kwargs)