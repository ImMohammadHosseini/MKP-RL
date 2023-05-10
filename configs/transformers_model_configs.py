
"""

"""

from dataclasses import dataclass

@dataclass
class TransformerKnapsackConfig (object):
    model_type = "TransformerKnapsack"

    def __init__(
        self,
        inst_obs_size: int = 120,
        knapsack_obs_size: int = 60,
        problem_dim: int = 5,
        nhead: int = 5,
        d_hid: int = 16,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dropout: float = 0.1,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        self.inst_obs_size = inst_obs_size
        self.knapsack_obs_size = knapsack_obs_size
        self.input_dim = problem_dim + 1
        self.output_dim = 2 * problem_dim
        self.max_length = self.inst_obs_size + self.knapsack_obs_size + 3
        self.nhead = nhead
        self.batch_first = batch_first
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        
        super(TransformerKnapsackConfig, self).__init__(**kwargs)