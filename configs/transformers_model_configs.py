
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
        nhead: int = 4,
        d_hid: int = 32,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        first_train_lr = 0.01,
        first_train_epoc = 100, 
        first_train_batch_size = 8,
        batch_first: bool = True,
        **kwargs,
    ) -> None:
        self.inst_obs_size = inst_obs_size
        self.knapsack_obs_size = knapsack_obs_size
        self.problem_dim = problem_dim
        self.input_dim = problem_dim + 1
        self.output_dim = 32#2 * self.input_dim
        self.max_length = self.inst_obs_size + self.knapsack_obs_size + 3
        self.nhead = nhead
        self.batch_first = batch_first
        self.d_hid = d_hid
        self.dropout = dropout
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.first_train_lr = first_train_lr
        self.first_train_epoc = first_train_epoc
        self.first_train_batch_size = first_train_batch_size
        
        super(TransformerKnapsackConfig, self).__init__(**kwargs)