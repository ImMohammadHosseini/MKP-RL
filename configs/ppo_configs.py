
"""

"""

from dataclasses import dataclass, field
from typing import Optional



@dataclass
class PPOConfig(object):
    """
    this implementation is inspired by: https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_config.py
    Configuration class for PPOTrainer
    """
    steps: Optional[int] = field(default=20000, metadata={"help": "Number of training steps"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "Adam learning rate"})
    adap_kl_ctrl: Optional[bool] = field(default=True, metadata={"help": "Use adaptive KL control, otherwise linear"})
    init_kl_coef: Optional[float] = field(
        default=0.2,
        metadata={"help": "Initial KL penalty coefficient (used for adaptive and linear control)"},
    )
    target: Optional[float] = field(default=6, metadata={"help": "Target KL value for adaptive KL control"})
    horizon: Optional[float] = field(default=10000, metadata={"help": "Horizon for adaptive KL control"})
    gamma: Optional[float] = field(default=1, metadata={"help": "Gamma parameter for advantage calculation"})
    lam: Optional[float] = field(default=0.95, metadata={"help": "Lambda parameter for advantage calculation"})
    cliprange: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping in PPO policy gradient loss"}
    )
    cliprange_value: Optional[float] = field(
        default=0.2, metadata={"help": "Range for clipping values in loss calculation"}
    )
    vf_coef: Optional[float] = field(default=0.1, metadata={"help": "Scaling factor for value loss"})
    batch_size: Optional[int] = field(default=1, metadata={"help": "Number of samples per optimisation step"})
    forward_batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of samples forward passed through model at a time"},
    )
    mini_batch_size: Optional[int] = field(
        default=1, metadata={"help": "Number of samples optimized inside PPO together"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "The number of gradient accumulation steps"}
    )
    ppo_epochs: Optional[int] = field(
        default=4,
        metadata={"help": "Number of optimisation epochs per batch of samples"},
    )
    
    max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Maximum gradient norm for gradient clipping"}
    )
    seed: Optional[int] = field(default=0, metadata={"help": "Seed value for random generations"})
    
    early_stopping: Optional[bool] = field(
        default=False, metadata={"help": "Whether to stop the PPO optimization loop early is the KL too high"}
    )
    target_kl: Optional[float] = field(
        default=0.1, metadata={"help": "Stop early if we exceed this value by over 50%"}
    )
    compare_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Number of steps between comparison of the current reward with the best seen so far"},
    )
    