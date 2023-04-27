
"""

"""
import torch
from typing import Callable, List, Optional, Union
from datasets import Dataset
from torchrl.envs import (
    EnvBase,
)

class KnapsackAssignmentEnv(EnvBase):
    def __init__ (
            self,
            dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
            ):
        pass
    def _reset (self):
        pass
    def _step (self):
        pass
    
    def _set_seed (self):
        pass