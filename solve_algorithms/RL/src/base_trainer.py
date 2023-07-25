
"""

"""
import numpy as np
from typing import Optional


class BaseTrainer ():
    r"""
    this implementation is copied from: https://github.com/lvwerra/trl/blob/main/trl/trainer/base.py
    Base class for all trainers - this base class implements the basic functions that we
    need for a trainer.
    Each user is expected to implement their own trainer class that inherits from this base
    if they want to use a new training algorithm.
    """

    def __init__ (self, config):
        self.config = config

    def steps (self, *args):
        raise NotImplementedError("Not implemented")
        
    def test_step (self, *args):
        raise NotImplementedError("Not implemented")

    def loss (self, *args):
        raise NotImplementedError("Not implemented")
    
    def generate_batch (
        self, 
        n_states: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if batch_size == None:
            batch_size = self.config.ppo_batch_size
        if n_states == None:
            n_states = self.config.internal_batch
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return batches
    
    def internal_reward (self, *args):
        raise NotImplementedError("Not implemented")

    def external_train (self, *args):
        raise NotImplementedError("Not implemented")
        
    def load_models (self, *args):
        raise NotImplementedError("Not implemented")
        
    def save_models (self, *args):
        raise NotImplementedError("Not implemented")