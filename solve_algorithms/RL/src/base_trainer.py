
"""

"""



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

    def internal_reward (self, *args):
        raise NotImplementedError("Not implemented")

    def external_train (self, *args):
        raise NotImplementedError("Not implemented")
        
    def load_models (self, *args):
        raise NotImplementedError("Not implemented")
        
    def save_models (self, *args):
        raise NotImplementedError("Not implemented")