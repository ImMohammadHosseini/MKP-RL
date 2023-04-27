
"""

"""

from huggingface_hub import PyTorchModelHubMixin


class BaseTrainer(PyTorchModelHubMixin):
    r"""
    this implementation is copied from: https://github.com/lvwerra/trl/blob/main/trl/trainer/base.py
    Base class for all trainers - this base class implements the basic functions that we
    need for a trainer.
    The trainer needs to have the following functions:
        - step: takes in a batch of data and performs a step of training
        - loss: takes in a batch of data and returns the loss
        - compute_rewards: takes in a batch of data and returns the rewards
        - _build_models_and_tokenizer: builds the models and tokenizer
        - _build_dataset: builds the dataset
    Each user is expected to implement their own trainer class that inherits from this base
    if they want to use a new training algorithm.
    """

    def __init__(self, config):
        self.config = config

    def step(self, *args):
        raise NotImplementedError("Not implemented")

    def loss(self, *args):
        raise NotImplementedError("Not implemented")

    def compute_rewards(self, *args):
        raise NotImplementedError("Not implemented")

    def _save_pretrained(self, save_directory):
        raise NotImplementedError("Not implemented")