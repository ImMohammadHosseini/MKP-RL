
"""

"""
import torch
from typing import Callable, List, Optional, Union
from transformers import PreTrainedModel
from datasets import Dataset


from ..base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig

class PPOTrainer(BaseTrainer):
    r"""
    this implementation is inspired by: https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_trainer.py
    """
    
    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModel = None,
        #ref_model: PreTrainedModelWrapper = None,
        #tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator=None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        pass
    
    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        pass
    
    def generate(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        pass
    
    def _generate_batched(
        self,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        **generation_kwargs,
    ):
        pass
    
    @PPODecorators.empty_cuda_cache()
    def step(
        self,
        queries: List[torch.LongTensor],
        responses: List[torch.LongTensor],
        scores: List[torch.FloatTensor],
    ):
        pass
    
    def _early_stop(self, policykl):
        pass
    
    def gather_stats(self, stats):
        pass
    
    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: PreTrainedModel,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        pass
    
    @PPODecorators.empty_cuda_cache()
    def train_minibatch(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        pass
    
    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        pass
    
    