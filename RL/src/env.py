
"""

"""
import torch
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from accelerate import Accelerator
from typing import Callable, List, Optional, Union
from datasets import Dataset

import gymnasium as gym
from gymnasium import spaces

from torchrl.envs import (
    EnvBase,
)

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        config,
        info:dict,
        knapsack_batch_size : int,
        instance_batch_size : int,
        device = "cpu",
        #dataset: Union[torch.utils.data.Dataset, Dataset] = None,
        knapsack: Union[torch.utils.data.Dataset, Dataset] = None,
        inst_value: Union[torch.utils.data.Dataset, Dataset] = None,
    ):
        '''self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            **config.accelerator_kwargs,
        )
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=dict(trl_ppo_trainer_config=config.to_dict()),
            init_kwargs=config.tracker_kwargs,
        )'''
        
        self.observation_space = spaces.Dict(
            {
                "knapsack": spaces.Box(low=np.array(
                    [info['CAP_LOW'], info['CAP_LOW'], info['CAP_LOW'], 
                     info['CAP_LOW'], info['CAP_LOW']]), 
                                       high=np.array(
                    [info['CAP_HIGH'], info['CAP_HIGH'], info['CAP_HIGH'],
                     info['CAP_HIGH'], info['CAP_HIGH']]), 
                                       shape=(2,5), dtype=int),
                "instanc_value": spaces.Box(low=np.array(
                    [info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['VALUE_LOW']]), 
                                       high=np.array(
                    [info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['VALUE_HIGH']]), 
                                       shape=(2,6), dtype=int),
            }
        )
        self.action_space = spaces.Box(low=np.array([[0,0]]*len(knapsack)), 
                                       high=np.array([[len(knapsack), 
                                                       len(inst_value)]]*len(knapsack)), 
                                       dtype=int
        )
        
        #self.input_spec 
        #self.reward_spec 
        #self.batch_size 
        
        super().__init__()
        
        self.device = device
        self.knapsack_batch_size = knapsack_batch_size
        self.instance_batch_size = instance_batch_size
        self.knapsack = knapsack
        self.inst_value = inst_value
        #self.values = values
        
        
        #self.dataset = dataset
        #self._signature_columns = None
        #self.dataloader = self.prepare_dataloader(self.dataset)
        
        self.current_step = 0
        
    def prepare_dataloader(
        self, 
        dataset: Union[torch.utils.data.Dataset, Dataset],
        batch_size: int
    ):
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        return dataloader   
    
    def _get_obs(self):
        return {"knapsack": self._agent_location, 
                "instanc_value": self._target_location}
    
    def _get_info(self):
        pass
    
    def reset (self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase: 
        #super().reset(seed=seed)
        observation = self._get_obs()
    
    def step (
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        reward = reward_function()
        out = TensorDict(
            {
                "next": {
                    "th": new_th,
                    "thdot": new_thdot,
                    "params": tensordict["params"],
                    "reward": reward,
                    "done": done,
                }
            },
            tensordict.shape,
        )
        return out
    
    def reward_function (self,d):
        pass
    
    #def _set_seed (self, seed: Optional[int]):
    #    rng = torch.manual_seed(seed)
    #    self.rng = rng
    
    #def rollout (self):
    #    pass