
"""

"""
import torch
import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase
from typing import Callable, List, Optional, Union
from datasets import Dataset

import gymnasium as gym
from gymnasium import spaces
from src.data_structure.state_prepare import ExternalStatePrepare
from torchrl.envs import (
    EnvBase,
)

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        config,
        info:dict,
        state_dataClass: ExternalStatePrepare,
        device = "cpu",
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
                                       shape=(state_dataClass.knapsackObsSize,
                                              config.dim), dtype=int),
                "instance_value": spaces.Box(low=np.array(
                    [info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['VALUE_LOW']]), 
                                       high=np.array(
                    [info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['VALUE_HIGH']]), 
                                       shape=(state_dataClass.instanceObsSize,
                                              config.dim), dtype=int),
            }
        )
        self.action_space = spaces.Box(low=np.array([[0,0]] * \
                                                    state_dataClass.knapsackObsSize), 
                                       high=np.array([[state_dataClass.knapsackObsSize, 
                                                       state_dataClass.instanceObsSize]] * \
                                                     state_dataClass.knapsackObsSize), 
                                       dtype=int
        )
        
        super().__init__()
        self.config = config
        self.device = device
        self.statePrepare = state_dataClass
    
    def _get_obs(self) -> dict:
        stateCaps, stateWeightValues = self.statePrepare.getObservation()
        
        return {"knapsack": stateCaps, "instance_value": stateWeightValues}
        
    def _get_info(self):
        return ""
        
        
    def reset (self): 
        #super().reset(seed=seed)
        self.statePrepare.reset()
        externalObservation = self._get_obs()
        ACT = np.zeros((1,self.config.dim))
        externalObservation = np.append(ACT, np.append(np.append(externalObservation[
            "instance_value"], ACT, axis=0), np.append(externalObservation["knapsack"], 
                                                       ACT, axis=0),axis=0),
                                                       axis=0)
        info = self._get_info()
        externalObservation = torch.tensor(externalObservation).unsqueeze(dim=0)        
        return externalObservation, info
    
    def step (self, actions):
        self.statePrepare.changeNextState(actions) 
        terminated = self.statePrepare.is_terminated() #TODO add termination condition 
        
        if terminated:
            externalReward = self.reward_function(actions)
        else: externalReward = 0
        
        
        externalObservation = self._get_obs()
        ACT = np.zeros((1,self.config.dim))
        externalObservation = np.append(ACT, np.append(np.append(externalObservation[
            "instance_value"], ACT, axis=0), np.append(externalObservation["knapsack"], 
                                                       ACT, axis=0),axis=0),
                                                       axis=0)
        externalObservation = torch.tensor(externalObservation).unsqueeze(dim=0)
                                                       
                                                       
        info = self._get_info()
        
        return externalObservation, externalReward, terminated, info
    
    def response_decode (self, responce):
        pass
    
    def reward_function (self, actions):
        pass
    
    
    
    
    
    