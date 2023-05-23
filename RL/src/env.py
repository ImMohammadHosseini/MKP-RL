
"""

"""
import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from src.data_structure.state_prepare import ExternalStatePrepare

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        dim: int,
        info:dict,
        state_dataClass: ExternalStatePrepare,
        no_change_long:int,
        device = "cpu",
    ):
        self.dim = dim
        
        self.observation_space = spaces.Dict(
            {
                "knapsack": spaces.Box(low=np.array(
                    [[info['CAP_LOW'], info['CAP_LOW'], info['CAP_LOW'], 
                     info['CAP_LOW'], info['CAP_LOW'],0]]*state_dataClass.knapsackObsSize), 
                                       high=np.array(
                    [[info['CAP_HIGH'], info['CAP_HIGH'], info['CAP_HIGH'],
                     info['CAP_HIGH'], info['CAP_HIGH'],0]]*state_dataClass.knapsackObsSize), 
                                       shape=(state_dataClass.knapsackObsSize,
                                              self.dim), dtype=int),
                "instance_value": spaces.Box(low=np.array(
                    [[info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['VALUE_LOW']]]*state_dataClass.instanceObsSize), 
                                       high=np.array(
                    [[info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], 
                     info['VALUE_HIGH']]]*state_dataClass.instanceObsSize), 
                                       shape=(state_dataClass.instanceObsSize,
                                              self.dim), dtype=int),
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
        #self.len_observation = self.observation_space["knapsack"].shape[0]+self.observation_space["instance_value"].shape[0]+3
        self.device = device
        self.statePrepare = state_dataClass
        #self.statePrepare.normalizeData(info['CAP_HIGH'], info['VALUE_HIGH'])
        self.no_change_long = no_change_long
        self.no_change = 0
    
    def _get_obs(self) -> dict:
        stateCaps, stateWeightValues = self.statePrepare.getObservation()
        
        return {"knapsack": stateCaps, "instance_value": stateWeightValues}
        
    def _get_info(self):
        return ""
        
        
    def reset (self): 
        #super().reset(seed=seed)
        self.statePrepare.reset()
        externalObservation = self._get_obs()
        ACT = np.zeros((1,self.dim))
        externalObservation = np.append(ACT, np.append(np.append(externalObservation[
            "instance_value"], ACT, axis=0), np.append(externalObservation["knapsack"], 
                                                       ACT, axis=0),axis=0),
                                                       axis=0)
        info = self._get_info()
        '''if externalObservation.shape[0] < self.len_observation:
            externalObservation = self._pad_left(externalObservation, self.len_observation,
                                                 ACT)'''
        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device).unsqueeze(dim=0)        
        return externalObservation, info
    
    def step (self, step_actions):
        if len(step_actions) == 0: self.no_change += 1
        else: self.statePrepare.changeNextState(step_actions) 
        terminated = self.statePrepare.is_terminated() or self.no_change == self.no_change_long 
        
        if terminated:
            externalReward = self.reward_function()
        else: externalReward = 0
        
        externalObservation = self._get_obs()
        ACT = np.zeros((1,self.dim))
        externalObservation = np.append(ACT, np.append(np.append(externalObservation[
            "instance_value"], ACT, axis=0), np.append(externalObservation["knapsack"], 
                                                       ACT, axis=0),axis=0),
                                                       axis=0)
        '''if externalObservation.shape[0] < self.len_observation:
            externalObservation = self._pad_left(externalObservation, self.len_observation,
                                                 ACT)'''
        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device).unsqueeze(dim=0)
        info = self._get_info()
        
        return externalObservation, externalReward, terminated, info
    
    
    def response_decode (self, responce):
        pass
    
    def reward_function (self):
        external_reward = 0
        for ks in self.statePrepare.knapsacks:
            external_reward += ks.score_ratio()
        return external_reward
    
    
    
    
    