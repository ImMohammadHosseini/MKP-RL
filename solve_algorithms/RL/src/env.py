
"""

"""
import torch
import numpy as np

import gymnasium as gym
from gymnasium import spaces
from src.data_structure.state_prepare import StatePrepare

class KnapsackAssignmentEnv (gym.Env):
    def __init__ (
        self,
        vector_dim: int,
        info: dict,
        no_change_long: int,
        knapsackObsSize: int,
        instanceObsSize: int,
        device: torch.device,
    ):
        self.vector_dim = vector_dim
        '''self.observation_space = spaces.Dict(
            {
                "knapsack": spaces.Box(low=np.array(
                    [[[info['CAP_LOW'], info['CAP_LOW'], info['CAP_LOW'], 
                     info['CAP_LOW'], info['CAP_LOW'],0]]*knapsackObsSize]*main_batch_size), 
                                       high=np.array(
                    [[[info['CAP_HIGH'], info['CAP_HIGH'], info['CAP_HIGH'],
                     info['CAP_HIGH'], info['CAP_HIGH'],0]]*knapsackObsSize]*main_batch_size), 
                                       shape=(main_batch_size, knapsackObsSize,
                                              self.dim), dtype=int),
                "instance_value": spaces.Box(low=np.array(
                    [[[info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['VALUE_LOW']]]*instanceObsSize]*main_batch_size), 
                                       high=np.array(
                    [[[info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], 
                     info['VALUE_HIGH']]]*instanceObsSize]*main_batch_size), 
                                       shape=(main_batch_size, instanceObsSize,
                                              self.dim), dtype=int),
            }
        )
        self.action_space = spaces.Box(low=np.array([[0,0]] * \
                                                    knapsackObsSize), 
                                       high=np.array([[knapsackObsSize, 
                                                       instanceObsSize]] * \
                                                     knapsackObsSize), 
                                       dtype=int
        )'''
        
        super().__init__()
        self.device = device
        self.info = info
        self.knapsackObsSize = knapsackObsSize
        self.instanceObsSize = instanceObsSize
        self.no_change_long = no_change_long
        
    def setStatePrepare (
        self,
        stateDataClass: np.ndarray,
    ):
        self.statePrepare = stateDataClass
        
    def _get_obs (self) -> dict:
        '''batchCaps = np.zeros((0,30,4)); 
        batchWeightValues = np.zeros((0,30,4))
        for statePrepare in self.statePrepares:
            stateCaps, stateWeightValues = statePrepare.getObservation()
            batchCaps = np.append(batchCaps, np.expand_dims(stateCaps, 0), 0) 
            batchWeightValues = np.append(batchWeightValues, 
                                          np.expand_dims(stateWeightValues, 0), 0)'''
        stateCaps, stateWeightValues = self.statePrepare.getObservation1()
        #print(stateCaps.shape)
        #print(stateWeightValues.shape)
        batchCaps = np.expand_dims(stateCaps, 0)
        batchWeightValues = np.expand_dims(stateWeightValues, 0)

        #print(self.dd)
        return {"knapsack": batchCaps, "instance_value":batchWeightValues}
        
    def _get_info (self):
        return ""
        
        
    def reset (self, cahnge_problem=True): 
        self.no_change = 0
        self.statePrepare.reset(cahnge_problem)
        externalObservation = self._get_obs()
        SOD = np.array([[[1.]*self.vector_dim]])
        EOD = np.array([[[2.]*self.vector_dim]])
        
        
        sod_instance_value = np.insert(externalObservation[
            "instance_value"], self.statePrepare.pad_len, SOD, axis=1)
        
        externalObservation = np.append(np.append(sod_instance_value, EOD, axis=1), 
                                            np.append(externalObservation["knapsack"], 
                                             EOD, axis=1),axis=1)
        #print(self.dd)
        info = self._get_info()

        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device)
        
        return externalObservation, info
    
    def step (self, step_actions):
        terminated = False
        invalid_action_end_index = max(np.where(step_actions == -1)[0], default=-1)
        if invalid_action_end_index == step_actions.shape[1]-1: self.no_change += 1
        else: self.no_change=0
        self.statePrepare.changeNextState(
            step_actions[invalid_action_end_index+1:]) #exteraRewards=
        #exteraRewards = torch.tensor(exteraRewards)             
        terminated = terminated or self.statePrepare.is_terminated()
        
        
        info = self._get_info()

        
        externalObservation = self._get_obs()
        SOD = np.array([[[1.]*self.vector_dim]])
        EOD = np.array([[[2.]*self.vector_dim]])

        shape = externalObservation["instance_value"].shape
        sod_instance_value = np.zeros((shape[0], shape[1]+1, shape[2]))
        sod_instance_value = np.insert(externalObservation[
            "instance_value"], self.statePrepare.pad_len, SOD, axis=1)
           
        externalObservation = np.append(np.append(sod_instance_value, EOD, axis=1), 
                                            np.append(externalObservation["knapsack"], 
                                             EOD, axis=1),axis=1)
        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device)
        
        return externalObservation, 0, terminated, info
    
    
    def response_decode (self, responce):
        pass
    
    def final_score (self):
        score = []
        remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score.append(ks.getValues())
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        
        return np.sum(score, 0), np.mean(remain_cap_ratio)
    
    
    
    
    