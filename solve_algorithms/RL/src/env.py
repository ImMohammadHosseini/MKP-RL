
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
        #state_dataClass: ExternalStatePrepare,
        no_change_long:int,
        knapsackObsSize:int,
        instanceObsSize:int,
        batchSize:int,
        device = "cpu",
    ):
        self.dim = dim
        
        self.observation_space = spaces.Dict(
            {
                "knapsack": spaces.Box(low=np.array(
                    [[[info['CAP_LOW'], info['CAP_LOW'], info['CAP_LOW'], 
                     info['CAP_LOW'], info['CAP_LOW'],0]]*knapsackObsSize]*batchSize), 
                                       high=np.array(
                    [[[info['CAP_HIGH'], info['CAP_HIGH'], info['CAP_HIGH'],
                     info['CAP_HIGH'], info['CAP_HIGH'],0]]*knapsackObsSize]*batchSize), 
                                       shape=(batchSize, knapsackObsSize,
                                              self.dim), dtype=int),
                "instance_value": spaces.Box(low=np.array(
                    [[[info['WEIGHT_LOW'], info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['WEIGHT_LOW'], info['WEIGHT_LOW'], 
                     info['VALUE_LOW']]]*instanceObsSize]*batchSize), 
                                       high=np.array(
                    [[[info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], info['WEIGHT_HIGH'],
                     info['WEIGHT_HIGH'], info['WEIGHT_HIGH'], 
                     info['VALUE_HIGH']]]*instanceObsSize]*batchSize), 
                                       shape=(batchSize, instanceObsSize,
                                              self.dim), dtype=int),
            }
        )
        self.action_space = spaces.Box(low=np.array([[0,0]] * \
                                                    knapsackObsSize), 
                                       high=np.array([[knapsackObsSize, 
                                                       instanceObsSize]] * \
                                                     knapsackObsSize), 
                                       dtype=int
        )
        
        super().__init__()
        self.device = device
        self.batchSize = batchSize
        self.info = info
        self.knapsackObsSize = knapsackObsSize
        self.instanceObsSize = instanceObsSize
        #self.set_statePrepare(state_dataClass)
        self.no_change_long = no_change_long
        
    
    def setStatePrepare (
        self,
        stateDataClasses: np.ndarray,
    ):
        self.statePrepares = stateDataClasses
        for statePrepare in self.statePrepares: 
            statePrepare.normalizeData(self.info['CAP_HIGH'], self.info['VALUE_HIGH'])

    def _get_obs (self) -> dict:
        batchCaps = np.zeros((0,self.knapsackObsSize,6)); 
        batchWeightValues = np.zeros((0,self.instanceObsSize+1,6))
        for statePrepare in self.statePrepares:
            stateCaps, stateWeightValues = statePrepare.getObservation()
            batchCaps = np.append(batchCaps, np.expand_dims(stateCaps, 0), 0) 
            batchWeightValues = np.append(batchWeightValues, 
                                          np.expand_dims(stateWeightValues, 0), 0)
        return {"knapsack": batchCaps, "instance_value":batchWeightValues}
        
    def _get_info (self):
        return ""
        
        
    def reset (self): 
        self.no_change = 0#TODO
        #super().reset(seed=seed)
        for statePrepare in self.statePrepares: statePrepare.reset()
        externalObservation = self._get_obs()
        #SOD = np.array([[[1.]*self.dim]]*self.batchSize)
        EOD = np.array([[[2.]*self.dim]]*self.batchSize)
        #PAD = np.array([[[0.]*self.dim]]*self.batchSize)
        
        externalObservation = np.append(np.append(externalObservation[
            "instance_value"],EOD, axis=1), np.append(externalObservation["knapsack"], 
                                         EOD, axis=1),axis=1)#,
                                        # axis=0)
        info = self._get_info()

        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device)#.unsqueeze(dim=0)
        return externalObservation, info
    
    def step (self, step_actions):
        if len(step_actions) == 0: self.no_change += 1; #print(self.no_change)
        externalReward = self.statePrepare.changeNextState(step_actions) 
        terminated = (self.statePrepare.is_terminated() or self.no_change > self.no_change_long)
        
        info = self._get_info()

        if terminated:
            return None, externalReward, terminated, info
        
        externalObservation = self._get_obs()
        #ACT = np.zeros((1,self.dim))
        SOD = np.array([[1.]*self.dim])
        EOD = np.array([[2.]*self.dim])
        PAD = np.array([[0.]*self.dim])
        externalObservation = np.append(np.append(externalObservation[
            "instance_value"],EOD, axis=0), np.append(externalObservation["knapsack"], 
                                         EOD, axis=0),axis=0)#,
                                         #axis=0)
        
        externalObservation = torch.tensor(externalObservation, 
                                           dtype=torch.float32, 
                                           device=self.device).unsqueeze(dim=0) 
        return externalObservation, externalReward, terminated, info
    
    
    def response_decode (self, responce):
        pass
    
    def final_score (self):
        score = 0; remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score += ks.score_ratio()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        remain_cap_ratio = np.mean(remain_cap_ratio)
        return score, remain_cap_ratio
    
    
    
    
    