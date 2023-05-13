
"""

"""
import numpy as np
from torch import tensor
from dataclasses import asdict, dataclass
from typing import Dict, List
from .knapsack import Knapsack

@dataclass
class ExternalStatePrepare:
    knapsacks: List[Knapsack]
    
    remainInstanceWeights: np.ndarray
    remainInstanceValues: np.ndarray
    
    pickedWeightsValues: List[int] = None
    
    def __init__ (self, allCapacities: np.ndarray, weights: np.ndarray, 
                  values: np.ndarray, k_obs_size: int, i_obs_size: int) -> None:
        self.weights = weights
        self.values = values
        
        self._setKnapsack(allCapacities)
        
        self.knapsackObsSize = k_obs_size
        self.instanceObsSize = i_obs_size
        
    def reset (self) -> None:
        self.remainInstanceWeights = self.weights
        self.remainInstanceValues = self.values
        for k in self.knapsacks: k.reset()
        
    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
     
    def getObservation (self) -> np.ndarray:
        self.stateCaps = np.array([k.getKnapsackRemainCap() \
                                   for k in self.knapsacks])
        self.stateCaps = np.append(self.stateCaps, np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        try:
            self.stateWeightValues = np.append(self.remainInstanceWeights[
                :self.instanceObsSize], self.remainInstanceValues[
                    :self.instanceObsSize], axis=1)
            self.remainInstanceWeights = self.remainInstanceWeights[self.instanceObsSize:]
            self.remainInstanceValues = self.remainInstanceValues[self.instanceObsSize:]
        except: 
            pad_len = self.instanceObsSize - len(self.remainInstanceWeights)
            padding = np.zeros((pad_len, len(self.remainInstanceWeights[0])+1))
            self.stateWeightValues = np.append(padding, 
                                               np.append(self.remainInstanceWeights, 
                                               self.remainInstanceValues, axis=1),
                                               axis=0)
            self.remainInstanceWeights = None
            self.remainInstanceValues = None
        
        
        return self.stateCaps, self.stateWeightValues
    
    def pad_left(self, sequence, final_length, padding_token):
        return [padding_token] * (final_length - len(sequence)) + sequence
    
    def is_terminated (self):
        return True if self.remainInstanceWeights == None or \
            len(self.remainInstanceWeights) == 0 else False
    
    def changeNextState (self):
        pass
    