
"""

"""
import numpy as np
from torch import tensor
from dataclasses import asdict, dataclass
from typing import Dict, List
from .knapsack import Knapsack

@dataclass
class StatePrepare:
    knapsacks: List[Knapsack]
    
    remainInstanceWeights: np.ndarray
    remainInstanceValues: np.ndarray
    
    pickedWeightsValues: List[int] = None
    
    def __init__ (self, allCapacities: np.ndarray, weights: np.ndarray, 
                  values: np.ndarray, k_obs_size: int, i_obs_size: int):
        self.remainInstanceWeights = weights
        self.remainInstanceValues = values
        self._setKnapsack(allCapacities)
        
        self.knapsackObsSize = k_obs_size
        self.instanceObsSize = i_obs_size
        
    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
     
    def getObservation (self) -> np.ndarray:
        self.stateCaps = np.array([c.getKnapsackRemainCap() \
                                   for c in self.knapsacks])
        self.stateCaps = np.append(self.stateCaps, np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        try:
            self.stateWeightValues = np.append(self.remainInstanceWeights[
                :self.instanceObsSize], self.remainInstanceValues[
                    :self.instanceObsSize], axis=1)
            self.remainInstanceWeights = self.remainInstanceWeights[self.instanceObsSize:]
            self.remainInstanceValues = self.remainInstanceValues[self.instanceObsSize:]
        except: 
            pad_len = self.instanceObsSize - self.remainInstanceWeights
            padding = np.zeros((pad_len, len(self.remainInstanceWeights[0]+1)))
            self.stateWeightValues = np.append(np.append(self.remainInstanceWeights, 
                                               self.remainInstanceValues, axis=1),
                                               padding, axis=0)
            self.remainInstanceWeights = None
            self.remainInstanceValues = None
            
        return self.stateCaps, self.stateWeightValues
               
    def changeNextState (self):
        pass
    