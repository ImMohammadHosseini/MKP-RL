
"""

"""
import numpy as np
from dataclasses import dataclass
from typing import List
from .knapsack import Knapsack
from typing import List, Optional

@dataclass
class ExternalStatePrepare:
    knapsacks: List[Knapsack]
    
    remainInstanceWeights: np.ndarray
    remainInstanceValues: np.ndarray
    
    pickedWeightsValues: List[int] = None
    
    def __init__ (
        self, 
        allCapacities: np.ndarray, 
        weights: np.ndarray, 
        values: np.ndarray, 
        k_obs_size: Optional[int] = None, 
        i_obs_size: Optional[int] = None
    ) -> None:
        assert len(weights) == len(values)
        self.weights = weights
        self.values = values
        self._setKnapsack(allCapacities)
        
        if k_obs_size == None: 
            self.knapsackObsSize = len(allCapacities)
        else: self.knapsackObsSize = k_obs_size
        
        if i_obs_size == None:
            self.instanceObsSize = len(self.weights)
        else: self.instanceObsSize = i_obs_size
        self.pad_len = 0
    
    def normalizeData (self, maxCap, maxValue):
        self.weights = self.weights / maxCap
        self.values = self.values / maxValue
        for k in self.knapsacks: k.capacities = k.getCap() / maxCap
            
    def reset (self) -> None:
        shuffle = np.random.permutation(len(self.weights))
        self.remainInstanceWeights = self.weights[shuffle]
        self.remainInstanceValues = self.values[shuffle]
        for k in self.knapsacks: k.reset()
        
    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
    
    def getObservation (self) -> np.ndarray:
        self.stateCaps = np.array([k.getRemainCap() \
                                   for k in self.knapsacks])
        self.stateCaps = np.append(self.stateCaps, np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        self.pad_len = self.instanceObsSize - len(self.remainInstanceWeights)
        if self.pad_len <= 0:
            self.pad_len = 0
            self.stateWeightValues = np.append(self.remainInstanceWeights[
                :self.instanceObsSize], self.remainInstanceValues[
                    :self.instanceObsSize], axis=1)
            self.remainInstanceWeights = self.remainInstanceWeights[self.instanceObsSize:]
            self.remainInstanceValues = self.remainInstanceValues[self.instanceObsSize:]
        else: 
            padding = np.zeros((1, len(self.remainInstanceWeights[0])+1))
            self.stateWeightValues = self._pad_left(np.append(
                self.remainInstanceWeights, self.remainInstanceValues, axis=1),
                padding)
            '''np.append(padding, 
                                               np.append(self.remainInstanceWeights, 
                                               self.remainInstanceValues, axis=1),
                                               axis=0)'''
            self.remainInstanceWeights = np.zeros((0, self.weights.shape[1]))
            self.remainInstanceValues = np.zeros(0)
        
        return self.stateCaps, self.stateWeightValues

    def _pad_left(
        self,
        sequence: np.ndarray, 
        padding_token: np.ndarray,
    ):
        #self.statePrepare.pad_len = final_length - len(sequence)
        return np.append(np.repeat(padding_token, self.pad_len, axis=0), 
                         sequence, axis=0)
    
    def getKnapsack (self, index) -> Knapsack:
        return self.knapsacks[index]
    
    def getObservedInstWeight (self, index):
        return self.stateWeightValues[index][:-1]
    
    def getObservedInstValue (self, index):
        return self.stateWeightValues[index][-1]
    
    def is_terminated (self):
        return True if len(self.remainInstanceWeights) == 0 else False
    
    def changeNextState (self, acts):
        deleteList = []
        for inst_act, ks_act in acts:
            knapSack = self.getKnapsack(ks_act)
            cap = knapSack.getRemainCap()
            weight = self.getObservedInstWeight(inst_act)
            value = self.getObservedInstValue(inst_act)
            assert all(cap >= weight)
            assert inst_act >= self.pad_len
            knapSack.addInstance(weight, value)
            deleteList.append(inst_act)
        self.stateWeightValues = np.delete(self.stateWeightValues, 
                                           deleteList, axis=0)
        
        self.remainInstanceWeights = np.append(self.remainInstanceWeights,
                                               self.stateWeightValues[:,:-1],
                                               axis=0)
        self.remainInstanceValues = np.append(self.remainInstanceValues,
                                              np.expand_dims(self.stateWeightValues[:,-1], axis=1),
                                              axis=0)
        for k in self.knapsacks: k.resetExpectedCap()