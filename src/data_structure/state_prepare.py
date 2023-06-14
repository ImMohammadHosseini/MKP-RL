
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
        allCapacities: Optional[np.ndarray] = None, 
        weights: Optional[np.ndarray] = None, 
        values: Optional[np.ndarray] = None, 
        k_obs_size: Optional[int] = None, 
        i_obs_size: Optional[int] = None
    ) -> None:
        if not weights is None and not values is None and not allCapacities is None:
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
    
    def set_new_problem (
        self, 
        allCapacities: np.ndarray, 
        weights: np.ndarray, 
        values: np.ndarray,
    ):
        assert len(weights) == len(values)
        self.weights = weights
        self.values = values
        self._setKnapsack(allCapacities)
    
    def normalizeData (self, maxCap, maxValue):
        self.weights = self.weights / maxCap
        self.values = self.values / maxValue
        for k in self.knapsacks: k.capacities = k.getCap() / maxCap
        
    def reset (self) -> None:
        #print('reset')
        shuffle = np.random.permutation(len(self.weights))
        self.remainInstanceWeights = self.weights[shuffle]
        self.remainInstanceValues = self.values[shuffle]
        shuffle = np.random.permutation(len(self.knapsacks))
        for k in self.knapsacks: k.reset()
        self.knapsacks = list(np.array(self.knapsacks)[shuffle])

    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
    
    def getObservation (self) -> np.ndarray:
        SOD = np.array([[1.]*(self.weights.shape[1]+1)])
        #EOD = np.array([[2.]*self.dim])
        #PAD = np.array([[0.]*self.dim])
        self.stateCaps = np.array([k.getRemainCap() \
                                   for k in self.knapsacks])
        self.stateCaps = np.append(self.stateCaps, np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        self.pad_len = self.instanceObsSize - len(self.remainInstanceWeights)
        if self.pad_len <= 0:
            self.pad_len = 0
            self.stateWeightValues = np.append(SOD, np.append(self.remainInstanceWeights[
                :self.instanceObsSize], self.remainInstanceValues[
                    :self.instanceObsSize], axis=1), axis=0)
            self.remainInstanceWeights = self.remainInstanceWeights[self.instanceObsSize:]
            self.remainInstanceValues = self.remainInstanceValues[self.instanceObsSize:]
        else: 
            padding = np.zeros((1, len(self.remainInstanceWeights[0])+1))
            self.stateWeightValues = self._pad_left(np.append(SOD, np.append(
                self.remainInstanceWeights, self.remainInstanceValues, axis=1), axis=0),
                padding)
            '''np.append(padding, 
                                               np.append(self.remainInstanceWeights, 
                                               self.remainInstanceValues, axis=1),
                                               axis=0)'''
            self.remainInstanceWeights = np.zeros((0, self.weights.shape[1]))
            self.remainInstanceValues = np.zeros((0,1))
        
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
    
    def getObservedKS (self, index):
        return self.stateCaps[index]
    
    def getObservedInst (self, index):
        return self.stateWeightValues[index]
    
    def getObservedInstWeight (self, index):
        return self.stateWeightValues[index][:-1]
    
    def getObservedInstValue (self, index):
        return self.stateWeightValues[index][-1]
    
    def is_terminated (self):
        return True if len(self.remainInstanceWeights) == 0 else False
    
    def changeNextState (self, acts):
        deleteList = []
        external_reward = []
        for inst_act, ks_act in acts:
            knapSack = self.getKnapsack(ks_act)
            cap = knapSack.getRemainCap()
            weight = self.getObservedInstWeight(inst_act)
            value = self.getObservedInstValue(inst_act)
            if not all(cap >= weight):
                print(knapSack.getExpectedCap())
                np.set_printoptions(precision=10)
                print(cap)
                np.set_printoptions(precision=10)
                print(weight)
            assert all(cap >= weight)
            if not inst_act >= self.pad_len:
                print('pad_len', self.pad_len)
                print('inst_act', inst_act)
            assert inst_act >= self.pad_len
            knapSack.addInstance(weight, value)
            external_reward.append(value / np.sum(weight))
            deleteList.append(inst_act)
        self.stateWeightValues = np.delete(self.stateWeightValues, 
                                           deleteList, axis=0)
        self.stateWeightValues = self.stateWeightValues[self.pad_len:]
        
        self.remainInstanceWeights = np.append(self.remainInstanceWeights,
                                               self.stateWeightValues[:,:-1],
                                               axis=0)
        self.remainInstanceValues = np.append(self.remainInstanceValues,
                                              np.expand_dims(self.stateWeightValues[:,-1], axis=1),
                                              axis=0)
        for k in self.knapsacks: k.resetExpectedCap()
        
        return external_reward