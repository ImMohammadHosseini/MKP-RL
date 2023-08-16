
"""

"""
import numpy as np
from numpy import unravel_index
from dataclasses import dataclass
from typing import List
from .knapsack import Knapsack
from typing import List, Optional
from sklearn.metrics.pairwise import cosine_similarity

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
        ks_main_data: np.ndarray,
        instance_main_data: np.ndarray,
        k_obs_size: Optional[int] = None, 
        i_obs_size: Optional[int] = None
    ) -> None:
        assert len(weights) == len(values)        
        
        self.ks_main_data = ks_main_data
        self.instance_main_data = instance_main_data
        
        #inst_sim = cosine_similarity(instance_main_data[:len(weights)], weights, dense_output=False)
        #ks_sim = cosine_similarity(ks_main_data, allCapacities, dense_output=False)
        
        '''if not (np.diagonal(inst_sim) > .9).all():
            order = [0]*len(weights)
            for _ in range(len(inst_sim)):
                index = unravel_index(inst_sim.argmax(), inst_sim.shape)
                order[index[0]] = index[1]
                inst_sim[:,index[1]] = 0
                inst_sim[index[0],:] = 0
            self.weights = weights[order]
            self.values = values[order]
        else :'''
        self.weights = weights
        self.values = values
        
        '''if not (np.diagonal(ks_sim) > .9).all():
            order = [0]*len(allCapacities)
            for _ in range(len(ks_sim)):
                index = unravel_index(ks_sim.argmax(), ks_sim.shape)
                order[index[0]] = index[1]
                ks_sim[:,index[1]] = 0
                ks_sim[index[0],:] = 0
            self._setKnapsack(allCapacities[order])
        else :'''
        self._setKnapsack(allCapacities)

        if k_obs_size == None: 
            self.knapsackObsSize = len(allCapacities)
        else: self.knapsackObsSize = k_obs_size
        
        if i_obs_size == None:
            self.instanceObsSize = len(self.weights)
        else: self.instanceObsSize = i_obs_size
        self.pad_len = 0
    
    '''def normalizeData (self, maxCap, maxWeight, maxValue):
        self.weights = self.weights / maxWeight
        self.values = self.values / maxValue
        for k in self.knapsacks: k.capacities = k.getCap() / maxCap'''
        
    def reset (self) -> None:
        self.pad_len = 0
        self.ks_order = None
        #shuffle = np.random.permutation(len(self.weights))
        self.remainInstanceWeights = self.weights#[shuffle]
        self.remainInstanceValues = self.values#[shuffle]
        #shuffle = np.random.permutation(len(self.knapsacks))
        for k in self.knapsacks: k.reset()
        #self.knapsacks = list(np.array(self.knapsacks)[shuffle])
        
    def _setKnapsack (self, allCapacities):
        self.knapsacks = [Knapsack(i, c) for i, c in enumerate(allCapacities)]
    
    def getObservation1 (self) -> np.ndarray:
        self.stateCaps = np.array([k.getRemainCap() \
                                   for k in self.knapsacks])
        print(self.stateCaps.shape)
            
        '''ks_sim = cosine_similarity(self.ks_main_data, self.stateCaps, dense_output=False)
        if not (np.diagonal(ks_sim) > .9).all():
            self.ks_order = [0]*len(self.stateCaps)
            for _ in range(len(ks_sim)):
                index = unravel_index(ks_sim.argmax(), ks_sim.shape)
                self.ks_order[index[0]] = index[1]
                ks_sim[:,index[1]] = 0
                ks_sim[index[0],:] = 0
            self.stateCaps = self.stateCaps[self.ks_order]
        else :'''
        self.ks_order = None
        self.stateCaps = np.append(self.stateCaps, np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        '''inst_sim = cosine_similarity(self.instance_main_data[:len(self.remainInstanceWeights)], 
                                     self.remainInstanceWeights, dense_output=False)
        if not (np.diagonal(inst_sim) > .9).all():
            order = [0]*len(self.remainInstanceWeights)
            for _ in range(len(inst_sim)):
                index = unravel_index(inst_sim.argmax(), inst_sim.shape)
                order[index[0]] = index[1]
                inst_sim[:,index[1]] = 0
                inst_sim[index[0],:] = 0
            self.remainInstanceWeights = self.remainInstanceWeights[order]
            self.remainInstanceValues = self.remainInstanceValues[order]'''
        
        
        self.pad_len = self.instanceObsSize - len(self.remainInstanceWeights)
        
        #print(self.remainInstanceWeights)
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
            self.remainInstanceValues = np.zeros((0,1))
        
        return self.stateCaps, self.stateWeightValues

    def getObservation (self) -> np.ndarray:
        self.pad_len = self.instanceObsSize - len(self.remainInstanceWeights)
        sw = self.remainInstanceWeights[:self.instanceObsSize]
        stateWeight = sw
        stateValue = self.remainInstanceValues[:self.instanceObsSize]
        self.ks_order = None
        self.stateCaps = []    
        for k in self.knapsacks:
            self.stateCaps.append(k.getRemainCap())
            z = self.stateCaps[-1] == 0.0
            z = float(z) * 1e-8
            stateWeight = np.append(stateWeight, (sw/(self.stateCaps[-1]+z)),-1)
            
        self.stateCaps = np.append(np.array(self.stateCaps), np.zeros((len(self.stateCaps),1)), 
                                   axis=1)
        
        if self.pad_len <= 0:
            self.pad_len = 0
            self.stateWeightValues = np.append(sw, stateValue, axis=1)
            returnWeightValues = np.append(stateWeight, stateValue, axis=1)
            self.remainInstanceWeights = self.remainInstanceWeights[self.instanceObsSize:]
            self.remainInstanceValues = self.remainInstanceValues[self.instanceObsSize:]
        else: 
            self.stateWeightValues = self._pad_left(np.append(sw, stateValue, axis=1),
                np.zeros((1, len(sw[0])+1)) )
            returnWeightValues = self._pad_left(np.append(stateWeight, stateValue, axis=1),
                np.zeros((1, len(stateWeight[0])+1)) )
            self.remainInstanceWeights = np.zeros((0, self.weights.shape[1]))
            self.remainInstanceValues = np.zeros((0,1))
        
        return self.stateCaps, returnWeightValues
    
    def _pad_left(
        self,
        sequence: np.ndarray, 
        padding_token: np.ndarray,
    ):
        #self.statePrepare.pad_len = final_length - len(sequence)
        return np.append(np.repeat(padding_token, self.pad_len, axis=0), 
                         sequence, axis=0)
    
    def getRealKsAct (self, index):
        try:
            return self.ks_order[index]
        except:
            return index
        
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
        caps = np.array([k.getRemainCap() for k in self.knapsacks])
        terminated = ~np.array([(cap*(1/self.remainInstanceWeights)>=1).all(
            1).any() for cap in caps]).any()
        return True if len(self.remainInstanceWeights) == 0  or terminated \
            else False
    
    def changeNextState (self, acts):
        deleteList = []
        external_reward = []
        for inst_act, ks_act in acts:
            knapSack = self.getKnapsack(ks_act)
            cap = knapSack.getRemainCap()
            weight = self.getObservedInstWeight(inst_act)
            value = self.getObservedInstValue(inst_act)
            if not all(cap >= weight):
            #    print('kk',knapSack.getExpectedCap())
            #    np.set_printoptions(precision=10)
                print('cap', cap)
            #    np.set_printoptions(precision=10)
                print('w', weight)
            assert all(cap >= weight)
            #if not inst_act >= self.pad_len:
            #    print('pad_len', self.pad_len)
            #    print('inst_act', inst_act)
            assert inst_act >= self.pad_len
            knapSack.addInstance(weight, value)
            external_reward.append(value / np.sum(weight))
            deleteList.append(inst_act)
        #print(len(deleteList))
        #print('len dd', len(self.stateWeightValues))

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