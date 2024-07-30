
"""

"""
import numpy as np
from dataclasses import dataclass

@dataclass
class Knapsack:
    knapsackId: int
    capacities: np.ndarray
    instanceWeights: np.ndarray
    instanceValues: np.ndarray
    
    def __init__(self, 
                 kid: int, 
                 capacities: np.ndarray,
                 dim: int,
                 obj:int
    ):
        
        self.knapsackId = kid
        self.capacities = capacities
        self.dim = dim
        self.obj = obj
    
    def reset (self) -> None:
        self.instanceWeights = np.zeros((0, self.dim))
        self.instanceValues = np.zeros((0, self.obj))
        self.resetExpectedCap()
        
    def getRemainCap(self) -> np.ndarray:
        fullPart = np.sum(self.instanceWeights, axis=0)
        return self.capacities - fullPart
    
    def getExpectedCap (self):
        return self.expectedCap
    
    def removeExpectedCap (self, newCap: np.ndarray):
        self.expectedCap -= newCap
        
    def resetExpectedCap (self):
        self.expectedCap = self.getRemainCap()
        
    def addInstance(self, instWeight, instValue) -> None:
        self.instanceValues = np.append(self.instanceValues, 
                                        np.expand_dims(instValue, axis=0), 
                                        axis=0)
        self.instanceWeights = np.append(self.instanceWeights, 
                                         np.expand_dims(instWeight, axis=0), 
                                         axis=0)
        

    '''def score_ratio (self):
        ratio = 0
        for weight, value in zip(self.instanceWeights, self.instanceValues):
            ratio += value / np.sum(weight)
        return ratio'''
    
    def getValues (self) -> int:
        return np.sum(self.instanceValues, 0)
    
    def getCap (self):
        return self.capacities