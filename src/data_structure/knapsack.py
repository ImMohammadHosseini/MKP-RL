
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
    
    def __init__(self, kid: int, capacities: np.ndarray):
        self.knapsackId = kid
        self.capacities = capacities
    
    def reset (self) -> None:
        self.instanceWeights = np.array([])
        self.instanceValues = np.array([])
        self.resetExpectedCap()
        
    def getRemainCap(self) -> np.ndarray:
        fullPart = np.sum(self.instanceWeights, axis=0)
        return self.capacities - fullPart
    
    def getExpectedCap (self):
        return self.expectedCap
    
    def addExpectedCap (self, newCap: np.ndarray):
        self.expectedCap =- newCap
        
    def resetExpectedCap (self):
        self.expectedCap = self.getRemainCap()
        
    def addInstance(self, instWeight, instValue) -> None:
        self.instanceValues = np.append(self.instanceValues, instValue, axis=0)
        self.instanceWeights = np.append(self.instanceWeights, instWeight, axis=0)
        
    def getCap (self):
        return self.capacities
    
    def getValues (self) -> int:
        return np.sum(self.instanceValue)