
"""

"""
import numpy as np
from dataclasses import asdict, dataclass
from typing import Dict, List

@dataclass
class Knapsack:
    knapsackId: int
    capacities: np.ndarray
    instanceWeight: np.ndarray
    instanceValue: np.ndarray
    
    def __init__(self, kid: int, capacities: np.ndarray):
        self.knapsackId = kid
        self.capacities = capacities
    
    def reset (self) -> None:
        self.instanceWeight = np.array([])
        self.instanceValue = np.array([])
        self.resetExpectedCap()
        
    def getKnapsackRemainCap(self) -> np.ndarray:
        fullPart = np.sum(self.instanceWeight, axis=0)
        return self.capacities - fullPart
    
    def getExpectedCap (self):
        return self.expectedCap
    
    def addExpectedCap (self, newCap: np.ndarray):
        self.expectedCap =- newCap
        
    def resetExpectedCap (self):
        self.expectedCap = self.capacities
        
    def addInstance(self) -> None:
        pass
    
    
    def getKnapsackCap (self):
        return self.capacities
    
    def getValues (self) -> int:
        return np.sum(self.instanceValue)