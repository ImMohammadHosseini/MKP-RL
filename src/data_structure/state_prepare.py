
"""

"""
from torch import tensor
from dataclasses import asdict, dataclass
from typing import Dict, List
from .knapsack import Knapsack

@dataclass
class StatePrepare:
    knapsacks: List[Knapsack]
    
    remainInstanceWeights: List[int]
    remainInstanceValues: List[int]
    
    pickedWeightsValues: List[int] = None
    
    def __init__ (self, iid: int, weights: List, values: int):
        self.remainInstanceWeights = weights
        self.remainInstanceValues = values

    def getState (self) -> tensor:
        pass
    
    