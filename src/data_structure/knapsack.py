
"""

"""

from dataclasses import asdict, dataclass
from typing import Dict, List

@dataclass
class Knapsack:
    knapsackId: int
    capacities: List[int]
    instanceIds: List[List[int]]
    
    def getKnapsackRemainCap(self) -> List[int]:
        pass
    
    def addInstance(self) -> None:
        pass
    
    def getKnapsackCap (self):
        pass
    
    def getValues (self) -> int:
        pass