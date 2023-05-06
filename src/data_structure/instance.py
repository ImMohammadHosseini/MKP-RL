
"""

"""

from dataclasses import asdict, dataclass
from typing import Dict, List

@dataclass
class Instance:
    instanceId: int
    weights: List[int]
    value: int
    
    picked: bool 
    
    def __init__(self, iid: int, weights: List, value: int):
        self.instanceId = iid
        self.weights = weights
        self.value = value
        self.picked = False
        
    def

    