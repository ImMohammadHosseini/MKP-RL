
"""

"""
import numpy as np



def multiObjectiveDimentionalKP (
        problemNums: int,
        dim:int,
        obj:int,
        instNums:int,
        ksNums:int,
        info):
    """
    multi-Objective multi-Dimentional multipleKnapSack, 
    multi capacities for each knapsack, 
    multi weight and multi value for each instance
    set a random dataset for ksNums knapsack and instNums instance
    produce knapsack capacity array in size of (ksNums,dim), 
    instance weight array in size of (instNums,dim)  and 
    instance value array in size of (instNums,obj)
    """
    capacities = list(np.random.randint(low=info['CAP_LOW'], high=info['CAP_HIGH'], 
                                   size=(problemNums,ksNums,dim)))
    weights = list(np.random.randint(low=info['WEIGHT_LOW'], high=info['WEIGHT_HIGH'], 
                                size=(problemNums,instNums,dim)))
    values = list(np.random.randint(low=info['VALUE_LOW'], high=info['VALUE_HIGH'], 
                               size=(problemNums,instNums,obj)))
    return capacities, weights, values