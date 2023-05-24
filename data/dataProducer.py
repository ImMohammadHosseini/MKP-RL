
"""

"""
import numpy as np


def multipleKnapSackData (M:int, N:int, info):
    """
    multipleKnapSack, one capacities for each knapsack, one weight for each instance
    set a random dataset for M knapsack and N instance
    produce knapsack capacity array in size of M, instance weight array 
    in size of N  and instance value array in size of N
    """
    capacities = np.random.randint(low=info['CAP_LOW'], high=info['CAP_HIGH'],
                                   size=M)
    weights = np.random.randint(low=info['WEIGHT_LOW'], high=info['WEIGHT_HIGH'],
                                size=N)
    values = np.random.randint(low=info['VALUE_LOW'], high=info['VALUE_HIGH'],
                               size=N)
    return capacities, weights, values

def multiDimentional (M:int, N:int):
    pass

def multiObjectiveDimentional (D:int, M:int, N:int, info):
    """
    multi-Objective multi-Dimentional multipleKnapSack, 
    multi capacities for each knapsack, 
    multi weight and one value for each instance
    set a random dataset for M knapsack and N instance
    produce knapsack capacity array in size of (d,M), instance weight array 
    in size of (d,N)  and instance value array in size of N
    """
    capacities = np.random.randint(low=info['CAP_LOW'], high=info['CAP_HIGH'], 
                                   size=(M,D))
    weights = np.random.randint(low=info['WEIGHT_LOW'], high=info['WEIGHT_HIGH'], 
                                size=(N,D))
    values = np.random.randint(low=info['VALUE_LOW'], high=info['VALUE_HIGH'], 
                               size=(N,1))
    return capacities, weights, values