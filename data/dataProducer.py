
"""

"""
import numpy as np


def multipleKnapSackData (M:int, N:int):
    """
    multipleKnapSack, one capacities for each knapsack, one weight for each instance
    set a random dataset for M knapsack and N instance
    produce knapsack capacity array in size of M, instance weight array 
    in size of N  and instance value array in size of N
    """
    capacities = np.random.randint(low=5, high=150, size=M)
    weights = np.random.randint(low=2, high=100, size=N)
    values = np.random.randint(low=1, high=200, size=N)
    return capacities, weights, values

def multiDimentional (M:int, N:int):
    pass

def multiObjectiveDimentional (d:int, M:int, N:int):
    """
    multi-Objective multi-Dimentional multipleKnapSack, 
    multi capacities for each knapsack, 
    multi weight and one value for each instance
    set a random dataset for M knapsack and N instance
    produce knapsack capacity array in size of (d,M), instance weight array 
    in size of (d,N)  and instance value array in size of N
    """
    capacities = np.random.randint(low=5, high=200, size=(d,M))
    weights = np.random.randint(low=2, high=100, size=(d,N))
    values = np.random.randint(low=1, high=200, size=N)
    return capacities, weights, values