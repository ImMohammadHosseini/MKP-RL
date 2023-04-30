
"""

"""
import numpy as np

def getSituations (caps:np.ndarray, weights:np.ndarray, dim:int):
    """
    Data preparation for situations. Since the number of instances can be 
    much more than knapsacks, we use 2*knapsack numbers(N) for instances in every
    situation and in every situation, all 2N instance knapsack comes first then
    ACT(token of zeros) token, after that all knapsacks and ACT token comes
    """
    
    w = np.append(weights, np.zeros((len(weights)%(2*len(caps)),5)), axis=0)
    length = len(w)//(2*len(caps))
    w = w.reshape((length,2*len(caps)*dim))
    c = np.array([caps.reshape((len(caps)*dim,)),]*length)
    ACT = np.zeros((length,dim))
    situations = np.append(np.append(w, ACT, axis=1), 
                           np.append(c, ACT, axis=1),
                           axis=1)
    return situations
    
def getPrompt (situations:np.ndarray):
    pass