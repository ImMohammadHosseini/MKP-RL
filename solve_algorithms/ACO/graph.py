
import numpy as np

class Node:
    
    def __init__(
        self,
        node_id: int,
        inst_id: int,
        ks_id: int,
        weight: np.ndarray, 
        value: np.ndarray,
        cap: np.ndarray,
    ):
        self.id = node_id
        self.inst_id = inst_id
        self.ks_id = ks_id
        self.weight = weight
        self.value = value
        #self.cap = cap
    
class Graph():
    
    def __init__ (
        self, 
        weights, values, caps, alpha=1, beta=3, d=.2, min_ph=.5, profit=.1,
                  b_profit=.5):
        self.make_nodes(weights, values, caps)#TODO
        self.alpha = alpha
        self.beta = beta
        self.d = d
        self.min_ph = min_ph
        self.profit = profit
        self.b_profit = b_profit
        self.caps = caps
        self.values = values
        self.weights = weights
        
        self.graph_weights = {}
        self.prof ={}#####################
        self.pheromone = {}
        
        for node_i in self.nodes:
            for node_j in self.nodes:
                if node_i.id == node_j.id:
                    self.weights[(node_i.id,node_j.id)] = 0
                    self.prof[(node_i.id,node_j.id)] = 0
                    self.phs[(node_i.id,node_j.id)] = 0
                else:
                    self.weights[(node_i.id,node_j.id)] = self.nodes[j].weight
                    self.prof[(node_i.id,node_j.id)] = self.nodes[j].value
                    self.phs[(node_i.id,node_j.id)] = min_ph
                    
    def make_nodes (self, weights, values, caps):
        self.nodes = []
        n_id = 0
        for inst_id, wv in zip(weights, values):
            for ks_id, c in enumerate(caps):
                self.nodes.append(Node(n_id, inst_id, ks_id, wv[0], wv[1], c))
                n_id += 1
                
    
    def 
    