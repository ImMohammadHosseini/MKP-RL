
import numpy as np
from random import randint
from copy import deepcopy


class Ant:
    def __init__(self, graph, start):
        self.graph = graph
        self.ks_num = graph.caps.shape[0]
        self.path = [start]
        self.path_weight = np.zeros((self.ks_num, graph.weights.shape[1]), dtype=float)
        if (self.graph.nodes[start].weight < self.graph.caps[self.graph.nodes[start].ks_id]).all():
            self.path_weight[self.graph.nodes[start].ks_id] += self.graph.nodes[start].weight
        self.path_value = deepcopy(self.graph.nodes[start].value)
        
      
    def get_path (self):
        length = len(self.path)
        return [tuple((self.path[i], self.path[(i + 1) % length]))
            for i in range(length)]
    
    def get_available(self):
        nodes = set(range(len(self.graph.nodes)))
        chosen_instance = set([int(p/self.ks_num)*self.ks_num+ks for p in self.path for ks in range(self.ks_num)])
        available = nodes - chosen_instance
        #twjs = self.twjs(graph)
        
        not_good = []    
        for _ in available:
            ks_id = self.graph.nodes[_].ks_id
            if ((self.path_weight[ks_id] + self.graph.nodes[_].weight) > self.graph.caps[ks_id]).any():
                not_good.append(_)
        available -= set(not_good) 
        return available
    
    def cycle(self):
        available = self.get_available()
        
        counter = 0
        while available:
            counter += 1
            self.next_(available)
            available = self.get_available()
    
    def next_(self, available):
        if len(available) == 1:
            new_id = available.pop()
            self.path.append(new_id)
            self.path_weight[self.graph.nodes[new_id].ks_id] += self.graph.nodes[new_id].weight
            self.path_value += self.graph.nodes[new_id].value
            
        if not available:
            return

        total = 0
        probabilities = {}
        for node_id in available:
            probabilities[node_id] = self.graph.get_probability(
                    self.path[-1], node_id)
            total += probabilities[node_id]

        threshold = np.random.rand(2)

        probability = 0
        for node_id in available:
            probability += probabilities[node_id] / total
            if (threshold < probability).all():
                self.path.append(node_id)
                self.path_weight[self.graph.nodes[node_id].ks_id] += self.graph.nodes[node_id].weight
                self.path_value += self.graph.nodes[node_id].value
                return
        node_id =  available.pop()
        self.path.append(node_id)
        self.path_weight[self.graph.nodes[node_id].ks_id] += self.graph.nodes[node_id].weight
        self.path_value += self.graph.nodes[node_id].value

    
    
class Colony:
    
    def __init__(self, ant_counts):
        self.ants = []
        self.ant_counts = ant_counts
        
        self.best = None
        self.min_sum = float("inf")
        self.max_value = np.array(0)
        
        
    def reset_ants(self, graph):
        self.ants = []
        nodes = len(graph.nodes) - 1
        for _ in range(self.ant_counts):
            self.ants.append(Ant(graph, randint(0, nodes)))
    
    def get_accepted_actions (self, graph):
        accepted_actions = np.zeros((0,2),dtype=np.int32)
        for nid in self.best:
            accepted_actions = np.append(accepted_actions,
                                         [[graph.nodes[nid].inst_id, 
                                           graph.nodes[nid].ks_id]], 0)
        return accepted_actions
            
    def do_cycles(self, graph):
        if (self.max_value == 0).all():
            self.max_value = np.zeros((graph.values.shape[1]), dtype=float)
        self.reset_ants(graph=graph)
        for ant in self.ants:
            
            ant.cycle()
            graph.pheromones(answer = ant.get_path())
            if (self.max_value < ant.path_value).all():
                assert (ant.path_weight <= graph.caps).all()
                self.max_value = ant.path_value
                self.best = ant.path
            