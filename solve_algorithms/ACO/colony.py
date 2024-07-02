
import numpy as np
from random import randint, random


class Ant:
    def __init__(self, graph, start):
        self.graph = graph
        self.ks_num = graph.caps.shape[0]
        self.path = [start]
        self.path_weight = np.zeros((self.ks_num, graph.weight.shpae[1]), dtype=float)
        print(self.path_weight.shape)
        print(self.graph.nodes[start].weight)
        self.path_weight[self.graph.nodes[start].ks_id] += self.graph.nodes[start].weight
        self.path_value = self.graph.nodes[start].values
      
    def get_path (self):
        length = len(self.p)
        return [tuple((self.p[i], self.p[(i + 1) % length]))
            for i in range(length)]
    
    def get_available(self):
        nodes = set(range(len(self.graph.nodes)))
        chosen_instance = set([int(p/self.ks_num)*self.ks_num+ks for p in self.path for ks in range(self.ks_num)])
        available = nodes - chosen_instance
        #twjs = self.twjs(graph)
        
        not_good = []    
        for _ in available:
            ks_id = self.graph.nodes[_].ks_id
            print('ggg')
            print(self.path_weight[ks_id] + self.graph.nodes[_].weight)
            if self.path_weight[ks_id] + self.graph.nodes[_].weight > self.graph.caps[ks_id]:
                not_good.append(_)
        available -= set(not_good) 
        return available
    
    def cycle(self):
        available = self.get_available(self.graph)
        
        counter = 0
        while available:
            counter += 1
            self.next_(self.graph, available)
            available = self.get_available(self.graph)
    
    def next_(self, available):
        if len(available) == 1:
            new_id = available.pop()
            self.path.append(new_id)
            self.path_weight[self.graph.nodes[new_id].ks_id] += self.graph.nodes[new_id].weight
            self.path_value += self.graph.nodes[new_id].values
            
        if not available:
            return

        total = 0
        probabilities = {}
        for node_id in available:
            probabilities[node_id] = self.graph.get_probability(
                    self.path[-1], node_id)
            total += probabilities[node_id]
           

        threshold = random()
        probability = 0
        for node_id in available:
            probability += probabilities[node_id] / total
            if threshold < probability:
                self.path.append(node_id)
                self.path_weight[self.graph.nodes[node_id].ks_id] += self.graph.nodes[node_id].weight
                self.path_value += self.graph.nodes[node_id].values

                return
        node_id =  available.pop()
        self.path.append(node_id)
        self.path_weight[self.graph.nodes[node_id].ks_id] += self.graph.nodes[node_id].weight
        self.path_value += self.graph.nodes[node_id].values

    
    
class Colony:
    
    def __init__(self, ant_counts):
        self.ants = []
        self.ant_counts = ant_counts
        
        self.best = None
        self.min_sum = float("inf")
        self.max_value = 0
        
        
    def reset_ants(self, graph):
        self.ants = []
        nodes = len(graph.nodes) - 1
        for _ in range(self.A_counts):
            self.ants.append(Ant(graph, randint(0, nodes)))

    def do_cycles(self, graph):
        
        self.reset_ants(graph=graph)
        for ant in self.ants:
            
            ant.cycle(graph=graph)
            graph.pheromones(answer = ant.get_path())
            if self.max_value < ant.path_value:#TODO sum of path value for multi_obj
                assert (ant.path_weight <= graph.caps).all()
                self.max_value = ant.path_value
                self.best = ant.path