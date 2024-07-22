


import numpy as np
from src.data_structure.state_prepare import StatePrepare
from typing import Optional
from .ACO.graph import Graph
from .ACO.colony import Colony

class ACOSelect():
    def __init__ (
        self,
        num_instance: int,
        state_dataClass: StatePrepare,
    ):
        self.statePrepare = state_dataClass
        #self.greedy_num = num_instance
        #self.statePrepare = state_dataClass
        #self.dim = dim
        #self.no_change_long = 3
        #self.reset()
        
    def _choose_actions (
        self,
        iter_num: int,
        ant_num: int = 50
    ):
        caps, weightValues = self.statePrepare.getObservation1()
        caps = caps[:, :len(self.statePrepare.getObservedInstValue(0)):]
        values = weightValues[:, -len(self.statePrepare.getObservedInstValue(0)):]
        weights = weightValues[:, :len(self.statePrepare.getObservedInstValue(0)):]
        
        graph = Graph(weights, values, caps)
        
        #accepted_actions = np.zeros((0,2), dtype= int)
        ants = Colony(ant_num)
        for i in range(iter_num):
            ants.do_cycles(graph)
            accepted_actions = ants.get_accepted_actions(graph)
            graph.update_pheromones()
        return accepted_actions, 0

    def test_step (
        self,
        iter_num: int=20
    ):
        accepted_actions, steps = self._choose_actions(iter_num)
        self.statePrepare.changeNextState(accepted_actions)
        
        score, remain_cap = self.final_score()
        return score, remain_cap, steps
    
    #def getObservation (self):
    #    return self.caps, self.weightValues
    
    #def reset(self):
    #    self.statePrepare.reset()

    def final_score (self):
        score = 0
        remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score += ks.getValues()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        return score, np.mean(remain_cap_ratio)
    
    def reset(self):
        self.statePrepare.reset()
    