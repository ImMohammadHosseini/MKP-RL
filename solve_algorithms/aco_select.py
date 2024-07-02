


import numpy as np
from src.data_structure.state_prepare import StatePrepare
from typing import Optional
from ACO.graph import Graph

class ACOSelect():
    def __init__ (
        self,
        dim: int,
        state_dataClass: StatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.dim = dim
        #self.no_change_long = 3
        #self.reset()
        
    def _choose_actions (
        self,
        iter_num: int,
    ):
        caps,weightValues = self.statePrepare.getObservation1()
        values = weightValues[:, self.dim:]
        weights = weightValues[:, :self.dim]
        print(caps.shape)
        print(values.shape)
        print(weights.shape)
        
        graph = Graph(weights, values, caps)
        
        #accepted_actions = np.zeros((0,2), dtype= int)
        ants = All_ants(m, self.W)
        for i in range(iter_num):
            ants.do_cycles(self)
            accepted_actions = ants.best
            self.update_phs(ants)
    
        return accepted_actions, steps

    def test_step (
        self,
        iter_num: int=100
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
    