
"""

"""

import numpy as np
from src.data_structure.state_prepare import ExternalStatePrepare


class GreedySelect():
    def __init__ (
        self,
        state_dataClass: ExternalStatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.greedy_num = 3000
        self.no_change_long = 1e+3
    
    def _choose_actions_greedy_firstFit (
        self,
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        caps, weightValues = self.statePrepare.getObservation()
        greedy_inst_indx = np.flip(np.argsort(weightValues[:, -1]))[:self.greedy_num]
        for inst_idx in greedy_inst_indx:
            if inst_idx >= self.statePrepare.pad_len:
                weight = self.statePrepare.getObservedInstWeight(inst_idx)
                for ks_idx in range(caps.shape[0]):
                    knapSack = self.statePrepare.getKnapsack(ks_idx)
                    eCap = knapSack.getExpectedCap()
                    if all(eCap >= weight):
                        knapSack.removeExpectedCap(weight)
                        accepted_actions = np.append(accepted_actions, 
                                                     [[inst_idx, ks_idx]], 0)
                        break
        return accepted_actions
    
    def test_step (
        self,
    ):
        self.statePrepare.reset()
        step_acts = np.zeros((0,2), dtype= int)
        no_change = 0
        while no_change < self.no_change_long:
            accepted_actions = self._choose_actions_greedy_firstFit()
            if len(accepted_actions) == 0:
               no_change +=1
               continue
            step_acts = np.append(step_acts, accepted_actions, 0)
        self.statePrepare.changeNextState(step_acts)
        return self.score_ratio()
        
    
    def score_ratio (self):
        score_ratio = 0; remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score_ratio += ks.score_ratio()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        remain_cap_ratio = np.mean(remain_cap_ratio)
        #print('score_ratio:', score_ratio)
        #print('remain_cap_ratio:', remain_cap_ratio)
        return score_ratio
    