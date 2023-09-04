
"""

"""

import numpy as np
from src.data_structure.state_prepare import ExternalStatePrepare
from typing import Optional

class GreedySelect():
    def __init__ (
        self,
        num_instance: int,
        state_dataClass: ExternalStatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.greedy_num = num_instance
        self.no_change_long = 3
        self.reset()
    
    def _choose_actions_greedy_firstFit (
        self,
        pred_num: Optional[int]=None
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        self.caps, self.weightValues = self.statePrepare.getObservation1()
        values = self.weightValues[:, -1]
        weightSum = np.sum(self.weightValues[:, :-1],1)
        greedy_inst_indx = np.flip(np.argsort(np.trim_zeros(values)/np.trim_zeros(weightSum)))[:self.greedy_num]
        if pred_num is None:
            index_range = len(greedy_inst_indx)
        else:
            index_range = min(pred_num, len(greedy_inst_indx))
        for idx in range(index_range):
            inst_idx = greedy_inst_indx[idx]
            inst_idx += self.statePrepare.pad_len
            #if inst_idx >= self.statePrepare.pad_len:
            weight = self.statePrepare.getObservedInstWeight(inst_idx)
            for ks_idx in range(self.caps.shape[0]):
                ks_act = self.statePrepare.getRealKsAct(ks_idx)
                knapSack = self.statePrepare.getKnapsack(ks_act)
                eCap = knapSack.getExpectedCap()
                if all(eCap >= weight):
                    knapSack.removeExpectedCap(weight)
                    accepted_actions = np.append(accepted_actions, 
                                                 [[inst_idx, ks_act]], 0)
                    break
        return accepted_actions
    
    def test_step (
        self,
        pred_num: Optional[int]=None
    ):
        step_acts = np.zeros((0,2), dtype= int)
        no_change = 0
        #while no_change < self.no_change_long:
            
        accepted_actions = self._choose_actions_greedy_firstFit(pred_num)
        self.statePrepare.changeNextState(accepted_actions)
        #if len(accepted_actions) == 0:
        #    no_change +=1
        #    continue
        step_acts = np.append(step_acts, accepted_actions, 0)
        return self.score_ratio(), step_acts
    
    def getObservation (self):
        return self.caps, self.weightValues
    
    def reset(self):
        self.statePrepare.reset()

    def score_ratio (self):
        score_ratio = 0; remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score_ratio += ks.score_ratio()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        remain_cap_ratio = np.mean(remain_cap_ratio)
        #print('score_ratio:', score_ratio)
        #print('remain_cap_ratio:', remain_cap_ratio)
        return score_ratio
    