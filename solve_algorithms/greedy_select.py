
"""

"""

import numpy as np
from src.data_structure.state_prepare import StatePrepare
from typing import Optional

class GreedySelect():
    def __init__ (
        self,
        num_instance: int,
        state_dataClass: StatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.greedy_num = num_instance
        self.no_change_long = 3
        #self.reset()
    
    def _choose_actions_greedy_firstFit (
        self,
        greedy_method,
        pred_num: Optional[int]=None
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        self.caps, self.weightValues = self.statePrepare.getObservation1()
        if greedy_method == 'minw':
            weightMean = np.mean(self.weightValues[:, :len(self.statePrepare.getObservedInstValue(0))],1)
            greedy_inst_indx = np.flip(np.argsort(np.trim_zeros(weightMean)))[:self.greedy_num]
            
        elif greedy_method == 'maxv':
            valueMean = np.mean(self.weightValues[:, -len(self.statePrepare.getObservedInstValue(0)):],1)
            greedy_inst_indx = np.flip(np.argsort(np.trim_zeros(valueMean)))[:self.greedy_num]
            
        elif greedy_method == 'maxvdw':
            valueMean = np.mean(self.weightValues[:, -len(self.statePrepare.getObservedInstValue(0)):],1)
            weightMean = np.mean(self.weightValues[:, :len(self.statePrepare.getObservedInstValue(0))],1)
            greedy_inst_indx = np.flip(np.argsort(np.trim_zeros(valueMean)/np.trim_zeros(weightMean)))[:self.greedy_num]
                
        if pred_num is None:
            index_range = len(greedy_inst_indx)
        else:
            index_range = min(pred_num, len(greedy_inst_indx))
        
        steps = 0
        for idx in range(index_range):
            steps += 1
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
        return accepted_actions, steps
    
    
    
    def test_step (
        self,
        greedy_method,
        pred_num: Optional[int]=None
    ):
        step_acts = np.zeros((0,2), dtype= int)
        no_change = 0
        #while no_change < self.no_change_long:
            
        accepted_actions, steps = self._choose_actions_greedy_firstFit(greedy_method, pred_num)
        self.statePrepare.changeNextState(accepted_actions)
        #if len(accepted_actions) == 0:
        #    no_change +=1
        #    continue
        step_acts = np.append(step_acts, accepted_actions, 0)
        score, remain_cap = self.final_score()
        return score, remain_cap, steps
    
    def getObservation (self):
        return self.caps, self.weightValues
    
    def reset(self):
        self.statePrepare.reset()

    def final_score (self):
        score = 0
        remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score += ks.getValues()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        return score, np.mean(remain_cap_ratio)
    
    