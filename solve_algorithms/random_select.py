
"""

"""
import numpy as np
from src.data_structure.state_prepare import StatePrepare
from typing import Optional

class RandomSelect ():
    def __init__ (
        self,
        state_dataClass: StatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.reset()
        
    def _choose_actions_two_random (
        self,
        pred_num: Optional[int]=None
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        self.caps, self.weightValues = self.statePrepare.getObservation()
        inst_num = len(self.weightValues[~np.all(self.weightValues == 0, axis=1)])
        if pred_num is None:
            random_num = inst_num
        else:
            random_num = min(pred_num, inst_num)
        rand_inst_index = np.random.choice(inst_num, random_num, replace=False)
        rand_ks_index = np.random.choice(self.caps.shape[0], random_num, replace=True)
        for inst_idx, ks_idx in zip(rand_inst_index, rand_ks_index):
            knapSack = self.statePrepare.getKnapsack(ks_idx)
            eCap = knapSack.getExpectedCap()
            weight = self.statePrepare.getObservedInstWeight(inst_idx+self.statePrepare.pad_len)
            if all(eCap >= weight):
                knapSack.removeExpectedCap(weight)
                accepted_actions = np.append(accepted_actions, 
                                             [[inst_idx+self.statePrepare.pad_len, ks_idx]], 0)
        return accepted_actions
    
    def _choose_actions_random_firstFit (
        self,
        pred_num: Optional[int]=None

    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        self.caps, self.weightValues = self.statePrepare.getObservation()
        inst_num = len(self.weightValues[~np.all(self.weightValues == 0, axis=1)])
        if pred_num is None:
            random_num = inst_num
        else:
            random_num = min(pred_num, inst_num)
        rand_inst_index = np.random.choice(inst_num, random_num, replace=False)
        for inst_idx in rand_inst_index:
            weight = self.statePrepare.getObservedInstWeight(inst_idx+self.statePrepare.pad_len)
            for ks_idx in range(self.caps.shape[0]):
                knapSack = self.statePrepare.getKnapsack(ks_idx)
                eCap = knapSack.getExpectedCap()
                if all(eCap >= weight):
                    knapSack.removeExpectedCap(weight)
                    accepted_actions = np.append(accepted_actions, 
                                                 [[inst_idx+self.statePrepare.pad_len, ks_idx]], 0)
                    break
        return accepted_actions
    
    
    def test_step_two_random (
        self,
        pred_num: Optional[int]=None
    ):
        
        step_acts = np.zeros((0,2), dtype= int)
            
        accepted_actions = self._choose_actions_two_random(pred_num)
        self.statePrepare.changeNextState(accepted_actions)
        #if len(accepted_actions) == 0:
        #    no_change +=1
        #    continue
        step_acts = np.append(step_acts, accepted_actions, 0)
        return self.score_ratio(), step_acts
    
    
        '''self.statePrepare.reset()
        step_acts = np.zeros((0,2), dtype= int)
        no_change = 0
        #print('choose two random action')
        while no_change < self.no_change_long:
            accepted_actions = self._choose_actions_two_random()
            if len(accepted_actions) == 0:
               no_change +=1 
               #if no_change % 1e2 == 0:
               #    print(no_change)
               continue
            step_acts = np.append(step_acts, accepted_actions, 0)
        self.statePrepare.changeNextState(step_acts)
        self.score_ratio()'''
        
    def test_step_randomInst_ksFirstFit (
        self,
        pred_num: Optional[int]=None
    ):
        step_acts = np.zeros((0,2), dtype= int)
            
        accepted_actions = self._choose_actions_random_firstFit(pred_num)
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

        