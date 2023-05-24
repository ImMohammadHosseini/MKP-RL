
"""

"""
import numpy as np
from src.data_structure.state_prepare import ExternalStatePrepare

class RandomSelect ():
    def __init__ (
        self,
        state_dataClass: ExternalStatePrepare,
    ):
        self.statePrepare = state_dataClass
        self.random_num = 3000
        self.no_change_long = 1e+3
        
    def _choose_actions_two_random (
        self,
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        caps, weightValues = self.statePrepare.getObservation()
        rand_inst_index = np.random.choice(weightValues.shape[0], self.random_num, replace=False)
        rand_ks_index = np.random.choice(caps.shape[0], self.random_num, replace=True)
        for inst_idx, ks_idx in zip(rand_inst_index, rand_ks_index):
            if inst_idx >= self.statePrepare.pad_len:
                knapSack = self.statePrepare.getKnapsack(ks_idx)
                eCap = knapSack.getExpectedCap()
                weight = self.statePrepare.getObservedInstWeight(inst_idx)
                if all(eCap >= weight):
                    knapSack.removeExpectedCap(weight)
                    accepted_actions = np.append(accepted_actions, 
                                                [[inst_idx, ks_idx]], 0)
        return accepted_actions
    
    def _choose_actions_random_firstFit (
        self,
    ):
        accepted_actions = np.zeros((0,2), dtype= int)
        if (self.statePrepare.is_terminated()):
            return accepted_actions
        caps, weightValues = self.statePrepare.getObservation()
        rand_inst_index = np.random.choice(weightValues.shape[0], self.random_num, replace=False)
        for inst_idx in rand_inst_index:
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
        print('choose two random action')
        while no_change < self.no_change_long:
            accepted_actions = self._choose_actions_two_random()
            if len(accepted_actions) == 0:
               no_change +=1 
               #if no_change % 1e2 == 0:
               #    print(no_change)
               continue
            step_acts = np.append(step_acts, accepted_actions, 0)
        print(len(step_acts))
        self.statePrepare.changeNextState(step_acts)
        self.score_ratio()
        
        self.statePrepare.reset()
        step_acts = np.zeros((0,2), dtype= int)
        no_change = 0
        print('choose instance randomly and knapsack first fit')
        while no_change < self.no_change_long:
            accepted_actions = self._choose_actions_random_firstFit()
            if len(accepted_actions) == 0:
               no_change +=1 
               #if no_change % 1e2 == 0:
               #    print(no_change)
               continue
            step_acts = np.append(step_acts, accepted_actions, 0)
        print(len(step_acts))
        self.statePrepare.changeNextState(step_acts)
        self.score_ratio()
    
    def score_ratio (self):
        score_ratio = 0; remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score_ratio += ks.score_ratio()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        remain_cap_ratio = np.mean(remain_cap_ratio)
        print('score_ratio:', score_ratio)
        print('remain_cap_ratio:', remain_cap_ratio)
        