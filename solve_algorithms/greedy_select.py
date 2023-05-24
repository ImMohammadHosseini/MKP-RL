
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
        self.random_num = 3000
        self.no_change_long = 1e+3
    
    def _choose_actions_greedy_firstFit (
        self,
    ):
        caps, weightValues = self.statePrepare.getObservation()

    
    def test_step (
        self,
    ):
        pass
    
    def score_ratio (self):
        score_ratio = 0; remain_cap_ratio = []
        for ks in self.statePrepare.knapsacks:
            score_ratio += ks.score_ratio()
            remain_cap_ratio.append(ks.getRemainCap()/ks.getCap())
        remain_cap_ratio = np.mean(remain_cap_ratio)
        print('score_ratio:', score_ratio)
        print('remain_cap_ratio:', remain_cap_ratio)
    