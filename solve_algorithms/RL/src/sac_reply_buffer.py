

import numpy as np
import torch
from typing import Optional
from configs.sac_configs import SACConfig
from configs.transformers_model_configs import TransformerKnapsackConfig


class SACReplayBuffer ():
    def __init__ (self, 
                  link_number, 
                  sacConfig: SACConfig,
                  modelConfig: TransformerKnapsackConfig,
                  action_num):
        self.link_number = link_number
        self.max_buffer_size = sacConfig.buffer_size
        self.normal_batch_size = sacConfig.normal_batch_size
        #add buffer for normal train
        self._normal_transitions_stored = 0
        self.normal_observation = np.zeros((self.max_buffer_size,
                                            modelConfig.max_length,
                                            modelConfig.input_encode_dim))
        self.normal_new_observation = np.zeros((self.max_buffer_size, 
                                                modelConfig.max_length,
                                                modelConfig.input_encode_dim))
        self.normal_action = np.zeros((self.max_buffer_size, sacConfig.generat_link_number,
                                      action_num))
        self.normal_reward = np.zeros((self.max_buffer_size, sacConfig.generat_link_number))
        self.normal_done = np.zeros(self.max_buffer_size, dtype=np.bool)
        self.normal_weights = np.zeros(self.max_buffer_size)
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.batch = None
        #TODO add normal_internalObservation normal_steps
        
        #TODO add buffer for extra train
    
    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.normal_weights[self.batch] = prediction_errors
        
    def save_normal_step (
        self, 
        observation: torch.tensor,
        new_observation: torch.tensor,
        actions: torch.tensor,
        rewards: torch.tensor,
        done: bool,
        steps: Optional[torch.tensor] = None,
        internalObservations: Optional[torch.tensor] = None,

    ):
        #TODO add normal_internalObservation normal_steps

        index = self._normal_transitions_stored % self.max_buffer_size
        self.normal_observation[index] = observation.cpu().detach().numpy()
        self.normal_new_observation[index] = new_observation.cpu().detach().numpy()
        self.normal_action[index] = torch.cat(actions,1).cpu().detach().numpy()
        self.normal_reward[index] = torch.cat(rewards,1).cpu().detach().numpy()
        self.normal_done[index] = done
        self.normal_weights[index] = self.max_weight
        self._normal_transitions_stored += 1
        
    def save_estra_step (
        self,
    ):
        raise NotImplementedError("Not implemented")
    
    def get_normal_memory (self):
        #TODO add normal_internal_Observation normal_steps

        max_mem = min(self._normal_transitions_stored, self.max_buffer_size)
        set_weights = self.normal_weights[:max_mem] + self.delta
        probs = set_weights / sum(set_weights)
        
        self.batch = np.random.choice(range(max_mem), self.normal_batch_size, p=probs, replace=False)
        
        return torch.from_numpy(self.normal_observation[self.batch]), \
        torch.from_numpy(self.normal_new_observation[self.batch]), \
        torch.from_numpy(self.normal_action[self.batch]), \
        torch.from_numpy(self.normal_reward[self.batch]), \
        torch.from_numpy(self.normal_done[self.batch])

    
    
    def get_extra_memory (
        self,
    ):
        raise NotImplementedError("Not implemented")