
"""

"""
import torch
from typing import Optional

class PPOReplayBuffer ():
    def __init__ (self, link_number):
        self.link_number = link_number
        self.reset_internal_memory()
        self.reset_external_memory()
        
    
    def reset_normal_memory (self) :
        self.normal_observation = [] 
        self.normal_internalObservation = []
        self.normal_action = [] 
        self.normal_prob = [] 
        self.normal_val = []
        self.normal_reward = []
        self.normal_steps = []
        self.normal_done = []
    
    def reset_extra_memory (self) :
        self.extra_observation = [] 
        self.extra_internalObservation = []
        self.extra_action = [] 
        self.extra_prob = [] 
        self.extra_val = []
        self.extra_reward = []
        self.extra_steps = []
        self.extra_done = []
        
    def save_normal_step (
        self, 
        observation: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        rewards: torch.tensor,
        done: bool,
        steps: Optional[torch.tensor] = None,
        internalObservations: Optional[torch.tensor] = None,

    ):
        self.normal_observation.append(torch.cat([observation.unsqueeze(1)]*self.link_number, 1).cpu())
        self.normal_action.append(actions.cpu())
        self.normal_prob.append(probs.cpu())
        self.normal_val.append(values.cpu())
        self.normal_reward.append(rewards.cpu())
        
        if done == True:
            self.normal_done.append(torch.tensor([False]*(self.link_number-1)+[True]))
        else:
            self.normal_done.append(torch.tensor([done]*self.link_number))
            
        if not isinstance(steps, type(None)):
            self.normal_steps.append(steps.cpu())

        if not isinstance(internalObservations, type(None)):
            self.normal_internalObservation.append(internalObservations.cpu())

    def save_extra_step (
        self, 
        observation: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        rewards: torch.tensor,
        done: bool,
        steps: Optional[torch.tensor] = None,
        internalObservations: Optional[torch.tensor] = None,
    ):
        self.extra_observation.append(torch.cat([observation.unsqueeze(1)]*self.link_number, 1).cpu())
        self.extra_action.append(actions.cpu())
        self.extra_prob.append(probs.cpu())
        self.extra_val.append(values.cpu())
        self.extra_reward.append(rewards.cpu())
        
        if done == True:
            self.extra_done.append(torch.tensor([False]*(self.link_number-1)+[True]))
        else:
            self.extra_done.append(torch.tensor([done]*self.link_number))
            
        if not isinstance(steps, type(None)):
            self.extra_steps.append(steps.cpu())

        if not isinstance(internalObservations, type(None)):
            self.extra_internalObservation.append(internalObservations.cpu())
        
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches
        