
"""

"""
import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional
import time


from copy import deepcopy

from torch.autograd import Variable
from os import path, makedirs


class PPOBase():
    r"""
    this implementation is inspired by: 
        https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    """
    
    def __init__(
        self,
        info: List,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,

    ):
        super().__init__(config)
        
        self.info = info
        self.actor_model = actor_model
        self.normal_critic_model = normal_critic_model
        self.extra_critic_model = extra_critic_model

        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                    lr=self.config.actor_lr)
        self.normal_critic_optimizer = Adam(self.normal_critic_model.parameters(), 
                                             lr=self.config.critic_lr)
        self.extra_critic_optimizer = Adam(self.extra_critic_model.parameters(), 
                                             lr=self.config.critic_lr)
        
    def _choose_actions (
        self, 
        *args
    ):
        raise NotImplementedError("Not implemented")
        
    def internal_reward (
        self, 
        *args
    ):
        raise NotImplementedError("Not implemented")
        
    def make_steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
        fraction_flag: bool,
        step_flag: bool,
        pad_len_need: bool,
    ):
        accepted_action = np.array([[-1]*2]*self.config.generate_link_number, dtype= int)
        pad_len = statePrepares.pad_len if pad_len_need else 0
        actions = []; probs = []; values = []; rewards = []; internal_observation = []
        steps = []; step = 0; prompt = None
        for i in range(0, self.config.generat_link_number):
            firstGenerated, secondGenerated, prompt = self.actor_model.generateOneStep(
                externalObservation, step, prompt)
            act, prob = self._choose_actions(firstGenerated, secondGenerated)
            actions.append(act); probs.append(prob)
            internalReward = self.internal_reward(act, accepted_action, step, 
                                                  prompt, pad_len)
            steps.append(step); internal_observation.append(prompt); 
            rewards.append(internalReward)
            values.append(self.critic_model(externalObservation, prompt))
            
        if not fraction_flag: 
            rewards = sum(rewards)
            probs = sum(probs)
            
        if step_flag:
            return internalObservations, actions, accepted_action, probs, values, rewards, steps
        else:
            return None, actions, accepted_action, probs, values, rewards, None
    
    def train (
        self, 
    ):
        memoryObs, memoryIntObs, memoryAct, memoryPrb, memoryVal, memoryRwd, \
            memoryStp, memoryDon = self.replayBuffer.
            
        squeeze(1)
        
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch()
            
            advantage = torch.zeros(self.config.internal_batch,
                                    dtype=torch.float32, device=self.device)
            for t in range(self.config.internal_batch-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.internal_batch-1):
                    a_t += discount*(memoryRwd[k] + self.config.gamma*memoryVal[k+1]*\
                                     (1-int(done[k])) - memoryVal[k])                
                    discount *= self.config.gamma*self.config.gae_lambda
                #print(a_t)
                advantage[t] = a_t
        
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch].to(self.device)
                batchVals = vals[batch].to(self.device)