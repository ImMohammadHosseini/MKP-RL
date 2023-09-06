
"""

"""

import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional
import time

from torch.distributions.categorical import Categorical
from src.data_structure.state_prepare import ExternalStatePrepare
from copy import deepcopy

from torch.autograd import Variable
from .src.base_trainer import BaseTrainer
from .src.ppo_base import PPOBase

from configs.ppo_configs import PPOConfig
from os import path, makedirs


class EncoderPPOTrainer_t2(PPOBase):
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/EncoderPPOTrainer_t2/'
        super().__init__(config, savePath, actor_model, normal_critic_model,
                         extra_critic_model)
        self.info = info
        
        if path.exists(self.save_path):
            self.load_models()
            self.pretrain_need = False
        else: self.pretrain_need = True
    
    def _choose_actions (
        self, 
        generatedAction,
        secondGenerated,
    ):pass
    
class EncoderPPOTrainer_t3(PPOBase):
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/EncoderPPOTrainer_t2/'
        super().__init__(config, savePath, actor_model, normal_critic_model,
                         extra_critic_model)
        self.info = info
        
        if path.exists(self.save_path):
            self.load_models()
            self.pretrain_need = False
        else: self.pretrain_need = True
        
    
    def _choose_actions (
        self, 
        firstGenerated,
        secondGenerated,
    ):
        act_dist = Categorical(firstGenerated)
        act = act_dist.sample()
        log_prob = act_dist.log_prob(act).unsqueeze(0).unsqueeze(0)
        return act.unsqueeze(0).unsqueeze(0), log_prob
    
    
    def normal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        step, 
        prompt, 
        statePrepares,
    ):
        act = int(action)
        inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
        ks_act = statePrepares.getRealKsAct(int(
            act % self.actor_model.config.knapsack_obs_size))
        knapSack = statePrepares.getKnapsack(ks_act)
            
        eCap = knapSack.getExpectedCap()
        weight = statePrepares.getObservedInstWeight(inst_act)
        value = statePrepares.getObservedInstValue(inst_act)
            
        if inst_act < statePrepares.pad_len: 
            reward = -(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW']))
        
        elif all(eCap >= weight):
            reward = +(value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
        else:
            reward = -(value / np.sum(weight))
            
        return torch.tensor([reward]).unsqueeze(0)
    
    def train (
        self, 
        mode = 'normal'
    ):  
        if mode == 'normal':
            n_state = self.config.normal_batch
            batch_size = self.config.ppo_normal_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon \
                = self.memory.get_normal_memory()
            criticModel = self.normal_critic_model
            criticOptim = self.normal_critic_optimizer
            
        elif mode == 'extra':
            n_state = self.config.extra_batch
            batch_size = self.config.ppo_extra_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon \
                 = self.memory.get_extra_memory()
            criticModel = self.extra_critic_model
            criticOptim = self.extra_critic_optimizer

        obs = memoryObs.squeeze().to(self.actor_model.device)
        acts = memoryAct.to(self.actor_model.device)
        probs = memoryPrb.to(self.actor_model.device)
        rewards = memoryRwd.to(self.actor_model.device)
        vals = memoryVal.to(self.actor_model.device)
        done = memoryDon.to(self.actor_model.device)
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)

            advantage = torch.zeros(self.config.normal_batch, dtype=torch.float32)
            for t in range(self.config.normal_batch-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.normal_batch-1):
                    a_t += discount*(rewards[k] + self.config.gamma*vals[k+1]*\
                                     (1-int(done[k])) - vals[k])                
                    discount *= self.config.gamma*self.config.gae_lambda
                advantage[t] = a_t

            advantage = advantage.to(self.actor_model.device)
            for batch in batches:
                batchObs = obs[batch]
                batchActs = acts[batch]
                batchProbs = probs[batch].squeeze()
                batchVals = vals[batch]

                firstGenerated, _, _ = self.actor_model.generateOneStep(
                    batchObs, None, None)
                
                act_dist = Categorical(firstGenerated)
                
                new_log_probs = act_dist.log_prob(batchActs.squeeze().to(self.actor_model.device))
                newVal = criticModel(batchObs)
                
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch].unsqueeze(dim=1) + batchVals
            
                critic_loss = (returns-newVal)**2
                critic_loss = torch.mean(critic_loss)

                total_loss = (actor_loss + 0.5*critic_loss).detach()
                total_loss.requires_grad=True
                
                self.actor_optimizer.zero_grad()
                criticOptim.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.normal_critic_optimizer.step()


        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
        
        
        
        
        
   