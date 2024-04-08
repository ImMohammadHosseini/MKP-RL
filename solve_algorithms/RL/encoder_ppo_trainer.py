
"""

"""

import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional
import time

from torch.distributions.categorical import Categorical
from copy import deepcopy

from .src.ppo_base import PPOBase

from configs.ppo_configs import PPOConfig
from os import path, makedirs


class EncoderPPOTrainer_t2(PPOBase):
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        dim: int, 
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/EncoderPPOTrainer_t2/'
        super().__init__(config, savePath, dim, actor_model, normal_critic_model,
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
        inst_dist = Categorical(firstGenerated)
        ks_dist = Categorical(secondGenerated)
        inst_act = inst_dist.sample() 
        ks_act = ks_dist.sample()
        
        inst_log_probs = inst_dist.log_prob(inst_act)
        ks_log_probs = ks_dist.log_prob(ks_act)
            
        act = torch.cat([inst_act.unsqueeze(0), ks_act.unsqueeze(0)], dim=1)
        log_prob = (inst_log_probs+ks_log_probs).unsqueeze(0)
        
        return act, log_prob
    
    def normal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        step, 
        prompt, 
        statePrepares,
    ):
        inst_act = int(action[0,0]) 
        ks_act = statePrepares.getRealKsAct(int(action[0,1]))
        knapSack = statePrepares.getKnapsack(ks_act)
        ks = torch.tensor(statePrepares.getObservedKS(ks_act),
                          dtype=torch.float32).unsqueeze(0)
        inst = torch.tensor(statePrepares.getObservedInst(inst_act), 
                            dtype=torch.float32).unsqueeze(0)
            
        eCap = knapSack.getExpectedCap()
        weight = statePrepares.getObservedInstWeight(inst_act)
        value = statePrepares.getObservedInstValue(inst_act)
        
        
        if inst_act < statePrepares.pad_len or inst_act in accepted_action[:,0]: 
            reward = -(self.dim*self.info['VALUE_HIGH']/(self.dim*self.info['WEIGHT_LOW']))
        
        elif all(eCap >= weight):
            reward = +(self.dim*value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
            
        else:
            reward = -(self.dim*value / np.sum(weight))
            
        reward = torch.tensor([reward]).unsqueeze(0)    
        return reward , accepted_action, step, prompt
    
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
                batchObs = obs[batch].detach()
                batchActs = acts[batch].detach()
                batchProbs = probs[batch].squeeze().detach()
                batchVals = vals[batch].detach()
                batchAdvs = advantage[batch].detach()
                
                batchObs.requires_grad = True
                #batchActs.requires_grad = True
                batchProbs.requires_grad = True
                batchVals.requires_grad = True
                batchAdvs.requires_grad = True
                
                
                firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                    batchObs, None, None)
                
                inst_dist = Categorical(firstGenerated)
                ks_dist = Categorical(secondGenerated)
                inst_log_probs = inst_dist.log_prob(batchActs[:,:,0].squeeze())                
                ks_log_probs = ks_dist.log_prob(batchActs[:,:,1].squeeze())
                new_log_probs = (inst_log_probs+ks_log_probs)
                
                newVal = criticModel(batchObs)
                
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = batchAdvs * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*batchAdvs
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batchAdvs.unsqueeze(dim=1) + batchVals
            
                critic_loss = (returns-newVal)**2
                critic_loss = torch.mean(critic_loss)

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor_optimizer.zero_grad()
                criticOptim.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                criticOptim.step()


        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
                
    
class EncoderPPOTrainer_t3(PPOBase):
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        dim: int, 
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/EncoderPPOTrainer_t3/'
        super().__init__(config, savePath, dim, actor_model, normal_critic_model,
                         extra_critic_model)
        self.info = info
        
        self.dim = dim
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
        return act.unsqueeze(0), log_prob
    
    
    def normal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        step, 
        prompt, 
        statePrepares,
    ):
        act = int(action)
        if act == self.actor_model.config.knapsack_obs_size * self.actor_model.config.inst_obs_size:
            return torch.tensor([0]).unsqueeze(0), accepted_action, step, prompt
        inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
        ks_act = statePrepares.getRealKsAct(int(
            act % self.actor_model.config.knapsack_obs_size))
        knapSack = statePrepares.getKnapsack(ks_act)
            
        eCap = knapSack.getExpectedCap()
        weight = statePrepares.getObservedInstWeight(inst_act)
        value = statePrepares.getObservedInstValue(inst_act)
        
        if inst_act < statePrepares.pad_len: 
            reward = np.mean(-((self.dim*self.info['VALUE_HIGH'])/(self.dim*self.info['WEIGHT_LOW'])))
        
        elif all(eCap >= weight):
            reward = np.mean(+((self.dim*value) / np.sum(weight)))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]

        else:
            reward = np.mean(-((self.dim*value) / np.sum(weight)))
        reward = torch.tensor([reward]).unsqueeze(0)   
        return reward , accepted_action, step, prompt
    
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

        #criticModel = self.normal_critic_model
        #criticOptim = self.normal_critic_optimizer
        
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
                batchObs = obs[batch].detach()
                batchActs = acts[batch].detach()
                batchProbs = probs[batch].squeeze().detach()
                batchVals = vals[batch].detach()
                batchAdvs = advantage[batch].detach()
                
                batchObs.requires_grad = True
                #batchActs.requires_grad = True
                batchProbs.requires_grad = True
                batchVals.requires_grad = True
                batchAdvs.requires_grad = True
                
                firstGenerated, _, _ = self.actor_model.generateOneStep(
                    batchObs, None, None)
                
                act_dist = Categorical(firstGenerated)
                #entropy_loss = act_dist.entropy().mean()
                
                new_log_probs = act_dist.log_prob(batchActs.squeeze().to(self.actor_model.device))
                newVal = criticModel(batchObs)
                
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = batchAdvs * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*batchAdvs
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batchAdvs.unsqueeze(dim=1) + batchVals
            
                critic_loss = (returns-newVal)**2
                critic_loss = torch.mean(critic_loss)

                total_loss = actor_loss + 0.5*critic_loss# + 0.1*entropy_loss
                #total_loss.requires_grad=True
                
                self.actor_optimizer.zero_grad()
                criticOptim.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                criticOptim.step()


        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
        
        
        
        
        
   