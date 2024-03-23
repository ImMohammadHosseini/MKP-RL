
"""

"""

import torch
import numpy as np
from typing import List

from torch.distributions.categorical import Categorical

from .src.ppo_base import PPOBase
from configs.ppo_configs import PPOConfig
from os import path

class FractionPPOTrainer_t1(PPOBase):
    r"""
    """
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/FractionPPOTrainer_t1/'
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
            reward = -(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW']))
        
        elif all(eCap >= weight):
            prompt[0,step+1] = torch.cat([inst, ks], 1).to(self.actor_model.device)
            step += 1
            reward = +(value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
        else:
            reward = -(value / np.sum(weight))
        print(prompt)
        reward = torch.tensor([reward]).unsqueeze(0)    
        return reward , accepted_action, step, prompt
    
    def train (
        self, 
        mode
    ):  
        if mode == 'normal':
            n_state = self.config.normal_batch
            batch_size = self.config.ppo_normal_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_normal_memory()
            criticModel = self.normal_critic_model
            criticOptim = self.normal_critic_optimizer
            
        elif mode == 'extra':
            n_state = self.config.extra_batch
            batch_size = self.config.ppo_extra_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_extra_memory()
            criticModel = self.extra_critic_model
            criticOptim = self.extra_critic_optimizer

        obs = memoryObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        acts = memoryAct.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        probs = memoryPrb.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        rewards = memoryRwd.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        vals = memoryVal.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        done = memoryDon.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        stps = memoryStp.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        intObs = memoryIntObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        
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
                #print(a_t)
                advantage[t] = a_t

            advantage = advantage.to(self.actor_model.device)
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch]
                batchVals = vals[batch]
                
                print(batchSteps.size())

                firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                    batchObs, batchSteps, batchIntObs)
                print(len(firstGenerated))
                #new_log_probs = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                #                             device=self.actor_model.device)
                #for i in range(batch_size):
                inst_dist = Categorical(firstGenerated)
                ks_dist = Categorical(secondGenerated)
                inst_log_probs = inst_dist.log_prob(batchActs[:,0].squeeze())                
                ks_log_probs = ks_dist.log_prob(batchActs[:,1].squeeze())
                new_log_probs = (inst_log_probs+ks_log_probs).squeeze()
                
                print(batchActs.size())

                newVal = criticModel(batchObs, batchIntObs) 

                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + batchVals

                critic_loss = (returns-newVal.squeeze())**2
                critic_loss = torch.mean(critic_loss)
                
                total_loss = (actor_loss + 0.5*critic_loss).detach()
                total_loss.requires_grad=True
                
                self.actor_optimizer.zero_grad()
                criticOptim.zero_grad()
                total_loss.backward()
                print(total_loss.grad)
                self.actor_optimizer.step()
                self.normal_critic_optimizer.step()

        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
    
class FractionPPOTrainer_t2(PPOBase):
    r"""
    """
    def __init__(
        self,
        info: List,
        save_path: str,
        config: PPOConfig,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,
        
    ):
        savePath = save_path +'/RL/FractionPPOTrainer_t2/'
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
            reward = -(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW']))
        
        elif all(eCap >= weight):
            prompt[0,step+1] = torch.cat([inst, ks], 1).to(self.actor_model.device)
            step += 1
            reward = +(value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
        else:
            reward = -(value / np.sum(weight))
            
        reward = torch.tensor([reward]).unsqueeze(0)    
        return reward , accepted_action, step, prompt
    
    def train (
        self, 
        mode
    ):  
        if mode == 'normal':
            n_state = self.config.normal_batch
            batch_size = self.config.ppo_normal_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_normal_memory()
            criticModel = self.normal_critic_model
            criticOptim = self.normal_critic_optimizer
            
        elif mode == 'extra':
            n_state = self.config.extra_batch
            batch_size = self.config.ppo_extra_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_extra_memory()
            criticModel = self.extra_critic_model
            criticOptim = self.extra_critic_optimizer

        obs = memoryObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        acts = memoryAct.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        probs = memoryPrb.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        rewards = memoryRwd.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        vals = memoryVal.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        done = memoryDon.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        stps = memoryStp.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        intObs = memoryIntObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        
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
                #print(a_t)
                advantage[t] = a_t

            advantage = advantage.to(self.actor_model.device)
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch]
                batchVals = vals[batch]
                
                firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                    batchObs, batchSteps, batchIntObs)
                
                inst_dist = Categorical(firstGenerated)
                ks_dist = Categorical(secondGenerated)
                inst_log_probs = inst_dist.log_prob(batchActs[:,0].squeeze())                
                ks_log_probs = ks_dist.log_prob(batchActs[:,1].squeeze())
                new_log_probs = (inst_log_probs+ks_log_probs)
                
                newVal = criticModel(batchObs, batchIntObs) 

                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = advantage[batch] + batchVals

                critic_loss = (returns-newVal.squeeze())**2
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
        
class FractionPPOTrainer_t3(PPOBase):
    r"""
    """
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
        savePath = save_path +'/RL/FractionPPOTrainer_t3'+'_dim'+str(dim)+'/'
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
        log_prob = act_dist.log_prob(act).unsqueeze(0)
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
        inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
        ks_act = statePrepares.getRealKsAct(int(
            act % self.actor_model.config.knapsack_obs_size))
        knapSack = statePrepares.getKnapsack(ks_act)
        ks = torch.tensor(statePrepares.getObservedKS(ks_act),
                          dtype=torch.float32).unsqueeze(0)
        inst = torch.tensor(statePrepares.getObservedInst(inst_act), 
                            dtype=torch.float32).unsqueeze(0)
            
        eCap = knapSack.getExpectedCap()
        weight = statePrepares.getObservedInstWeight(inst_act)
        value = statePrepares.getObservedInstValue(inst_act)
            
        if inst_act < statePrepares.pad_len or inst_act in accepted_action[:,0]: 
            reward = -(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW']))
        
        elif all(eCap >= weight):
            prompt[0,step] = torch.cat([inst, ks], 1).to(self.actor_model.device)
            step += 1
            reward = +(value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
        else:
            reward = -(value / np.sum(weight))
            
        reward = torch.tensor([reward]).unsqueeze(0)    
        return reward , accepted_action, step, prompt
    
    def train (
        self, 
        mode
    ):  
        if mode == 'normal':
            n_state = self.config.normal_batch
            batch_size = self.config.ppo_normal_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_normal_memory()
            criticModel = self.normal_critic_model
            criticOptim = self.normal_critic_optimizer
            
        elif mode == 'extra':
            n_state = self.config.extra_batch
            batch_size = self.config.ppo_extra_batch_size
            memoryObs, memoryAct, memoryPrb, memoryVal, memoryRwd, memoryDon, \
                memoryStp, memoryIntObs = self.memory.get_extra_memory()
            criticModel = self.extra_critic_model
            criticOptim = self.extra_critic_optimizer

        obs = memoryObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        acts = memoryAct.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        probs = memoryPrb.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        rewards = memoryRwd.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        vals = memoryVal.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        done = memoryDon.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        stps = memoryStp.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        intObs = memoryIntObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        
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
                #print(a_t)
                advantage[t] = a_t

            advantage = advantage.to(self.actor_model.device)
            for batch in batches:
                batchObs = obs[batch].detach()
                batchIntObs = intObs[batch].detach()
                batchActs = acts[batch].detach()
                batchSteps = stps[batch].detach()
                batchProbs = probs[batch].detach()
                batchVals = vals[batch].detach()
                batchAdvs = advantage[batch].detach()

                batchObs.requires_grad = True
                batchIntObs.requires_grad = True
                #batchActs.requires_grad = True
                #batchSteps.requires_grad = True
                batchProbs.requires_grad = True
                batchVals.requires_grad = True
                batchAdvs.requires_grad = True
                
                firstGenerated, _, _ = self.actor_model.generateOneStep(
                    batchObs, batchSteps, batchIntObs)
                act_dist = Categorical(firstGenerated)
                
                new_log_probs = act_dist.log_prob(batchActs.squeeze())
                newVal = criticModel(batchObs, batchIntObs) 
                prob_ratio = new_log_probs.exp() / batchProbs.exp()
                weighted_probs = batchAdvs * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*batchAdvs
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                returns = batchAdvs + batchVals
                
                critic_loss = (returns-newVal.squeeze())**2
                critic_loss = torch.mean(critic_loss)
                
                total_loss = actor_loss + 0.5*critic_loss
                self.actor_optimizer.zero_grad()
                criticOptim.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                criticOptim.step()

        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
    
   