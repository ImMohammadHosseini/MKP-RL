
"""

"""
import torch
import numpy as np
from torch.optim import Adam, SGD, RMSprop
from typing import List, Optional

from torch.distributions.categorical import Categorical
from src.data_structure.state_prepare import ExternalStatePrepare

from torch.autograd import Variable
from.src.ppo_base import PPOBase
from configs.ppo_configs import PPOConfig
from os import path, makedirs


class WholePPOTrainer_t1(PPOBase):
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
        savePath = save_path +'/RL/WholePPOTrainer_t1/'
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
            
        return torch.tensor([reward]).unsqueeze(0)
    
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
        
        obs = memoryObs.to(self.actor_model.device)
        acts = memoryAct.to(self.actor_model.device)
        probs = memoryPrb.to(self.actor_model.device)
        rewards = memoryRwd.to(self.actor_model.device)
        vals = memoryVal.to(self.actor_model.device)
        done = memoryDon.to(self.actor_model.device)
        stps = memoryStp.to(self.actor_model.device)
        intObs = memoryIntObs.to(self.actor_model.device)
        
        print(obs.size())# = memoryObs.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(acts.size())#  = memoryAct.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(probs.size())#  = memoryPrb.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(rewards.size())#  = memoryRwd.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(vals.size())#  = memoryVal.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(done.size())#  = memoryDon.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(stps.size())#  = memoryStp.flatten(start_dim=0,end_dim=1).to(self.actor_model.device)
        print(intObs.size())# 
        
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)
            advantage = torch.zeros(self.config.normal_batch, dtype=torch.float32)
            for t in range(self.config.normal_batch-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.normal_batch-1):
                    a_t += discount*(rewards[k] + self.config.gamma*vals[k+1].sum()*\
                                     (1-int(done[k,-1])) - vals[k].sum())                
                    discount *= self.config.gamma*self.config.gae_lambda
                advantage[t] = a_t
            advantage = advantage.to(self.actor_model.device)
    
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch]
                batchVals = vals[batch]
                
                new_log_probs = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                newVal = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                for i in range(batch_size):
                    firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                        batchObs[i], batchSteps[i], batchIntObs[i])
                    inst_dist = Categorical(firstGenerated)
                    ks_dist = Categorical(secondGenerated)
                    inst_log_probs = inst_dist.log_prob(batchActs[i,:,0].squeeze())
                    print(inst_log_probs.size())
                    
                    ks_log_probs = ks_dist.log_prob(batchActs[i,:,1].squeeze())
                    new_log_probs[i] = (inst_log_probs+ks_log_probs).sum()
                    
                    newVal[i] = criticModel(batchObs[i], batchIntObs[i]).sum()
                
                prob_ratio = new_log_probs.exp() / batchProbs.squeeze().exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + batchVals.sum()
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

    
class WholePPOTrainer_t2(PPOBase):
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
        savePath = save_path +'/RL/WholePPOTrainer_t2/'
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
            
        return torch.tensor([reward]).unsqueeze(0)
    
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
        
        obs = memoryObs.to(self.actor_model.device)
        acts = memoryAct.to(self.actor_model.device)
        probs = memoryPrb.to(self.actor_model.device)
        rewards = memoryRwd.to(self.actor_model.device)
        vals = memoryVal.to(self.actor_model.device)
        done = memoryDon.to(self.actor_model.device)
        stps = memoryStp.to(self.actor_model.device)
        intObs = memoryIntObs.to(self.actor_model.device)
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)
            advantage = torch.zeros(self.config.normal_batch, dtype=torch.float32)
            for t in range(self.config.normal_batch-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.normal_batch-1):
                    a_t += discount*(rewards[k] + self.config.gamma*vals[k+1].sum()*\
                                     (1-int(done[k,-1])) - vals[k].sum())                
                    discount *= self.config.gamma*self.config.gae_lambda
                advantage[t] = a_t
            advantage = advantage.to(self.actor_model.device)
    
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch]
                batchVals = vals[batch]
                
                new_log_probs = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                newVal = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                for i in range(batch_size):
                    firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                        batchObs[i], batchSteps[i], batchIntObs[i])
                    inst_dist = Categorical(firstGenerated)
                    ks_dist = Categorical(secondGenerated)
                    inst_log_probs = inst_dist.log_prob(batchActs[i,:,0].squeeze())
                    
                    ks_log_probs = ks_dist.log_prob(batchActs[i,:,1].squeeze())
                    new_log_probs[i] = (inst_log_probs+ks_log_probs).sum()
                    
                    newVal[i] = criticModel(batchObs[i], batchIntObs[i]).sum()
                
                prob_ratio = new_log_probs.exp() / batchProbs.squeeze().exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + batchVals.sum()
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
    
class WholePPOTrainer_t3(PPOBase):
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
        savePath = save_path +'/RL/WholePPOTrainer_t3/'
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
            prompt[0,step+1] = torch.cat([inst, ks], 1).to(self.actor_model.device)
            step += 1
            reward = +(value / np.sum(weight))
            knapSack.removeExpectedCap(weight)
            accepted_action = np.append(accepted_action,
                                         [[inst_act, ks_act]], 0)[1:]
        else:
            reward = -(value / np.sum(weight))
            
        return torch.tensor([reward]).unsqueeze(0)
    
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

        obs = memoryObs.to(self.actor_model.device)
        acts = memoryAct.to(self.actor_model.device)
        probs = memoryPrb.to(self.actor_model.device)
        rewards = memoryRwd.to(self.actor_model.device)
        vals = memoryVal.to(self.actor_model.device)
        done = memoryDon.to(self.actor_model.device)
        stps = memoryStp.to(self.actor_model.device)
        intObs = memoryIntObs.to(self.actor_model.device)
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)
            advantage = torch.zeros(self.config.normal_batch, dtype=torch.float32)
            for t in range(self.config.normal_batch-1):
                discount = 1
                a_t = 0
                for k in range(t, self.config.normal_batch-1):
                    a_t += discount*(rewards[k] + self.config.gamma*vals[k+1].sum()*\
                                     (1-int(done[k,-1])) - vals[k].sum())                
                    discount *= self.config.gamma*self.config.gae_lambda
                advantage[t] = a_t
            advantage = advantage.to(self.actor_model.device)
    
            for batch in batches:
                batchObs = obs[batch]
                batchIntObs = intObs[batch]
                batchActs = acts[batch]
                batchSteps = stps[batch]
                batchProbs = probs[batch]
                batchVals = vals[batch]
                
                new_log_probs = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                newVal = torch.tensor([0]*batch_size,  dtype=torch.float64, 
                                             device=self.actor_model.device)
                for i in range(batch_size):
                    firstGenerated, _, _ = self.actor_model.generateOneStep(
                        batchObs[i], batchSteps[i], batchIntObs[i])
                    act_dist = Categorical(firstGenerated)
                    new_log_probs[i] = act_dist.log_prob(batchActs[i].squeeze()).sum()
                    newVal[i] = criticModel(batchObs[i], batchIntObs[i]).sum()
                
                prob_ratio = new_log_probs.exp() / batchProbs.squeeze().exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                            1+self.config.cliprange)*advantage[batch]
                print(weighted_probs.size())
                print(weighted_clipped_probs.size())

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                returns = advantage[batch] + batchVals.sum()
                print(returns.size())

                critic_loss = (returns-newVal.squeeze())**2
                print(critic_loss.size())
                print(self.dd)
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
class PPOTrainer():
    r"""
    this implementation is inspired by: 
        https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_trainer.py and 
        https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    """
    
    def __init__(
        self,
        info: List,
        save_path: str,
        main_batch_size: int,
        config: Optional[PPOConfig] = None,
        actor_model: Optional[torch.nn.Module] = None,
        critic_model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(config)
        
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        
        
        self.device = device
        self.info = info
        self.main_batch_size = main_batch_size
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_model = self.critic_model.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = Adam(
                self.actor_model.parameters(), lr=self.config.actor_lr
            ) 
        else:
            self.actor_optimizer = actor_optimizer
            
        if critic_optimizer is None:
            self.critic_optimizer = Adam(
                self.critic_model.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer = critic_optimizer
            
        self.savePath = save_path +'/RL/PPOTrainer/'
        if path.exists(self.savePath):
            self.load_models()
            self.pretrain_need = False
        else: self.pretrain_need = True
            
        self.batch_actor_loss = torch.tensor(0., device=self.device)
        self.batch_critic_loss = torch.tensor(0., device=self.device)
    
        self._reset_memory()
        
    def _reset_memory (self) :
        self.memory_observation = [] 
        self.memory_internalObservation = []
        self.memory_action = [] 
        self.memory_prob = [] 
        self.memory_val = []
        self.memory_reward = []
        self.memory_steps = []
        self.memory_done = []
        
    def save_step (
        self, 
        observation: torch.tensor,
        internalObservation: torch.tensor,
        actions: np.ndarray, 
        sumProbs: torch.tensor,
        sumVals: torch.tensor, 
        sumRewards: torch.tensor,
        steps: torch.tensor,
        done: bool, 
    ):
        self.memory_observation.append(observation.cpu().unsqueeze(1))
        self.memory_internalObservation.append(internalObservation.cpu().unsqueeze(1))
        self.memory_action.append(actions.data.cpu().unsqueeze(1))
        #sumProbs = sumProbs.data
        self.memory_prob.append(sumProbs.data.cpu().unsqueeze(1))
        self.memory_val.append(sumVals.data.cpu().unsqueeze(1))
        self.memory_reward.append(sumRewards.cpu().unsqueeze(1))
        self.memory_steps.append(steps.cpu().unsqueeze(1))
        self.memory_done.append(torch.tensor([done]*self.main_batch_size).unsqueeze(1))
        
    def _choose_actions (
        self, 
        generatedInstance: torch.tensor, 
        generatedKnapsack: torch.tensor, 
    ):
        inst_dist = Categorical(generatedInstance)
        ks_dist = Categorical(generatedKnapsack)
        inst_act = inst_dist.sample() 
        ks_act = ks_dist.sample()
        
        inst_log_probs = inst_dist.log_prob(inst_act)
        ks_log_probs = ks_dist.log_prob(ks_act)
        acts = torch.cat([inst_act,
                          ks_act],dim=1)
        log_probs = inst_log_probs + ks_log_probs
        
        return acts, log_probs
    
    def reward (
        self,
        actions: torch.tensor,
        accepted_actions: np.ndarray,
        step: torch.tensor,
        prompt: torch.tensor,
        statePrepares: np.ndarray,
    ):
        rewards = []
        for index, act in enumerate(actions):
            inst_act = int(act[0])
            ks_act = statePrepares[index].getRealKsAct(int(act[1]))
            knapSack = statePrepares[index].getKnapsack(ks_act)
            ks = torch.tensor(statePrepares[index].getObservedKS(ks_act),
                              device=self.device, dtype=torch.float32).unsqueeze(0)
            inst = torch.tensor(statePrepares[index].getObservedInst(inst_act), 
                                device=self.device, dtype=torch.float32).unsqueeze(0)
            
            
            eCap = knapSack.getExpectedCap()
            weight = statePrepares[index].getObservedInstWeight(inst_act)
            
            if inst_act < statePrepares[index].pad_len or inst_act in accepted_actions[index,:,0]: 
                    rewards.append(-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                
                prompt[index][step[index]+1] = torch.cat([inst, ks], 1)
                step[index] += 1
                value = statePrepares[index].getObservedInstValue(inst_act)
                rewards.append(+(value / np.sum(weight)))
                knapSack.removeExpectedCap(weight)
                accepted_actions[index] = np.append(accepted_actions[index][1:],
                                                    [[inst_act, ks_act]], 0)
            else:
                rewards.append(-self.info['VALUE_LOW'])
        return torch.tensor(rewards, device=self.device)
    
    def make_steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
    ):
        actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int, device=self.device)
        sumProbs = torch.tensor([0]*self.main_batch_size, dtype=torch.float64, 
                                 device=self.device)
        sumRewards = torch.tensor([0]*self.main_batch_size, dtype=torch.float64, 
                                   device=self.device)
        internalObservation = torch.zeros((self.main_batch_size,0,self.config.generat_link_number+1,
                                           self.actor_model.config.input_decode_dim),
                                          device=self.device)
        
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        step = torch.tensor([0]*self.main_batch_size, dtype=torch.int64, device=self.device)
        steps = torch.zeros((self.main_batch_size,0), dtype=torch.int64, device=self.device)
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                step, externalObservation, prompt)
            internalObservation = torch.cat([internalObservation, prompt.unsqueeze(1)], 1)
            act, prob = self._choose_actions(generatedInstance, generatedKnapsack)
            actions = torch.cat([actions, act.unsqueeze(1)], 1)
            steps = torch.cat([steps, step.unsqueeze(1)], 1)
            reward = self.reward(act, accepted_actions, step, prompt, statePrepares)
            #print(internalObservation[-1])
            sumProbs += prob.squeeze()
            sumRewards += reward
            
            
        values = self.critic_model(externalObservation).squeeze(1)
        return internalObservation, actions, accepted_actions, sumProbs, values, sumRewards, steps
        
    def test_step (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
    ):
        prompt = None
        step_acts = np.zeros((0,2), dtype= int)
        for i in range(0, self.config.generat_link_number):
            stepGenerate, prompt = self.generate_step(externalObservation[0].unsqueeze(0), 2, 
                                                      self.config.generat_link_number, prompt)
            
            act, prob = self._choose_actions(stepGenerate, externalObservation)
            internalReward, accepted_action, accepted_prob = self.internal_reward(
                act, prob, statePrepare)
            
            if len(accepted_action) > 0:
                step_acts = np.append(step_acts, accepted_action, 0)
            else:
                prompt = prompt[:,:-2]
        
        return step_acts
        
    
    
    def train_minibatch (
        self, 
    ):
        memoryObs = torch.cat(self.memory_observation, 1)
        memoryIntObs = torch.cat(self.memory_internalObservation, 1)
        memoryAct = torch.cat(self.memory_action, 1) 
        memoryPrb = torch.cat(self.memory_prob, 1)
        memoryVal = torch.cat(self.memory_val, 1)
        memoryRwd = torch.cat(self.memory_reward, 1)
        memoryStp = torch.cat(self.memory_steps, 1)
        memoryDon = torch.cat(self.memory_done, 1)
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch()
            for index in range(self.main_batch_size):
                obs = memoryObs[index]
                intObs = memoryIntObs[index]
                acts = memoryAct[index]
                probs = memoryPrb[index]
                rewards = memoryRwd[index]
                vals = memoryVal[index]
                stps = memoryStp[index]
                done = memoryDon[index]
                advantage = torch.zeros(self.config.internal_batch,
                                        dtype=torch.float32, device=self.device)
                
                for t in range(self.config.internal_batch-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, self.config.internal_batch-1):
                        a_t += discount*(rewards[k] + self.config.gamma*vals[k+1]*\
                                         (1-int(done[k])) - vals[k])                
                        discount *= self.config.gamma*self.config.gae_lambda
                    advantage[t] = a_t
            
                for batch in batches:
                    batchObs = obs[batch]
                    batchIntObs = intObs[batch]
                    batchActs = acts[batch]
                    batchSteps = stps[batch]
                    batchProbs = probs[batch].to(self.device)
                    batchVals = vals[batch].to(self.device)
                    
                    new_log_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                                 dtype=torch.float64, device=self.device)

                    for i in range(0, self.config.generat_link_number):
                        generatedInstance, generatedKnapsack, _ = self.actor_model.generateOneStep(
                            batchSteps[:,i], batchObs.to(self.device), batchIntObs[:,i])
                        inst_dist = Categorical(generatedInstance.squeeze())
                        ks_dist = Categorical(generatedKnapsack.squeeze())
                        batchActs = batchActs.to(self.device)
                        inst_log_probs = inst_dist.log_prob(batchActs[:,i,0])#.unsqueeze(1))
                        ks_log_probs = ks_dist.log_prob(batchActs[:,i,1])#.unsqueeze(1))
                        newProbs = inst_log_probs + ks_log_probs
                        new_log_probs += newProbs.squeeze()
                    
                    try:
                        newVal = self.critic_model(batchObs.to(self.device)).squeeze()

                    except:
                        print(batchObs)
                        print(batchObs.size())
                        newVal = self.critic_model(batchObs.to(self.device)).squeeze()

                    prob_ratio = (new_log_probs - batchProbs).exp().to(self.device)
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                                1+self.config.cliprange)*advantage[batch]
                    
                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    returns = advantage[batch].unsqueeze(dim=1) + batchVals.unsqueeze(dim=1)
                    
                    critic_loss = (returns-newVal)**2
                    critic_loss = critic_loss.mean()
                    
                    actor_loss_leaf = Variable(actor_loss.data, requires_grad=True)
                    critic_loss_leaf = Variable(critic_loss.data, requires_grad=True)
                    
                    total_loss = actor_loss_leaf + 0.5*critic_loss_leaf
                    
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    total_loss.backward()
                    #actor_loss_leaf.backward()
                    self.actor_optimizer.step()
                    #self.critic_optimizer.zero_grad()
                    #critic_loss_leaf.backward()#retain_graph=True
                    self.critic_optimizer.step()
                    
        self._reset_memory()
    
            
    def load_models (self):
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_models (self):
        if not path.exists(self.savePath):
            makedirs(self.savePath)
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_model.state_dict(),
            'optimizer_state_dict': self.critic_optimizer.state_dict()}, 
            file_path)
        
        