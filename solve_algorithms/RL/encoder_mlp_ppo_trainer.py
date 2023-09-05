
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
        generatedAction,
        secondGenerated,
    ):
        acts = torch.zeros((0,1), dtype=torch.int)
        log_probs = torch.zeros((0,1), dtype=torch.int)
        for ga in generatedAction:
            act_dist = Categorical(ga)
            act = act_dist.sample() 
            
            acts = torch.cat([acts, act.unsqueeze(0).unsqueeze(0).cpu()], dim=0)
            log_probs = torch.cat([log_probs, act_dist.log_prob(act).unsqueeze(0).unsqueeze(0).cpu()], dim=0)
            print('fff', acts.size())
        return acts, log_probs
    
    
    def normal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        step, 
        prompt, 
        statePrepares,
    ):
        rewards = []
        for index, act in enumerate(action):
            #print(act)
            inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
            ks_act = statePrepares.getRealKsAct(int(
                act % self.actor_model.config.knapsack_obs_size))
            knapSack = statePrepares.getKnapsack(ks_act)
            
            eCap = knapSack.getExpectedCap()
            weight = statePrepares.getObservedInstWeight(inst_act)
            value = statePrepares.getObservedInstValue(inst_act)
            
            if inst_act < statePrepares.pad_len: 
                    rewards.append(-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                rewards.append(+(value/ np.sum(weight)))
                knapSack.removeExpectedCap(weight)
                
                accepted_action = np.append(accepted_action,
                                            [[inst_act, ks_act]], 0)[1:]
            else:
                rewards.append(-(value / np.sum(weight)))#-self.info['VALUE_LOW'])
        #print(rewards)
        return torch.tensor(rewards).unsqueeze(0)
    
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

        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)
            
            obs = memoryObs.squeeze().to(self.actor_model.device)
            acts = memoryAct.to(self.actor_model.device)
            probs = memoryPrb.to(self.actor_model.device)
            rewards = memoryRwd.to(self.actor_model.device)
            vals = memoryVal.to(self.actor_model.device)
            done = memoryDon.to(self.actor_model.device)

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
                batchObs = obs[batch].to(self.actor_model.device)

                batchActs = acts[batch]
                batchProbs = probs[batch].squeeze().to(self.actor_model.device)
                batchVals = vals[batch].to(self.actor_model.device)

                firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
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
                print(total_loss.grad)
                self.actor_optimizer.step()
                self.normal_critic_optimizer.step()


        self.memory.reset_normal_memory() if mode == 'normal' else \
            self.memory.reset_extra_memory()
        
        
        
        
        
        
        
class EncoderMlpPPOTrainer(BaseTrainer):
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
        config: PPOConfig,
        actor_model: torch.nn.Module,
        critic_model: torch.nn.Module,
        external_critic_model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        external_critic_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(config)
        
        self.device = device
        self.softmax = torch.nn.Softmax()
        self.info = info
        self.main_batch_size = main_batch_size
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_model = self.critic_model.to(self.device)
        self.external_critic_model = external_critic_model
        self.external_critic_model = self.external_critic_model.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = Adam(self.actor_model.parameters(), lr=self.config.actor_lr
            )
        else:
            self.actor_optimizer = actor_optimizer
        
        if critic_optimizer is None:
            self.critic_optimizer = Adam(self.critic_model.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer = critic_optimizer
            
        if external_critic_optimizer is None:
            self.external_critic_optimizer = Adam(self.external_critic_model.parameters(), lr=self.config.critic_lr
            )
        else:
            self.external_critic_model = external_critic_model
            
        self.savePath = save_path +'/RL/EncoderMlpPPOTrainer/'
        if path.exists(self.savePath):
            self.load_models()
            self.pretrain_need = False
        else: self.pretrain_need = True
            
        
        self._reset_memory()
        self._reset_ext_memory()
        
    def _reset_memory (self) :
        self.memory_observation = [] 
        self.memory_action = [] 
        self.memory_prob = [] 
        self.memory_val = []
        self.memory_internalReward = []
        self.memory_done = []
        
    def save_step (
        self, 
        externalObservation: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        internalRewards: torch.tensor,
        done: bool,
    ):
        self.memory_observation.append(torch.cat([externalObservation.unsqueeze(1)]*10, 1).cpu())
        self.memory_action.append(actions.data.cpu())
        self.memory_prob.append(probs.data.cpu())
        self.memory_val.append(values.data.cpu())
        self.memory_internalReward.append(internalRewards.cpu())
        self.memory_done.append(torch.tensor([[done]]*self.main_batch_size))
    
    def _reset_ext_memory (self) :
        self.memory_ext_observation = [] 
        self.memory_ext_action = [] 
        self.memory_ext_prob = [] 
        self.memory_ext_val = []
        self.memory_ext_internalReward = []
        self.memory_ext_done = []
        
    def save_ext_step (
        self, 
        externalObservation: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        internalRewards: torch.tensor,
        done: bool,
    ):
        self.memory_ext_observation.append(torch.cat([externalObservation.unsqueeze(1)]*10, 1).cpu())
        self.memory_ext_action.append(actions.data.cpu())
        self.memory_ext_prob.append(probs.data.cpu())
        self.memory_ext_val.append(values.data.cpu())
        self.memory_ext_internalReward.append(internalRewards.cpu())
        self.memory_ext_done.append(torch.tensor([[done]]*self.main_batch_size))
        
    def _choose_actions (
        self, 
        generatedAction,
    ):
        acts = torch.zeros((0,1), dtype=torch.int)
        log_probs = torch.zeros((0,1), dtype=torch.int)
        for ga in generatedAction:
            act_dist = Categorical(ga)
            act = act_dist.sample() 
            
            acts = torch.cat([acts, act.unsqueeze(0).unsqueeze(0).cpu()], dim=0)
            log_probs = torch.cat([log_probs, act_dist.log_prob(act).unsqueeze(0).unsqueeze(0).cpu()], dim=0)
        return acts, log_probs
    
    
    def internal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        statePrepares: np.ndarray,
    ):
        rewards = []
        for index, act in enumerate(action):
            #print(act)
            inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
            ks_act = statePrepares[index].getRealKsAct(int(
                act % self.actor_model.config.knapsack_obs_size))
            knapSack = statePrepares[index].getKnapsack(ks_act)
            
            eCap = knapSack.getExpectedCap()
            weight = statePrepares[index].getObservedInstWeight(inst_act)
            value = statePrepares[index].getObservedInstValue(inst_act)
            
            if inst_act < statePrepares[index].pad_len: 
                    rewards.append(-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                rewards.append(+(value/ np.sum(weight)))
                knapSack.removeExpectedCap(weight)
                accepted_action[index] = np.append(accepted_action[index][1:],
                                                   [[inst_act, ks_act]], 0)
            else:
                rewards.append(-(value / np.sum(weight)))#-self.info['VALUE_LOW'])
        #print(rewards)
        return torch.tensor(rewards, device=self.device).unsqueeze(0)
    
    def make_steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
    ):
        accepted_action = np.array([[[-1]*2]*1]*self.main_batch_size, dtype= int)
        generatedAction = self.actor_model.generateOneStep(externalObservation)
        act, prob = self._choose_actions(generatedAction)
        value = self.critic_model(externalObservation)
        internalReward = self.internal_reward(act, accepted_action, statePrepares)
        #print(accepted_action)
        return act, accepted_action, prob, value, internalReward
    
    def train_minibatch (
        self, 
        mode = 'int'
    ):
        '''for param in self.actor_model.parameters():
            param.requires_grad = True
        for param in self.critic_model.parameters():
            param.requires_grad = True
        for param in self.external_critic_model.parameters():
            param.requires_grad = True'''
            
        if mode == 'int':
            n_state = self.config.internal_batch
            batch_size = self.config.ppo_int_batch_size
            memoryObs = torch.cat(self.memory_observation, 1)
            memoryAct = torch.cat(self.memory_action, 1) 
            memoryPrb = torch.cat(self.memory_prob, 1)
            memoryVal = torch.cat(self.memory_val, 1)
            memoryRwd = torch.cat(self.memory_internalReward, 1)
            memoryDon = torch.cat(self.memory_done, 1)
            criticModel = self.critic_model
        elif mode == 'ext':
            n_state = self.config.external_batch
            batch_size = self.config.ppo_ext_batch_size
            memoryObs = torch.cat(self.memory_ext_observation, 1)
            memoryAct = torch.cat(self.memory_ext_action, 1) 
            memoryPrb = torch.cat(self.memory_ext_prob, 1)
            memoryVal = torch.cat(self.memory_ext_val, 1)
            memoryRwd = torch.cat(self.memory_ext_internalReward, 1)
            memoryDon = torch.cat(self.memory_ext_done, 1)
            criticModel = self.external_critic_model

        
        #memoryObs.requires_grad = True
        #memoryAct.requires_grad = True 
        #memoryPrb.requires_grad = True
        #memoryVal.requires_grad = True
        #memoryRwd.requires_grad = True
        #memoryDon.requires_grad = True
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch(n_state, batch_size)
            
            for index in range(self.main_batch_size):
                obs = memoryObs[index].squeeze()
                acts = memoryAct[index]
                probs = memoryPrb[index].squeeze()
                rewards = memoryRwd[index].squeeze()
                vals = memoryVal[index].squeeze()
                done = memoryDon[index].squeeze()
                advantage = torch.zeros(self.config.internal_batch,
                                        dtype=torch.float32, device=self.device)
                for t in range(self.config.internal_batch-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, self.config.internal_batch-1):
                        a_t += discount*(rewards[k] + self.config.gamma*vals[k+1]*\
                                         (1-int(done[k])) - vals[k])                
                        discount *= self.config.gamma*self.config.gae_lambda
                    #print(a_t)
                    advantage[t] = a_t
            
                for batch in batches:
                    batchObs = obs[batch]
                    batchActs = acts[batch]
                    batchProbs = probs[batch].to(self.device)
                    batchVals = vals[batch].to(self.device)
                    
                    generatedAct = self.actor_model.generateOneStep(batchObs.to(self.device))
                    
                    act_dist = Categorical(generatedAct)
                    entropy_loss = act_dist.entropy().unsqueeze(1)
                    new_log_probs = act_dist.log_prob(batchActs.to(self.device))
                    newVal = criticModel(batchObs.to(self.device))
                    prob_ratio = new_log_probs.exp().to(self.device) / batchProbs.exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    
                    
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                                1+self.config.cliprange)*advantage[batch]
                    

                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    returns = advantage[batch].unsqueeze(dim=1) + batchVals.unsqueeze(dim=1)
                    critic_loss = (returns-newVal)**2
                    critic_loss = torch.mean(critic_loss)
                    entropy_loss = -torch.mean(entropy_loss)
                    

                    total_loss = actor_loss + 0.5*critic_loss# + 0.1*entropy_loss
                    
                    self.actor_optimizer.zero_grad()
                    #actor_loss.backward()

                    self.critic_optimizer.zero_grad()
                    total_loss.backward()
                    self.critic_optimizer.step()
                    self.actor_optimizer.step()


        self._reset_memory() if mode == 'int' else self._reset_ext_memory()
        '''for param in self.actor_model.parameters():
            param.requires_grad = False
        for param in self.critic_model.parameters():
            param.requires_grad = False'''
                    
    def external_train (
        self,
        externalObservation: torch.tensor,
        actions: torch.tensor,
        old_log_probs: torch.tensor,
        external_reward,
        statePrepare: ExternalStatePrepare,
    ):
        
        values = self.external_critic_model(externalObservation.to(self.device)) #TODO all values are same
        for _ in range(self.config.ppo_external_epochs):
            _, batches = self._generate_batch(int(actions.size(0)), self.config.ppo_external_batch_size)
            self._train_minibatch(int(actions.size(0)), externalObservation,
                                  actions, old_log_probs, values.squeeze(), 
                                  external_reward, batches, mode = "external")
            
    def load_models (self):
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
        file_path = self.savePath + "/" + self.external_critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.external_critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.external_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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
        
        file_path = self.savePath + "/" + self.external_critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.external_critic_model.state_dict(),
            'optimizer_state_dict': self.external_critic_optimizer.state_dict()}, 
            file_path)
    