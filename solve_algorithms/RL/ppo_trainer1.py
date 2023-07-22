
"""

"""
import torch
import numpy as np
from torch.optim import AdamW, SGD, RMSprop
from typing import List, Optional
#import time


from .src.core import (
    set_seed, 
    masked_mean,
    masked_var,
    masked_whiten,
    clip_by_value,
    entropy_from_logits,
    flatten_dict,
)
from torch.distributions.categorical import Categorical
from src.data_structure.state_prepare import ExternalStatePrepare
#from copy import deepcopy

from torch.autograd import Variable
from .src.base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig
from os import path, makedirs
#from torchrl.objectives import ClipPPOLoss
#from torchrl.objectives.value import GAE

class PPOTrainer(BaseTrainer):
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
            self.actor_optimizer = AdamW(
                self.actor_model.parameters(), lr=self.config.actor_lr
            )#filter(lambda p: p.requires_grad, 
        else:
            self.actor_optimizer = actor_optimizer
            
        if critic_optimizer is None:
            self.critic_optimizer = AdamW(
                self.critic_model.parameters(), lr=self.config.critic_lr
            )#lambda p: p.requires_grad, 
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
        self.memory_action = [] 
        self.memory_prob = [] 
        self.memory_val = []
        self.memory_reward = []
        self.memory_done = []
        
    def save_step (
        self, 
        observation: torch.tensor, 
        actions: np.ndarray, 
        sumProbs: torch.tensor,
        sumVals: torch.tensor, 
        sumRewards: torch.tensor,
        done: bool, 
    ):
        
        self.memory_observation.append(observation.cpu().unsqueeze(1))
        self.memory_action.append(actions.data.cpu().unsqueeze(1))
        #sumProbs = sumProbs.data
        self.memory_prob.append(sumProbs.data.cpu().unsqueeze(1))
        self.memory_val.append(sumVals.data.cpu().unsqueeze(1))
        self.memory_reward.append(sumRewards.cpu().unsqueeze(1))
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
            
            #prompt[index] = torch.cat([prompt[index][1:],inst], 0)
            #prompt[index] = torch.cat([prompt[index][1:],ks], 0)

            if inst_act < statePrepares[index].pad_len or inst_act in accepted_actions[index,:,0]: 
                    rewards.append(-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                prompt[index] = torch.cat([prompt[index][1:], 
                                           torch.cat([inst, ks], 1)], 0)
                #print(prompt[index])
                #prompt[index] = torch.cat([prompt[index][1:], ks], 0)
                
                value = statePrepares[index].getObservedInstValue(inst_act)
                rewards.append(+(value / np.sum(weight)))
                knapSack.removeExpectedCap(weight)
                #print('ks', ks_act, 'inst', inst_act)
                accepted_actions[index] = np.append(accepted_actions[index][1:],
                                                    [[inst_act, ks_act]], 0)
            else:
                rewards.append(-self.info['VALUE_LOW'])
        return torch.tensor(rewards, device=self.device)
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
        done: Optional[torch.tensor] = None,
    ):
        actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int, device=self.device)
        sumProbs = torch.tensor([0]*self.main_batch_size, dtype=torch.float64, 
                                 device=self.device)
        sumRewards = torch.tensor([0]*self.main_batch_size, dtype=torch.float64, 
                                   device=self.device)
        
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        
        for step in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                step, externalObservation, prompt)
            act, prob = self._choose_actions(generatedInstance, generatedKnapsack)
            actions = torch.cat([actions, act.unsqueeze(1)], 1)
            reward = self.reward(act, accepted_actions, prompt, statePrepares)
            sumProbs += prob.squeeze()
            sumRewards += reward
            
            
        values = self.critic_model(externalObservation).squeeze(1)
        return actions, accepted_actions, sumProbs, values, sumRewards
        
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
        
    def generate_batch (
        self, 
        n_states: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if batch_size == None:
            batch_size = self.config.ppo_batch_size
        if n_states == None:
            n_states = self.config.internal_batch
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        return batches
    
    def train_minibatch (
        self, 
    ):
        #print('h')
        memoryObs = torch.cat(self.memory_observation, 1)
        memoryAct = torch.cat(self.memory_action, 1) 
        memoryPrb = torch.cat(self.memory_prob, 1)
        memoryVal = torch.cat(self.memory_val, 1)
        memoryRwd = torch.cat(self.memory_reward, 1)
        memoryDon = torch.cat(self.memory_done, 1)
        #print(memoryObs.size()) 
        #print(memoryAct.size()) 
        #print(memoryPrb.size())
        #print(memoryVal.size())
        #print(memoryRwd.size())
        #print(memoryDon.size())
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch()
            
            batch_actor_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_critic_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
            count = 0
            
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
                    #batchRewards = rewards[batch]
                    batchVals = vals[batch].to(self.device)
                    #batchDone = done[batch]
                    
                    
                    prompt = None
                    new_log_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                                 dtype=torch.float64, device=self.device)
                    
                    for step in range(0, self.config.generat_link_number):
                        generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                            step, batchObs.to(self.device), prompt)
                        inst_dist = Categorical(generatedInstance.squeeze())
                        ks_dist = Categorical(generatedKnapsack.squeeze())
                        batchActs = batchActs.to(self.device)
                        
                        #print('a',batchActs[:,i,0].size())
                        inst_log_probs = inst_dist.log_prob(batchActs[:,step,0])#.unsqueeze(1))
                        ks_log_probs = ks_dist.log_prob(batchActs[:,step,1])#.unsqueeze(1))
                        newProbs = inst_log_probs + ks_log_probs
                        new_log_probs += newProbs.squeeze()
                    
                    newVal = self.critic_model(batchObs.to(self.device)).squeeze()

                    prob_ratio = (new_log_probs - batchProbs).exp().to(self.device)
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                                1+self.config.cliprange)*advantage[batch]
                    #print(weighted_clipped_probs.size())
                    #print(weighted_probs.size())

                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    returns = advantage[batch].unsqueeze(dim=1) + batchVals.unsqueeze(dim=1)
                    #print('sizeeeeeee', returns.size())
                    #print('returns', returns)
                    #print('new_val', newVal)
                    critic_loss = (returns-newVal)**2
                    critic_loss = critic_loss.mean()
                    
                    count += 1
                    batch_actor_loss += actor_loss.data
                    batch_critic_loss += critic_loss.data
                    
            actor_loss_leaf = Variable(batch_actor_loss.data/count, requires_grad=True)
            critic_loss_leaf = Variable(batch_critic_loss.data/count, requires_grad=True)
            
            total_loss = actor_loss_leaf + 0.5*critic_loss_leaf
            #print(actor_loss_leaf)
            #print(critic_loss_leaf)
            self.actor_optimizer.zero_grad()
            #self.critic_optimizer.zero_grad()

            #total_loss.backward()
            actor_loss_leaf.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss_leaf.backward()#retain_graph=True
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
         
        #file_path = self.savePath + "/" + self.external_critic_model.name + ".ckpt"
        #checkpoint = torch.load(file_path)
        #self.external_critic_model.load_state_dict(checkpoint['model_state_dict'])
        #self.external_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
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
        
        #file_path = self.savePath + "/" + self.external_critic_model.name + ".ckpt"
        #torch.save({
        #    'model_state_dict': self.external_critic_model.state_dict(),
        #    'optimizer_state_dict': self.external_critic_optimizer.state_dict()}, 
        #    file_path)
    