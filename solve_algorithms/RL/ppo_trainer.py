
"""

"""
import torch
import numpy as np
from torch.optim import AdamW
from typing import List, Optional
import time


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
from copy import deepcopy

from torch.autograd import Variable
from .src.base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig
from os import path, makedirs
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

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
        ref_model: Optional[torch.nn.Module] = None,
        critic_model: Optional[torch.nn.Module] = None,
        external_critic_model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        ref_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        external_critic_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        super().__init__(config)
        
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        
        
        self.device = device
        self.softmax = torch.nn.Softmax()
        self.info = info
        self.main_batch_size = main_batch_size
        
        self.actor_model = actor_model
        self.ref_model = ref_model
        self.critic_model = critic_model
        self.critic_model = self.critic_model.to(self.device)
        self.external_critic_model = external_critic_model
        self.external_critic_model = self.external_critic_model.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.actor_model.parameters()), lr=self.config.actor_lr
            )
        else:
            self.actor_optimizer = actor_optimizer
        
        if ref_optimizer is None:
            self.ref_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.ref_model.parameters()), lr=self.config.actor_lr
            )
        else:
            self.ref_optimizer = ref_optimizer
            
        if critic_optimizer is None:
            self.critic_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.critic_model.parameters()), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer = critic_optimizer
            
        if external_critic_optimizer is None:
            self.external_critic_optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.external_critic_model.parameters()), lr=self.config.critic_lr
            )
        else:
            self.external_critic_model = external_critic_model
            
        self.savePath = save_path +'/RL/PPOTrainer/'
        if path.exists(self.savePath):
            self.load_models()
            
        self.batch_actor_loss = torch.tensor(0., device=self.device)
        self.batch_critic_loss = torch.tensor(0., device=self.device)
    
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
    
    def internal_reward (
        self,
        actions: torch.tensor,
        probs: torch.tensor,
        accepted_actions: np.ndarray,
        accepted_probs: np.ndarray,
        prompt: torch.tensor,
        internalObservation: torch.tensor,
        statePrepares: np.ndarray,
    ):
        rewards = []
        for index, act in enumerate(actions):
            inst_act = int(act[0])
            ks_act = int(act[1])
            knapSack = statePrepares[index].getKnapsack(ks_act)
            ks = torch.tensor(statePrepares[index].getObservedKS(ks_act),
                              device=self.device, dtype=torch.float32).unsqueeze(0)
            inst = torch.tensor(statePrepares[index].getObservedInst(inst_act), 
                                device=self.device, dtype=torch.float32).unsqueeze(0)
            
            internalObservation[index] = torch.cat([internalObservation[index][1:], 
                                                    torch.cat([inst, ks], 1)], 0)
            #internalObservation[index] = torch.cat([internalObservation[index][1:], ks], 0)

            eCap = knapSack.getExpectedCap()
            weight = statePrepares[index].getObservedInstWeight(inst_act)
            if inst_act < statePrepares[index].pad_len or inst_act in accepted_actions[index,:,0]: 
                    rewards.append(-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                prompt[index] = torch.cat([prompt[index][1:], 
                                           torch.cat([inst, ks], 1)], 0)
                #prompt[index] = torch.cat([prompt[index][1:], ks], 0)
                
                value = statePrepares[index].getObservedInstValue(inst_act)
                rewards.append(value / np.sum(weight))
                knapSack.removeExpectedCap(weight)
                accepted_actions[index] = np.append(accepted_actions[index][1:],
                                                    [[inst_act, ks_act]], 0)
                accepted_probs[index] = np.append(accepted_probs[index][1:], float(probs[index]))
            else:
                rewards.append(-self.info['VALUE_LOW'])
        return torch.tensor(rewards, device=self.device).unsqueeze(1)
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
        done: Optional[torch.tensor] = None,
    ):
        actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int, device=self.device)
        probs = torch.tensor([], dtype=torch.float64, device=self.device)
        internalRewards = torch.tensor([], dtype=torch.float64, device=self.device)
        internalObservations = torch.zeros([self.main_batch_size, 0, 
                                            self.config.generat_link_number, 
                                            12],#self.actor_model.config.output_dim], 
                                           device=self.device)
        internalObservation = None
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        accepted_probs = np.array([[None]*self.config.generat_link_number]*self.main_batch_size)
        
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                externalObservation, self.config.generat_link_number, prompt)
            act, prob = self._choose_actions(generatedInstance, generatedKnapsack)
            if internalObservation == None: internalObservation = deepcopy(prompt)
            internalReward = self.internal_reward(act, prob, accepted_actions, 
                                                  accepted_probs , prompt, 
                                                  internalObservation,
                                                  statePrepares)
            
            internalObservations = torch.cat([internalObservations, internalObservation.unsqueeze(1)], 1)
            actions = torch.cat([actions, act.unsqueeze(1)], 1)
            probs = torch.cat([probs, prob], 1)
            internalRewards = torch.cat([internalRewards, internalReward], 1)
            if i % self.config.internal_batch == self.config.internal_batch-1:
                #print(internalRewards)
                for index in range(self.main_batch_size):
                    eo = torch.cat([externalObservation[index].unsqueeze(0)]*(self.config.internal_batch), 0)
                    io = internalObservations[index]
                    values = self.critic_model(eo.to(self.device), 
                                               io.to(self.device)).squeeze()
                 
                    for _ in range(self.config.ppo_epochs):
                        dones, batches = self.generate_batch(done=done)
                        self._train_minibatch(self.config.internal_batch, eo, 
                                              actions[index], probs[index], 
                                              values, internalRewards[index], 
                                              batches, io, dones)

                self._update_models()
                actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int, 
                                      device=self.device)
                probs = torch.tensor([], dtype=torch.float64, device=self.device)
                internalRewards = torch.tensor([], dtype=torch.float64, 
                                               device=self.device)
                internalObservation = None
                internalObservations = torch.zeros([self.main_batch_size, 0, 
                                                    self.config.generat_link_number, 
                                                    12],#self.actor_model.config.output_dim], 
                                                   device=self.device)
        return accepted_actions, accepted_probs
    
    
            
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
        done: Optional[torch.tensor] = None,
    ):
        if batch_size == None:
            batch_size = self.config.ppo_batch_size
        if n_states == None:
            n_states = self.config.internal_batch
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        if done is None:
            dones = torch.tensor([False]*n_states)
        else:
            dones = torch.tensor([done]*n_states).squeeze()
        
        return dones,\
                batches
    
    def _train_minibatch (
        self, 
        n_states: int,
        external_obs: torch.tensor,
        actions: torch.tensor,
        old_log_probs: torch.tensor,
        vals: torch.tensor,
        rewards: torch.tensor,
        batches: List,
        internal_obs: Optional[torch.tensor]=None,
        dones: Optional[torch.tensor]=None,
        mode = "internal",
    ):
        advantage = torch.zeros(n_states, dtype=torch.float32, device=self.device)
        for t in range(n_states-1):
            discount = 1
            a_t = 0
            for k in range(t, n_states-1):
                a_t += discount*(int(rewards[k]) + self.config.gamma*(float(vals[k+1]))- (float(vals[k]))) # -int(dones[k])
                discount *= self.config.gamma*self.config.gae_lambda
            #print('advantage %.3f' % a_t, 'vals %.3f' % float(vals[t]))
            advantage[t] = a_t

        for batch in batches:
            eo = external_obs[batch]
            if mode == "internal":
                io = internal_obs[batch]
            else: io = None
            olp = old_log_probs[batch]
            acts = actions[batch]
            
            generatedInstance, generatedKnapsack, _ = self.actor_model.generateOneStep(
                eo, self.config.generat_link_number, io)
            
            new_inst_dist = Categorical(generatedInstance)
            new_ks_dist = Categorical(generatedKnapsack)
            
            if mode == "internal":
                critic_value = vals[batch]
                #self.critic_model(eo.to(self.device), io.to(self.device))
                new_log_probs = new_inst_dist.log_prob(acts[:,0]) + new_ks_dist.log_prob(acts[:,1])

            else:
                critic_value = self.external_critic_model(eo.to(self.device))
                new_log_probs = new_inst_dist.log_prob(acts[:,0]) + new_ks_dist.log_prob(acts[:,1])

            critic_value = torch.squeeze(critic_value)
            prob_ratio = (new_log_probs.exp() / olp.exp()).to(self.device)
            
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                        1+self.config.cliprange)*advantage[batch]
            
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch].unsqueeze(dim=1) + vals[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            
            self.batch_actor_loss += actor_loss.data
            self.batch_critic_loss += critic_loss.data
            
            
    def _update_models (
        self,
    ):
        actor_loss_leaf = Variable(self.batch_actor_loss.data/self.main_batch_size, 
                                   requires_grad=True)
        critic_loss_leaf = Variable(self.batch_critic_loss.data/self.main_batch_size,
                                    requires_grad=True)
        #print('al', float(actor_loss_leaf), '\ncl',float(critic_loss_leaf))
        #total_loss = actor_loss_leaf + 0.5*critic_loss_leaf
        #total_loss = Variable(total_loss.data, requires_grad=True)
        
        #self.actor_model.eval()
        self.actor_optimizer.zero_grad()
        self.ref_optimizer.zero_grad()
        self.critic_optimizer.zero_grad() 
        #if mode == "internal" else \
        #    self.external_critic_optimizer.zero_grad()
        #total_loss.backward()
        actor_loss_leaf.backward()
        self.actor_optimizer.step()
        critic_loss_leaf.backward(retain_graph=True)
        self.ref_optimizer.step()
        self.critic_optimizer.step() 
        #if mode == "internal" else \
        #    self.external_critic_optimizer.step()
        self.batch_actor_loss = torch.tensor(0., device=self.device)
        self.batch_critic_loss = torch.tensor(0., device=self.device)
    
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
    