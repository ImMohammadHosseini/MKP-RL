
"""

"""
import torch
import numpy as np
from torch.optim import AdamW
from typing import List, Optional
import time

from torch.distributions.categorical import Categorical
from src.data_structure.state_prepare import ExternalStatePrepare
from copy import deepcopy

from torch.autograd import Variable
from .src.base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig
from os import path, makedirs


class FractionPPOTrainer(BaseTrainer):
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
        external_critic_model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
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
            
        self.savePath = save_path +'/RL/FractionPPOTrainer/'
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
        self.memory_internalReward = []
        self.memory_steps = []
        self.memory_done = []
        
    def save_step (
        self, 
        externalObservation: torch.tensor,
        internalObservations: torch.tensor,
        actions: torch.tensor,
        probs: torch.tensor,
        values: torch.tensor, 
        internalRewards: torch.tensor,
        steps: torch.tensor,
        done: bool,
    ):
        self.memory_observation.append(torch.cat([externalObservation.unsqueeze(1)]*10, 1).cpu())
        self.memory_internalObservation.append(internalObservations.cpu())
        self.memory_action.append(actions.data.cpu())
        self.memory_prob.append(probs.data.cpu())
        self.memory_val.append(values.data.cpu())
        self.memory_internalReward.append(internalRewards.cpu())
        self.memory_steps.append(steps.cpu())
        
        if done == True:
            self.memory_done.append(torch.tensor([[False]*(self.config.generat_link_number-1)+[True]]*self.main_batch_size))
        else:
            self.memory_done.append(torch.tensor([[done]*self.config.generat_link_number]*self.main_batch_size))
        
    def _choose_actions (
        self, 
        generatedInstance, 
        generatedKnapsack, 
    ):
        acts = torch.zeros((0,2), dtype=torch.int)
        log_probs = torch.zeros((0,1), dtype=torch.int)
        for gi, gk in zip(generatedInstance, generatedKnapsack):
            inst_dist = Categorical(gi)
            ks_dist = Categorical(gk)
            inst_act = inst_dist.sample() 
            ks_act = ks_dist.sample()
        
            inst_log_probs = inst_dist.log_prob(inst_act)
            ks_log_probs = ks_dist.log_prob(ks_act)
            
            acts = torch.cat([acts, torch.cat([inst_act, ks_act], dim=1)], dim=0)
            log_probs = torch.cat([log_probs, inst_log_probs+ks_log_probs], dim=0)

        return acts, log_probs

    def internal_reward (
        self,
        actions: torch.tensor,
        accepted_actions: np.ndarray,
        step: torch.tensor,
        prompt: torch.tensor,
        statePrepares: np.ndarray,
    ):
        rewards = []
        for index, act in enumerate(actions):
            inst_act = int(act[0]) + statePrepares[index].pad_len
            ks_act = statePrepares[index].getRealKsAct(int(act[1]))
            knapSack = statePrepares[index].getKnapsack(ks_act)
            ks = torch.tensor(statePrepares[index].getObservedKS(ks_act),
                              device=self.device, dtype=torch.float32).unsqueeze(0)
            inst = torch.tensor(statePrepares[index].getObservedInst(inst_act), 
                                device=self.device, dtype=torch.float32).unsqueeze(0)
            
            
            eCap = knapSack.getExpectedCap()
            weight = statePrepares[index].getObservedInstWeight(inst_act)
            
            if inst_act in accepted_actions[index,:,0]: # inst_act < statePrepares[index].pad_len or
                    rewards.append(0)#-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
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
        probs = torch.zeros((self.main_batch_size,0), dtype=torch.float64, device=self.device)
        values = torch.zeros((self.main_batch_size,0), dtype=torch.float64, device=self.device)
        internalRewards = torch.zeros((self.main_batch_size,0), dtype=torch.float64, device=self.device)
        internalObservations = torch.zeros([self.main_batch_size, 0, 
                                            self.config.generat_link_number+1, 
                                            self.actor_model.config.input_decode_dim],
                                           device=self.device)
        #internalObservation = None
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        #accepted_probs = np.array([[None]*self.config.generat_link_number]*self.main_batch_size)
        step = torch.tensor([0]*self.main_batch_size, dtype=torch.int64, device=self.device)
        steps = torch.zeros((self.main_batch_size,0), dtype=torch.int64, device=self.device)
        
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                step, externalObservation, prompt)
            act, prob = self._choose_actions(generatedInstance, generatedKnapsack)
            steps = torch.cat([steps, step.unsqueeze(1)], 1)
            internalObservations = torch.cat([internalObservations, prompt.unsqueeze(1)], 1)
            value = self.critic_model(externalObservation, prompt)
            values = torch.cat([values, value],1)
            internalReward = self.internal_reward(act, accepted_actions, step, 
                                                  prompt, statePrepares)
            actions = torch.cat([actions, act.unsqueeze(1).to(self.device)], 1)
            probs = torch.cat([probs, prob.to(self.device)], 1)
           
            internalRewards = torch.cat([internalRewards, internalReward.unsqueeze(1)], 1)
            
        return internalObservations, actions, accepted_actions, probs, values, internalRewards, steps
    
    def train_minibatch (
        self, 
    ):
        memoryObs = torch.cat(self.memory_observation, 1)
        memoryIntObs = torch.cat(self.memory_internalObservation, 1)
        memoryAct = torch.cat(self.memory_action, 1) 
        memoryPrb = torch.cat(self.memory_prob, 1)
        memoryVal = torch.cat(self.memory_val, 1)
        memoryRwd = torch.cat(self.memory_internalReward, 1)
        memoryStp = torch.cat(self.memory_steps, 1)
        memoryDon = torch.cat(self.memory_done, 1)
        
        
        for _ in range(self.config.ppo_epochs):
            batches = self.generate_batch()
            
            #batch_actor_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
            #batch_critic_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
            #batch_entropy_loss = torch.tensor(0, device=self.device, dtype=torch.float32)
            #count = 0
            
            for index in range(self.main_batch_size):
                obs = memoryObs[index].squeeze()
                intObs = memoryIntObs[index]
                acts = memoryAct[index]
                probs = memoryPrb[index].squeeze()
                rewards = memoryRwd[index].squeeze()
                vals = memoryVal[index].squeeze()
                stps = memoryStp[index]
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
                    batchIntObs = intObs[batch]
                    batchActs = acts[batch]
                    batchSteps = stps[batch]
                    batchProbs = probs[batch].to(self.device)
                    batchVals = vals[batch].to(self.device)
                    
                    generatedInstance, generatedKnapsack, _ = self.actor_model.generateOneStep(
                        batchSteps, batchObs.to(self.device), batchIntObs)
                   
                    #batchActs = batchActs.to(self.device)
                    entropy_loss = torch.zeros((0,1), dtype=torch.int)
                    inst_log_probs = torch.zeros((0,1), dtype=torch.int)
                    ks_log_probs = torch.zeros((0,1), dtype=torch.int)

                    for i, generat in enumerate(zip(generatedInstance, generatedKnapsack)):
                        inst_dist = Categorical(generat[0])
                        ks_dist = Categorical(generat[1])
                        entropy_loss = torch.cat([entropy_loss, inst_dist.entropy()+ks_dist.entropy()], 0)
                        inst_log_probs = torch.cat([inst_log_probs, inst_dist.log_prob(
                            batchActs[i,0])], 0)
                        ks_log_probs = torch.cat([ks_log_probs, ks_dist.log_prob(
                            batchActs[i,1])], 0)

                    new_log_probs = inst_log_probs + ks_log_probs
                    
                    newVal = self.critic_model(batchObs.to(self.device), 
                                               batchIntObs.to(self.device))
                    
                    prob_ratio = (new_log_probs.to(self.device) - batchProbs).exp()
                    weighted_probs = advantage[batch] * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                                1+self.config.cliprange)*advantage[batch]
                    

                    actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                    returns = advantage[batch].unsqueeze(dim=1) + batchVals.unsqueeze(dim=1)
                    critic_loss = (returns-newVal)**2
                    critic_loss = torch.mean(critic_loss)
                    entropy_loss = -torch.mean(entropy_loss)

                    #count = 1
                    #batch_actor_loss += actor_loss.data
                    #batch_critic_loss += critic_loss.data
                    #batch_entropy_loss += torch.mean(entropy_loss)
                    actor_loss_leaf = Variable(actor_loss.data, requires_grad=True)
                    critic_loss_leaf = Variable(critic_loss.data, requires_grad=True)
                    entropy_loss_leaf = Variable(entropy_loss.data, requires_grad=True)

                    total_loss = actor_loss_leaf + 0.5*critic_loss_leaf + 0.1*entropy_loss_leaf
                    #print(actor_loss_leaf)
                    #print(critic_loss_leaf)
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()

                    #self.critic_optimizer.zero_grad()

                    total_loss.backward()
                    #actor_loss_leaf.backward()
                    self.actor_optimizer.step()
                    #self.critic_optimizer.zero_grad()
                    #critic_loss_leaf.backward()#retain_graph=True
                    self.critic_optimizer.step()
                    
        self._reset_memory()
                    
                    
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
    