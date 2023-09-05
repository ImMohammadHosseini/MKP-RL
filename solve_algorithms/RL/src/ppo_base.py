
"""

"""
from configs.ppo_configs import PPOConfig
from .ppo_reply_buffer import PPOReplayBuffer
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
        config: PPOConfig,
        save_path: str,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,

    ):
        self.config = config
        self.save_path = save_path
        self.actor_model = actor_model
        self.normal_critic_model = normal_critic_model
        self.extra_critic_model = extra_critic_model
        
        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                    lr=self.config.actor_lr)
        self.normal_critic_optimizer = Adam(self.normal_critic_model.parameters(), 
                                             lr=self.config.critic_lr)
        self.extra_critic_optimizer = Adam(self.extra_critic_model.parameters(), 
                                             lr=self.config.critic_lr)
        
        self.memory = PPOReplayBuffer(self.config.generat_link_number)
        
    def _choose_actions (
        self, 
        *args
    ):
        raise NotImplementedError("Not implemented")
        
    def normal_reward (
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
    ):
        accepted_action = np.array([[-1]*2]*self.config.generat_link_number, dtype= int)
        actions = []; probs = []; values = []; rewards = []; internalObservations = []
        steps = []; step = 0; prompt = None
        for i in range(0, self.config.generat_link_number):
            firstGenerated, secondGenerated, prompt = self.actor_model.generateOneStep(
                externalObservation, step, prompt)
            act, prob = self._choose_actions(firstGenerated, secondGenerated)
            print('fffff', prob.size())
            actions.append(act.unsqueeze(0)); probs.append(prob)
            internalReward = self.normal_reward(act, accepted_action, step, 
                                                  prompt, statePrepares)
            steps.append(step); internalObservations.append(prompt); 
            rewards.append(internalReward)
            values.append(self.normal_critic_model(externalObservation, prompt))
            
        if not fraction_flag: 
            rewards = [sum(rewards)]
            probs = [sum(probs)]
            
        if step_flag:
            return internalObservations, actions, accepted_action, probs, values, rewards, steps
        else:
            return None, actions, accepted_action, probs, values, rewards, None
    
    def train (
        *args
    ):
        raise NotImplementedError("Not implemented")
    
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
    
    def load_models (self):
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.normal_critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.normal_critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.normal_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
        file_path = self.savePath + "/" + self.extra_critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.extra_critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.extra_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_models (self):
        if not path.exists(self.savePath):
            makedirs(self.savePath)
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.normal_critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.normal_critic_model.state_dict(),
            'optimizer_state_dict': self.normal_critic_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.extra_critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.extra_critic_model.state_dict(),
            'optimizer_state_dict': self.extra_critic_optimizer.state_dict()}, 
            file_path)
     
        
        
        
        
        
        '''
        if mode == 'noraml':
        memoryObs, memoryIntObs, memoryAct, memoryPrb, memoryVal, memoryRwd, \
            memoryStp, memoryDon = self.replayBuffer.
        elif mode == 'extra':
            pass

            
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
                
                
                firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
                    batchObs, batchSteps, batchIntObs)
               
                #batchActs = batchActs.to(self.device)
                #entropy_loss = torch.zeros((0,1), dtype=torch.int)
                inst_log_probs = torch.zeros((0,1), dtype=torch.int)
                ks_log_probs = torch.zeros((0,1), dtype=torch.int)

                for i, generat in enumerate(zip(generatedInstance, generatedKnapsack)):
                    inst_dist = Categorical(generat[0])
                    ks_dist = Categorical(generat[1])
                    entropy_loss = torch.cat([entropy_loss, (inst_dist.entropy()+ks_dist.entropy()).unsqueeze(0)], 0)
                    inst_log_probs = torch.cat([inst_log_probs, inst_dist.log_prob(
                        batchActs[i,0]).unsqueeze(0)], 0)
                    ks_log_probs = torch.cat([ks_log_probs, ks_dist.log_prob(
                        batchActs[i,1]).unsqueeze(0)], 0)
                    
                    
                for i in range(0, self.config.generat_link_number):
                    firstGenerated, secondGenerated, prompt = self.actor_model.generateOneStep(
                        batchObs, batchSteps, batchIntObs)
                    _, new_log_probs = self._choose_actions(firstGenerated, secondGenerated)

                
                #act_dist = Categorical(generatedAct)
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
                

                total_loss = actor_loss + 0.5*critic_loss
                
                self.actor_optimizer.zero_grad()
                #actor_loss.backward()

                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.critic_optimizer.step()
                self.actor_optimizer.step()


        self._reset_memory() if mode == 'int' else self._reset_ext_memory()'''