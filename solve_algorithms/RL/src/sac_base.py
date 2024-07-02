

from configs.sac_configs import SACConfig
from .sac_reply_buffer import SACReplayBuffer
import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional
from copy import deepcopy
from os import path, makedirs


class SACBase():
    def __init__(
        self,
        config: SACConfig,
        save_path: str,
        dim: int,
        actor_model: torch.nn.Module,
        critic_local1: torch.nn.Module,
        critic_local2: torch.nn.Module,
        critic_target1: torch.nn.Module,
        critic_target2: torch.nn.Module,
        action_num:int,
    ):
        self.config = config
        self.save_path = save_path
        self.dim = dim
        self.actor_model = actor_model
        self.critic_local1 = critic_local1
        self.critic_local2 = critic_local2
        self.critic_target1 = critic_target1
        self.critic_target2 = critic_target2
        
        self.actor_optimizer = Adam(self.actor_model.parameters(), 
                                    lr=self.config.actor_lr)
        self.critic_optimizer1 = Adam(self.critic_local1.parameters(),
                                      lr=self.config.critic_lr)
        self.critic_optimizer2 = Adam(self.critic_local2.parameters(), 
                                      lr=self.config.critic_lr)
        
        self.soft_update(1.)
        
        self.log_alpha = torch.tensor(np.log(self.config.alpha_initial), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
        
        self.memory = SACReplayBuffer(self.config.generat_link_number, 
                                      self.config, self.actor_model.config, action_num)
        
        
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
        actions = []; internalObservations = []; rewards = []; #probs = []; values = []; 
        steps = []; prompt = None
        step = torch.tensor([1], dtype=torch.int64)
        
        for i in range(0, self.config.generat_link_number):
            firstGenerated, secondGenerated, prompt = self.actor_model.generateOneStep(
                externalObservation, step, prompt)
            steps.append(deepcopy(step))
            act, _, _ = self._choose_actions(firstGenerated, secondGenerated)
            #print(act)
            #print(prob)
            actions.append(act.unsqueeze(0))#; probs.append(prob)
            #print(actions)
            #print(probs)
            internalReward, accepted_action, step, prompt = self.normal_reward(
                act, accepted_action, step, prompt, statePrepares)
            
            internalObservations.append(prompt)
            rewards.append(internalReward)
            #values.append(self.normal_critic_model(externalObservation, prompt))
        steps = torch.cat(steps,0)
        if not fraction_flag: 
            rewards = [sum(rewards)]
            #probs = [sum(probs)]
            
        if step_flag:
            return internalObservations, actions, accepted_action, rewards, steps
        else:
            return None, actions, accepted_action, rewards, None
    
        
    def train (
        *args
    ):
        raise NotImplementedError("Not implemented")
    
    def soft_update(self, tau):
        for target_param, local_param in zip(self.critic_target1.parameters(), self.critic_local1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        
        for target_param, local_param in zip(self.critic_target2.parameters(), self.critic_local2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        
    def load_models (self):
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_local1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local1.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_local2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local2.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_target1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target1.load_state_dict(checkpoint['model_state_dict'])
        
        file_path = self.save_path + "/" + self.critic_target2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target2.load_state_dict(checkpoint['model_state_dict'])
        
    def save_models (self):
        if not path.exists(self.save_path):
            makedirs(self.save_path)
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_local1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local1.state_dict(),
            'optimizer_state_dict': self.critic_optimizer1.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_local2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local2.state_dict(),
            'optimizer_state_dict': self.critic_optimizer2.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_target1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target1.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.critic_target2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target2.state_dict()}, 
            file_path)