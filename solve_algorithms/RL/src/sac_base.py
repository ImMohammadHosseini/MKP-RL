

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
        
        
        self.target_entropy = 0.98 * -np.log(1 / self.actor_model.config.inst_obs_size) * \
            -np.log(1 / self.actor_model.config.knapsack_obs_size)
        self.log_alpha = torch.tensor(np.log(self.config.alpha_initial), requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
    
        
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None
        
        self.memory = SACReplayBuffer(self.config.generat_link_number)
        
        
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
    
    def train (
        *args
    ):
        raise NotImplementedError("Not implemented")
        
    def load_models (self):
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_local1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local1.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer1.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_local2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_local2.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer2.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_target1.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target1.load_state_dict(checkpoint['model_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_target2.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_target2.load_state_dict(checkpoint['model_state_dict'])
        
    def save_models (self):
        if not path.exists(self.savePath):
            makedirs(self.savePath)
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.critic_local1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local1.state_dict(),
            'optimizer_state_dict': self.critic_optimizer1.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.critic_local2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_local2.state_dict(),
            'optimizer_state_dict': self.critic_optimizer2.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.critic_target1.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target1.state_dict()}, 
            file_path)
        
        file_path = self.savePath + "/" + self.critic_target2.name + ".ckpt"
        torch.save({
            'model_state_dict': self.critic_target2.state_dict()}, 
            file_path)