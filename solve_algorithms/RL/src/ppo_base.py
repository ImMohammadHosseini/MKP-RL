
"""

"""
from configs.ppo_configs import PPOConfig
from .ppo_reply_buffer import PPOReplayBuffer
import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional
from copy import deepcopy
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
        dim: int,
        actor_model: torch.nn.Module,
        normal_critic_model: torch.nn.Module,
        extra_critic_model: torch.nn.Module,

    ):
        self.config = config
        self.save_path = save_path
        self.dim = dim
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
        steps = []; prompt = None
        step = torch.tensor([1], dtype=torch.int64)
        
        for i in range(0, self.config.generat_link_number):
            firstGenerated, secondGenerated, prompt = self.actor_model.generateOneStep(
                externalObservation, step, prompt)
            steps.append(deepcopy(step))
            act, prob = self._choose_actions(firstGenerated, secondGenerated)
            actions.append(act.unsqueeze(0)); probs.append(prob)
            internalReward, accepted_action, step, prompt = self.normal_reward(
                act, accepted_action, step, prompt, statePrepares)
            
            internalObservations.append(prompt)
            rewards.append(internalReward)
            values.append(self.normal_critic_model(externalObservation, prompt))
        steps = torch.cat(steps,0)
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
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.save_path + "/" + self.normal_critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.normal_critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.normal_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
        file_path = self.save_path + "/" + self.extra_critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.extra_critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.extra_critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def save_models (self):
        if not path.exists(self.save_path):
            makedirs(self.save_path)
        file_path = self.save_path + "/" + self.actor_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.actor_model.state_dict(),
            'optimizer_state_dict': self.actor_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.normal_critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.normal_critic_model.state_dict(),
            'optimizer_state_dict': self.normal_critic_optimizer.state_dict()}, 
            file_path)
        
        file_path = self.save_path + "/" + self.extra_critic_model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.extra_critic_model.state_dict(),
            'optimizer_state_dict': self.extra_critic_optimizer.state_dict()}, 
            file_path)
     