
"""

"""
import numpy as np
import torch
from os import path, makedirs
from typing import List, Optional
from torch.optim import AdamW, SGD, RMSprop
from torch.distributions.categorical import Categorical

from .src.base_trainer import BaseTrainer
from configs.sac_configs import SACConfig


class PPOTrainer(BaseTrainer):
    r"""
    this implementation is inspired by: 
        https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/SAC
        https://github.com/rlworkgroup/garage/tree/master
        
    """
    
    def __init__(
        self,
        info: List,
        save_path: str,
        main_batch_size: int,
        config: SACConfig,
        actor_model: torch.nn.Module,
        critic_model: torch.nn.Module,
        value_model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        value_optimizer: Optional[torch.optim.Optimizer] = None,

    ):
        self.device = device
        self.info = info
        self.main_batch_size = main_batch_size
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_model = self.critic_model.to(self.device)
        self.value_model = value_model
        self.value_model = self.value_model.to(self.device)
        
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
            
        if value_optimizer is None:
            self.value_optimizer = AdamW(
                self.value_model.parameters(), lr=self.config.value_lr
            )#lambda p: p.requires_grad, 
        else:
            self.critic_optimizer = critic_optimizer
            
        self.savePath = save_path +'/RL/SACTrainer/'
        if path.exists(self.savePath):
            self.load_models()
        
        self._transitions_stored = 0
        self.memory_observation = np.zeros((self.config.buffer_size, *input_shape))#TODO 
        self.memory_actions = np.zeros((self.config.buffer_size, n_actions)) 
        self.memory_reward = np.zeros(self.config.buffer_size)
        self.memory_new_observation = np.zeros((self.config.buffer_size, *input_shape))
        self.memory_done = np.zeros(self.config.buffer_size, dtype=np.bool)
    
    def save_step (
        self, 
        observation: torch.tensor, 
        actions: np.ndarray, 
        sumRewards: torch.tensor,
        new_observation: torch.tensor, 
        done: bool, 
    ):
        new_placement = self._transitions_stored % self.config.buffer_size
        
        self.memory_observation[new_placement] = observation
        self.memory_new_observation[new_placement] = new_observation
        self.memory_actions[new_placement] = actions
        self.memory_reward[new_placement] = sumRewards
        self.memory_done[new_placement] = done

        self._transitions_stored += 1
    
    def sample_step (
        self, 
    ):
        max_mem = min(self._transitions_stored, self.config.buffer_size)

        batch = np.random.choice(max_mem, self.config.sac_batch_size)

        observation = self.memory_observation[batch]
        observation_ = self.memory_new_observation[batch]
        actions = self.memory_actions[batch]
        sumRewards = self.memory_reward[batch]
        dones = self.memory_done[batch]

        return observation, actions, sumRewards, observation_, dones
        
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
        
        sumRewards = torch.tensor([0]*self.main_batch_size, dtype=torch.float64, 
                                   device=self.device)
        
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.internal_batch]*self.main_batch_size, dtype= int)
        
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                externalObservation, self.config.generat_link_number, prompt)
            act, _ = self._choose_actions(generatedInstance, generatedKnapsack)
            actions = torch.cat([actions, act.unsqueeze(1)], 1)
            reward = self.reward(act, accepted_actions, prompt, statePrepares)
            sumRewards += reward
            
        return actions, accepted_actions, sumRewards
    
    def train (
        self, 
    ):
        if self._transitions_stored < self.config.sac_batch_size :
            return
        
        observation, actions, sumRewards, observation_, dones = \
                self.sample_step()
            
        
    