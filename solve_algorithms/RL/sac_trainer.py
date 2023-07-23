
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


class SACTrainer(BaseTrainer):
    r"""
    this implementation is inspired by: 
        https://towardsdatascience.com/adapting-soft-actor-critic-for-discrete-action-spaces-a20614d4a50a
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
        critic_local1: torch.nn.Module,
        critic_local2: torch.nn.Module,
        critic_target1: torch.nn.Module,
        critic_target2: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer1: Optional[torch.optim.Optimizer] = None,
        critic_optimizer2: Optional[torch.optim.Optimizer] = None,

    ):
        self.device = device
        self.info = info
        self.main_batch_size = main_batch_size
        self.config = config
        
        self.actor_model = actor_model
        self.critic_local1 = critic_local1
        self.critic_local1 = self.critic_local1.to(self.device)
        self.critic_local2 = critic_local2
        self.critic_local2 = self.critic_local2.to(self.device)
        self.critic_target1 = critic_target1
        self.critic_target1 = self.critic_target1.to(self.device)
        self.critic_target2 = critic_target2
        self.critic_target2 = self.critic_target2.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = AdamW(
                self.actor_model.parameters(), lr=self.config.actor_lr
            )
        else:
            self.actor_optimizer = actor_optimizer
            
        if critic_optimizer1 is None:
            self.critic_optimizer1 = AdamW(
                self.critic_local1.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer1 = critic_optimizer1
            
        if critic_optimizer2 is None:
            self.critic_optimizer2 = AdamW(
                self.critic_local2.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer2 = critic_optimizer2
        
        self.target_entropy = 0.98 * -np.log(1 / self.actor_model.config.inst_obs_size) * \
            -np.log(1 / self.actor_model.config.knapsack_obs_size)
        self.log_alpha = torch.tensor(np.log(self.config.alpha_initial), requires_grad=True)
        self.alpha = self.log_alpha
        self.alpha_optimiser = AdamW([self.log_alpha], lr=self.config.alpha_lr)
    
        self.savePath = save_path +'/RL/SACTrainer/'
        if path.exists(self.savePath):
            self.load_models()
        
        self._transitions_stored = 0
        self.memory_observation = np.zeros((self.config.buffer_size, 
                                            self.actor_model.config.max_length,
                                            self.actor_model.config.input_dim))
        self.memory_actions = np.zeros((self.config.buffer_size, self.config.generat_link_number, 2))
        self.memory_reward = np.zeros(self.config.buffer_size)
        self.memory_new_observation = np.zeros((self.config.buffer_size, 
                                                self.actor_model.config.max_length,
                                                self.actor_model.config.input_dim))
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
        
        self.memory_observation[new_placement] = observation.cpu().detach().numpy()
        print(new_observation)
        self.memory_new_observation[new_placement] = new_observation.cpu().detach().numpy()
        self.memory_actions[new_placement] = actions.cpu().detach().numpy()
        self.memory_reward[new_placement] = sumRewards.cpu().detach().numpy()
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
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        
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
        
        observation, actions, sumRewards, next_observation, dones = \
                self.sample_step()
        
        observation = torch.tensor(observation, dtype=torch.float).to(self.actor.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.actor.device)
        sumRewards = torch.tensor(sumRewards, dtype=torch.float).to(self.actor.device)
        next_observation = torch.tensor(next_observation, dtype=torch.float).to(self.actor.device)
        dones = torch.tensor(dones).to(self.actor.device)
        
        #value = self.value(state).view(-1)
        #value_ = self.target_value(state_).view(-1)
        #value_[done] = 0.0
        prompt = None
        log_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                 dtype=torch.float64, device=self.device)
        action_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                 dtype=torch.float64, device=self.device)
        for i in range(0, self.config.generat_link_number):  
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                next_observation, self.config.generat_link_number, prompt)
            act, log_prob = self._choose_actions(generatedInstance, generatedKnapsack)
            log_probs += log_prob.squeeze()
            action_probs *= generatedInstance[list(range(0, self.config.generat_link_number)),
                                              act[:,0]]
            action_probs *= generatedKnapsack[list(range(0, self.config.generat_link_number)),
                                              act[:,1]]
        q1_values = self.critic_target1(next_observation)
        q2_values = self.critic_target2(next_observation)
        
        soft_state_values = (action_probs * (
                torch.min(q1_values, q2_values) - self.alpha * log_probs
        )).sum(dim=1)
        
        next_q_values = sumRewards + ~dones * self.config.discount_rate*soft_state_values
        
        print(self.critic_local1(observation).size())
        print(actions.size())
        print(self.dd)
        
        
        soft_q1_values = self.critic_local1(observation).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        soft_q2_values = self.critic_local2(observation).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    
        
        
        
        
        
        self.actor_optimiser.zero_grad()
        prompt = None
        log_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                 dtype=torch.float64, device=self.device)
        action_probs = torch.tensor([0]*self.config.ppo_batch_size, 
                                 dtype=torch.float64, device=self.device)
        
        for i in range(0, self.config.generat_link_number):  
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                observation, self.config.generat_link_number, prompt)
            
            act, log_prob = self._choose_actions(generatedInstance, generatedKnapsack)
            log_probs += log_prob.squeeze()
            print(act.size())
            action_probs *= generatedInstance[list(range(0, self.config.generat_link_number)),
                                              act[:,0]]
            action_probs *= generatedKnapsack[list(range(0, self.config.generat_link_number)),
                                              act[:,1]]

        q1_values = self.critic1(observation)
        q2_values = self.critic2(observation)
        inside_term = self.alpha * log_probs - torch.min(q1_values, q2_values)
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()
        print(actor_loss)
        
        actor_loss.backward()
        self.actor_optimiser.step()
            
    