
"""

"""
import numpy as np
import torch
from os import path, makedirs
from typing import List, Optional
from torch.optim import AdamW, SGD, RMSprop
from torch.distributions.categorical import Categorical

from .src.base_trainer import BaseTrainer
from configs.fraction_sac_configs import FractionSACConfig


class FractionSACTrainer(BaseTrainer):
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
        config: FractionSACConfig,
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
    
        self.savePath = save_path +'/RL/FractionSACTrainer/'
        if path.exists(self.savePath):
            self.load_models()
        
        self._transitions_stored = 0
        self.memory_externalObservation = np.zeros((self.config.buffer_size, 
                                                    self.actor_model.config.max_length,
                                                    self.actor_model.config.input_encode_dim))
        self.memory_internalObservation = np.zeros((self.config.buffer_size, 
                                                    self.config.generat_link_number+1, 
                                                    self.actor_model.config.input_decode_dim))
        self.memory_actions = np.zeros((self.config.buffer_size, 2))
        self.memory_reward = np.zeros(self.config.buffer_size)
        self.memory_newObservation = np.zeros((self.config.buffer_size, 
                                                self.actor_model.config.max_length,
                                                self.actor_model.config.input_encode_dim))
        self.memory_done = np.zeros(self.config.buffer_size, dtype=np.bool)
    
    def save_step (
        self, 
        externalObservation: torch.tensor, 
        internalObservations: torch.tensor,
        actions: np.ndarray, 
        rewards: torch.tensor,
        new_observation: torch.tensor, 
        done: bool, 
    ):
        #TODO add main_batch_size this implementation is just for one main batch
        #print('save_step')
        start_placement = self._transitions_stored % self.config.buffer_size
        end_placement = start_placement + self.config.generat_link_number
        #print(torch.cat([externalObservation]*10, 0).size())
        #print(internalObservations.size())
        #print(actions.size())
        #print(rewards.size())
        #print(new_observation.size())
        #print([done]*10)

        self.memory_externalObservation[start_placement:end_placement] = torch.cat(
            [externalObservation]*self.config.generat_link_number, 0).cpu().detach().numpy()
        self.memory_internalObservation[start_placement:end_placement] = internalObservations.squeeze(0).cpu().detach().numpy()
        #TODO new-observation consept WITH internal observation
        self.memory_newObservation[start_placement:end_placement] = torch.cat(
            [new_observation]*self.config.generat_link_number, 0).cpu().detach().numpy()
        self.memory_actions[start_placement:end_placement] = actions.squeeze(0).cpu().detach().numpy()
        self.memory_reward[start_placement:end_placement] = rewards.squeeze(0).cpu().detach().numpy()
        #TODO change done: when dine is true the last one is true and others are false beacause of internal observation
        self.memory_done[start_placement:end_placement] = [done]*self.config.generat_link_number

        self._transitions_stored += self.config.generat_link_number
    
    def sample_step (
        self, 
    ):
        max_mem = min(self._transitions_stored, self.config.buffer_size)

        batch = np.random.choice(max_mem, self.config.sac_batch_size)

        externalObservation = self.memory_externalObservation[batch]
        internalObservation = self.memory_internalObservation[batch]
        newObservation = self.memory_newObservation[batch]
        actions = self.memory_actions[batch]
        rewards = self.memory_reward[batch]
        dones = self.memory_done[batch]

        return externalObservation, internalObservation, actions, rewards, newObservation, dones
        
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
    
    def reward (
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
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
        done: Optional[torch.tensor] = None,
    ):
        actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int, device=self.device)
        rewards = torch.zeros((self.main_batch_size,0), dtype=torch.float64, device=self.device)
        internalObservations = torch.zeros([self.main_batch_size, 0, 
                                            self.config.generat_link_number+1, 
                                            self.actor_model.config.input_decode_dim],
                                           device=self.device)
        
        prompt = None
        accepted_actions = np.array([[[-1]*2]*self.config.generat_link_number]*self.main_batch_size, dtype= int)
        step = torch.tensor([0]*self.main_batch_size, dtype=torch.int64, device=self.device)
        steps = torch.zeros((self.main_batch_size,0), dtype=torch.int64, device=self.device)
        
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generateOneStep(
                step, externalObservation, prompt)
            steps = torch.cat([steps, step.unsqueeze(1)], 1)
            internalObservations = torch.cat([internalObservations, prompt.unsqueeze(1)], 1)
            act, _ = self._choose_actions(generatedInstance, generatedKnapsack)
            actions = torch.cat([actions, act.unsqueeze(1).to(self.device)], 1)
            reward = self.reward(act, accepted_actions, step, prompt, 
                                 statePrepares)

            rewards = torch.cat([rewards, reward.unsqueeze(1)], 1)
            
        return internalObservations, actions, accepted_actions, rewards, steps

    
    def train (
        self, 
    ):
        if self._transitions_stored < self.config.sac_batch_size :
            return
        
        externalObservation, internalObservation, actions, rewards, \
        nextObservation, dones = self.sample_step()
        
        externalObservation = torch.tensor(externalObservation, dtype=torch.float).to(self.device)
        internalObservation = torch.tensor(internalObservation, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        nextObservation = torch.tensor(nextObservation, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
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
        
        next_q_values = rewards + ~dones * self.config.discount_rate*soft_state_values
        
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