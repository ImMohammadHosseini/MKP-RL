
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
        self.alpha = self.log_alpha.to(self.device)
        self.alpha_optimizer = AdamW([self.log_alpha], lr=self.config.alpha_lr)
    
        self.savePath = save_path +'/RL/FractionSACTrainer/'
        if path.exists(self.savePath):
            self.load_models()
        
        #Replay buffer
        self._transitions_stored = 0
        self.memory_externalObservation = np.zeros((self.config.buffer_size, 
                                                    self.actor_model.config.max_length,
                                                    self.actor_model.config.input_encode_dim))
        self.memory_internalObservation = np.zeros((self.config.buffer_size, 
                                                    self.config.generat_link_number+1, 
                                                    self.actor_model.config.input_decode_dim))
        self.memory_actions = np.zeros((self.config.buffer_size, 2))
        self.memory_reward = np.zeros(self.config.buffer_size)
        self.memory_step = np.zeros(self.config.buffer_size)
        self.memory_newObservation = np.zeros((self.config.buffer_size, 
                                                self.actor_model.config.max_length,
                                                self.actor_model.config.input_encode_dim))
        self.memory_done = np.zeros(self.config.buffer_size, dtype=np.bool)
        self.weights = np.zeros(self.config.buffer_size)
        self.max_weight = 10**-2
        self.delta = 10**-4
        self.indices = None
        
    def save_step (
        self, 
        externalObservation: torch.tensor, 
        internalObservations: torch.tensor,
        actions: np.ndarray, 
        rewards: torch.tensor,
        new_observation: torch.tensor, 
        steps: torch.tensor, 
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
        #print(steps.size())
        #print([done]*10)

        self.memory_externalObservation[start_placement:end_placement] = torch.cat(
            [externalObservation]*self.config.generat_link_number, 0).cpu().detach().numpy()
        self.memory_internalObservation[start_placement:end_placement] = internalObservations.squeeze(0).cpu().detach().numpy()
        self.memory_newObservation[start_placement:end_placement] = torch.cat(
            [externalObservation]*(self.config.generat_link_number-1)+[new_observation], 
            0).cpu().detach().numpy()
        self.memory_actions[start_placement:end_placement] = actions.squeeze(0).cpu().detach().numpy()
        self.memory_reward[start_placement:end_placement] = rewards.squeeze(0).cpu().detach().numpy()
        self.memory_step[start_placement:end_placement] = steps.squeeze(0).cpu().detach().numpy()
        #TODO change done: when dine is true the last one is true and others are false beacause of internal observation
        self.memory_done[start_placement:end_placement] = [done]*self.config.generat_link_number

        self._transitions_stored += self.config.generat_link_number
    
    def sample_step (
        self, 
    ):
        max_mem = min(self._transitions_stored, self.config.buffer_size)

        set_weights = self.weights[:max_mem] + self.delta
        probabilities = set_weights / sum(set_weights)
        self.indices = np.random.choice(range(max_mem), self.config.sac_batch_size, 
                                        p=probabilities, replace=False)
        

        #batch = np.random.choice(max_mem, self.config.sac_batch_size)

        externalObservation = self.memory_externalObservation[self.indices]
        internalObservation = self.memory_internalObservation[self.indices]
        newObservation = self.memory_newObservation[self.indices]
        actions = self.memory_actions[self.indices]
        rewards = self.memory_reward[self.indices]
        steps = self.memory_step[self.indices]
        dones = self.memory_done[self.indices]

        return externalObservation, internalObservation, actions, rewards, \
            newObservation, steps, dones
            
    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors
  
    def _choose_actions (
        self, 
        generatedInstance, 
        generatedKnapsack, 
        statePrepares: Optional[np.ndarray] = None,
    ):
        acts = torch.zeros((0,2), dtype=torch.int)
        probs = torch.zeros((0,2), dtype=torch.int)
        log_probs = torch.zeros((0,2), dtype=torch.int)
        for i, generat in enumerate(zip(generatedInstance, generatedKnapsack)):
            pad_len = 0 if statePrepares == None else statePrepares[i].pad_len
            inst_dist = Categorical(generat[0])
            ks_dist = Categorical(generat[1])
            inst_act = inst_dist.sample() 
            ks_act = ks_dist.sample()
        
            inst_probs = generat[0][:,inst_act.squeeze()]
            ks_probs = generat[1][:,ks_act.squeeze()]
            inst_log_probs = inst_dist.log_prob(inst_act)
            ks_log_probs = ks_dist.log_prob(ks_act)
            inst_act += pad_len
            acts = torch.cat([acts, torch.cat([inst_act.unsqueeze(0), 
                                               ks_act.unsqueeze(0)], dim=1)], dim=0)
            probs = torch.cat([probs, torch.cat([inst_probs.unsqueeze(0), 
                                               ks_probs.unsqueeze(0)], dim=1)], dim=0)
            log_probs = torch.cat([log_probs, torch.cat([inst_log_probs.unsqueeze(0), 
                                               ks_log_probs.unsqueeze(0)], dim=1)], dim=0)


        return acts, probs.to(self.device), log_probs.to(self.device)
    
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
            inst_act = int(act[0])
            #TODO check  + statePrepares[index].pad_len
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
        actions = torch.zeros((self.main_batch_size,0,2), dtype=torch.int64, device=self.device)
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
            act, _, _ = self._choose_actions(generatedInstance, generatedKnapsack,
                                             statePrepares)
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
        nextObservation, steps, dones = self.sample_step()
        
        externalObservation = torch.tensor(externalObservation, dtype=torch.float).to(self.device)
        internalObservation = torch.tensor(internalObservation, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        steps = torch.tensor(steps, dtype=torch.int64).to(self.device)
        nextObservation = torch.tensor(nextObservation, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        #critic loss 
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        
        generatedInstance, generatedKnapsack, _ = self.actor_model.generateOneStep(
            steps, nextObservation, internalObservation)
        _, prob, log_prob = self._choose_actions(generatedInstance, generatedKnapsack)
        
        next1_values_inst, next1_values_ks, _ = self.critic_target1.generateOneStep(
            steps, nextObservation, internalObservation)
        next2_values_inst, next2_values_ks, _ = self.critic_target2.generateOneStep(
            steps, nextObservation, internalObservation)
        
        
        inst_soft_state_values = (prob[:,0].unsqueeze(1) * ( 
            torch.min(next1_values_inst, next2_values_inst) - self.alpha * log_prob[:,0].unsqueeze(1)
        )).sum(dim=1)
        ks_soft_state_values = (prob[:,1].unsqueeze(1) * ( 
            torch.min(next1_values_ks, next2_values_ks) - self.alpha * log_prob[:,1].unsqueeze(1)
        )).sum(dim=1)
        
        #TODO check addition instead of multiple
        soft_state_values = inst_soft_state_values * ks_soft_state_values
        next_q_values = rewards + ~dones * self.config.discount_rate*soft_state_values
        
        inst_soft_q1_values, ks_soft_q1_values, _ = self.critic_local1.generateOneStep(
            steps, externalObservation, internalObservation)
        inst_soft_q1_values = inst_soft_q1_values.gather(1, actions[:,0].unsqueeze(-1)).squeeze(-1)
        ks_soft_q1_values = ks_soft_q1_values.gather(1, actions[:,1].unsqueeze(-1)).squeeze(-1)
        soft_q1_values = inst_soft_q1_values * ks_soft_q1_values
        
        inst_soft_q2_values, ks_soft_q2_values, _ = self.critic_local2.generateOneStep(
            steps, externalObservation, internalObservation)
        inst_soft_q2_values = inst_soft_q2_values.gather(1, actions[:,0].unsqueeze(-1)).squeeze(-1)
        ks_soft_q2_values = ks_soft_q2_values.gather(1, actions[:,1].unsqueeze(-1)).squeeze(-1)
        soft_q2_values = inst_soft_q2_values * ks_soft_q2_values
        
        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q1_values, next_q_values)
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q2_values, next_q_values)
        
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic1_square_error, critic2_square_error)]
        self.update_weights(weight_update)
    
        critic1_loss = critic1_square_error.mean()
        critic2_loss = critic2_square_error.mean()
        
        critic1_loss.backward(retain_graph=True)
        self.critic_optimizer1.step()
        critic2_loss.backward(retain_graph=True)
        self.critic_optimizer2.step()
        
        #actor loss
        self.actor_optimizer.zero_grad()
        generatedInstance, generatedKnapsack, _ = self.actor_model.generateOneStep(
            steps, externalObservation, internalObservation)
        _, prob, log_prob = self._choose_actions(generatedInstance, generatedKnapsack)
        
        local1_values_inst, local1_values_ks, _ = self.critic_local1.generateOneStep(
            steps, externalObservation, internalObservation)
        local2_values_inst, local2_values_ks, _ = self.critic_local2.generateOneStep(
            steps, externalObservation, internalObservation)
        
        inst_inside_term = self.alpha * log_prob[:,0].unsqueeze(-1) - torch.min(local1_values_inst, local2_values_inst)
        ks_inside_term = self.alpha * log_prob[:,1].unsqueeze(-1) - torch.min(local1_values_ks, local2_values_ks)
        
        inst_actor_loss = (prob[:,0].unsqueeze(-1) * inst_inside_term).sum(dim=1).mean()
        ks_actor_loss = (prob[:,1].unsqueeze(-1) * ks_inside_term).sum(dim=1).mean()
        actor_loss = inst_actor_loss + ks_actor_loss
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #temperature loss
        self.alpha_optimizer.zero_grad()
        
        alpha_loss = -(self.log_alpha * ((
            log_prob[:,0] + log_prob[:,1]).unsqueeze(-1) + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp() 
        
        self.soft_update(self.critic_target1, self.critic_local1)
        self.soft_update(self.critic_target2, self.critic_local2)
        
       
    def soft_update(self, target_model, origin_model):
        for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
            target_param.data.copy_(self.config.tau * local_param.data + (1 - self.config.tau) * target_param.data)
        
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