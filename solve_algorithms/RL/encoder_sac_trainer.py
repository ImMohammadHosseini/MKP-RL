

"""

"""
import numpy as np
import torch
from os import path, makedirs
from typing import List, Optional
from torch.optim import Adam, SGD, RMSprop
from torch.distributions.categorical import Categorical

from .src.sac_base import SACBase
from configs.sac_configs import SACConfig

class EncoderSACTrainer_t3(SACBase):
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
        config: SACConfig,
        dim: int, 
        obj: int,
        actor_model: torch.nn.Module,
        critic_local1: torch.nn.Module,
        critic_local2: torch.nn.Module,
        critic_target1: torch.nn.Module,
        critic_target2: torch.nn.Module,
    ):
        savePath = save_path +'/RL/EncoderSACTrainer_t3/'
        super().__init__(config, savePath, dim, actor_model, critic_local1,
        critic_local2, critic_target1, critic_target2, action_num=1)
        self.info = info
        
        self.dim = dim
        self.obj = obj
        self.target_entropy = 0.98 * -np.log(1 / (self.actor_model.config.inst_obs_size * \
                                                  self.actor_model.config.knapsack_obs_size))
        if path.exists(self.save_path):
            
            self.load_models()
            self.pretrain_need = False
        else: self.pretrain_need = True
        
    def _choose_actions (
        self, 
        firstGenerated,
        secondGenerated,
    ):
        act_dist = Categorical(firstGenerated)
        act = act_dist.sample()
        #print(act)
        #prob = firstGenerated
        z = firstGenerated == 0.0
        z = z.float() * 1e-8
        log_prob = torch.log(firstGenerated + z)

        #log_prob = act_dist.log_prob(act).unsqueeze(0).unsqueeze(0)
        return act.unsqueeze(0), firstGenerated, log_prob
    
    def normal_reward (
        self,
        episod_num: int,
        action: torch.tensor,
        accepted_action: np.ndarray,
        step, 
        prompt, 
        statePrepares,
    ):
        act = int(action)
        #if act == self.actor_model.config.knapsack_obs_size * self.actor_model.config.inst_obs_size:
        #    return torch.tensor([0]).unsqueeze(0), accepted_action, step, prompt
        inst_act = int(act / self.actor_model.config.knapsack_obs_size) 
        ks_act = statePrepares.getRealKsAct(int(
            act % self.actor_model.config.knapsack_obs_size))
        knapSack = statePrepares.getKnapsack(ks_act)
            
        eCap = knapSack.getExpectedCap()
        weight = statePrepares.getObservedInstWeight(inst_act)
        value = statePrepares.getObservedInstValue(inst_act)
        
        if episod_num < 300001:
            if inst_act < statePrepares.pad_len: 
                reward = -4
            elif all(eCap >= weight):
                #dw = np.max(weight[np.where((eCap-weight)==np.min(eCap-weight))[0]])
                #reward = np.mean(+value / dw)
                reward = np.sum(value/(20*self.obj))
                knapSack.removeExpectedCap(weight)
                accepted_action = np.append(accepted_action,
                                             [[inst_act, ks_act]], 0)[1:]
    
            else:
                #dw = np.max(weight[np.where((eCap-weight)==np.min(eCap-weight))[0]])
                #1idx = int(np.where((eCap-weight)==np.min(eCap-weight))[0])
                #reward = np.mean(-value / dw) 
                reward = np.sum(-value/(20*self.obj))
                #reward = -np.mean(np.sum(value/(weight/eCap).reshape(self.dim,1),1))/50
        '''elif episod_num >= 100000 and episod_num < 200000:
            if inst_act < statePrepares.pad_len: 
                reward = -2
            elif all(eCap >= weight):
                reward = np.mean(np.sum(value/weight.reshape(self.dim,1),1))
                knapSack.removeExpectedCap(weight)
                accepted_action = np.append(accepted_action,
                                             [[inst_act, ks_act]], 0)[1:]
            else:
                reward = -np.mean(np.sum(value/weight.reshape(self.dim,1),1))
        elif episod_num >= 200000:
            if inst_act < statePrepares.pad_len: 
                reward = -1
            elif all(eCap >= weight):
                reward = np.mean(np.sum(value/(weight/eCap).reshape(self.dim,1),1))/50
                knapSack.removeExpectedCap(weight)
                accepted_action = np.append(accepted_action,
                                             [[inst_act, ks_act]], 0)[1:]
    
            else: 
                reward = -np.mean(np.sum(value/(weight/eCap).reshape(self.dim,1),1))/50'''
        reward = torch.tensor([reward]).unsqueeze(0) 
        return reward , accepted_action, step, prompt
    
    def train (
        self, 
        mode = 'normal'
    ):  
        if self.memory._normal_transitions_stored < self.config.normal_batch_size :
            return
        #TODO add extra batch size
        
        if mode == 'normal':
            #n_state = self.config.normal_batch
            #batch_size = self.config.ppo_normal_batch_size
            memoryObs, memorynNewObs, memoryAct, memoryRwd, memoryDon \
                = self.memory.get_normal_memory()
            #criticModel = self.normal_critic_model
            #criticOptim = self.normal_critic_optimizer
            
        elif mode == 'extra':
            pass #TODO add extra part
        
        obs = memoryObs.to(self.actor_model.device).to(torch.float32).detach()
        newObs = memorynNewObs.to(self.actor_model.device).to(torch.float32).detach()
        acts = memoryAct.to(self.actor_model.device).to(torch.int64).detach()
        rewards = memoryRwd.to(self.actor_model.device).detach()
        done = memoryDon.to(self.actor_model.device).detach()
        
        #critic loss 
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        
        #generatedAction = self.actor_model.generateOneStep(nextObservation)
        firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
            newObs, torch.tensor([1], dtype=torch.int64), None)
        _, prob, log_prob = self._choose_actions(firstGenerated, secondGenerated)
        
        #print(newObs.dtype)
        next1_values = self.critic_target1(newObs)
        next2_values = self.critic_target2(newObs)
        
        
        soft_state_values = (prob * (
                    torch.min(next1_values, next2_values) - self.alpha * log_prob
            )).sum(dim=1)
        
        next_q_values = rewards.squeeze() + ~done * self.config.discount_rate*soft_state_values
        
        soft_q1_values = self.critic_local1(obs)
        soft_q1_values = soft_q1_values.gather(1, acts[:,:,0]).squeeze(-1)
        
        soft_q2_values = self.critic_local2(obs)
        soft_q2_values = soft_q2_values.gather(1, acts[:,:,0]).squeeze(-1)
        
        critic1_square_error = torch.nn.MSELoss(reduction="none")(soft_q1_values.float(), next_q_values.float())
        critic2_square_error = torch.nn.MSELoss(reduction="none")(soft_q2_values.float(), next_q_values.float())
        
        weight_update = [min(l1.item(), l2.item()) for l1, l2 in zip(critic1_square_error, critic2_square_error)]
        self.memory.update_weights(weight_update)

        critic1_loss = critic1_square_error.mean()#.to(torch.float32)
        critic2_loss = critic2_square_error.mean()#.to(torch.float32)
        
        
        critic1_loss.backward(retain_graph=True)
        critic2_loss.backward(retain_graph=True)
        self.critic_optimizer1.step()
        self.critic_optimizer2.step()
        
        #actor loss
        self.actor_optimizer.zero_grad()
        firstGenerated, secondGenerated, _ = self.actor_model.generateOneStep(
            obs, torch.tensor([1], dtype=torch.int64), None)
        #generatedAction = self.actor_model.generateOneStep(externalObservation)
        _, prob, log_prob = self._choose_actions(firstGenerated, secondGenerated)
        
        local1_values = self.critic_local1(obs)
        local2_values = self.critic_local2(obs)
        
        inside_term = self.alpha * log_prob - torch.min(local1_values, local2_values)
        
        actor_loss = (prob * inside_term).sum(dim=1).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #temperature loss
        self.alpha_optimizer.zero_grad()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp() 
        
        self.soft_update(self.config.tau)
        #self.soft_update(self.critic_target2, self.critic_local2)
    
    #def soft_update(self, target_model, origin_model):
    #    for target_param, local_param in zip(target_model.parameters(), origin_model.parameters()):
    #        target_param.data.copy_(self.config.tau * local_param.data + (1 - self.config.tau) * target_param.data)
        
        
        
class EncoderMlpSACTrainer(SACBase):
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
        #main_batch_size: int,
        config: SACConfig,
        dim: int, 
        actor_model: torch.nn.Module,
        critic_local1: torch.nn.Module,
        critic_local2: torch.nn.Module,
        critic_target1: torch.nn.Module,
        critic_target2: torch.nn.Module,
        #device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer1: Optional[torch.optim.Optimizer] = None,
        critic_optimizer2: Optional[torch.optim.Optimizer] = None,

    ):
        #self.device = device
        self.info = info
        #self.main_batch_size = main_batch_size
        self.config = config
        self.dim = dim
        
        self.actor_model = actor_model
        self.critic_local1 = critic_local1
        #self.critic_local1 = self.critic_local1.to(self.device)
        self.critic_local2 = critic_local2
        #self.critic_local2 = self.critic_local2.to(self.device)
        self.critic_target1 = critic_target1
        #self.critic_target1 = self.critic_target1.to(self.device)
        self.critic_target2 = critic_target2
        #self.critic_target2 = self.critic_target2.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = Adam(
                self.actor_model.parameters(), lr=self.config.actor_lr
            )
            
        else:
            self.actor_optimizer = actor_optimizer
            
        if critic_optimizer1 is None:
            self.critic_optimizer1 = Adam(
                self.critic_local1.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer1 = critic_optimizer1
            
        if critic_optimizer2 is None:
            self.critic_optimizer2 = Adam(
                self.critic_local2.parameters(), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer2 = critic_optimizer2
        
        self.target_entropy = 0.98 * -np.log(1 / self.actor_model.config.inst_obs_size) * \
            -np.log(1 / self.actor_model.config.knapsack_obs_size)
        self.log_alpha = torch.tensor(np.log(self.config.alpha_initial), requires_grad=True)
        self.alpha = self.log_alpha.to(self.device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=self.config.alpha_lr)
    
        self.savePath = save_path +'/RL/EncoderMlpSACTrainer/'
        if path.exists(self.savePath):
            self.load_models()
        
        #Replay buffer
        self._transitions_stored = 0
        self.memory_externalObservation = np.zeros((self.config.buffer_size, 
                                                    self.actor_model.config.max_length,
                                                    self.actor_model.config.input_encode_dim))
        self.memory_actions = np.zeros(self.config.buffer_size)
        self.memory_reward = np.zeros(self.config.buffer_size)
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
        actions: np.ndarray, 
        rewards: torch.tensor,
        new_observation: torch.tensor, 
        done: bool, 
    ):
        #TODO add main_batch_size this implementation is just for one main batch
        #print('save_step')
        placement = self._transitions_stored % self.config.buffer_size

        self.memory_externalObservation[placement] = externalObservation.cpu().detach().numpy()
        self.memory_newObservation[placement] = new_observation.cpu().detach().numpy()
        self.memory_actions[placement] = actions.squeeze(0).cpu().detach().numpy()
        self.memory_reward[placement] = rewards.squeeze(0).cpu().detach().numpy()
        self.memory_done[placement] = [done]
        
        self._transitions_stored += 1
    
    def sample_step (
        self, 
    ):
        max_mem = min(self._transitions_stored, self.config.buffer_size)

        set_weights = self.weights[:max_mem] + self.delta
        probabilities = set_weights / sum(set_weights)
        try:
            self.indices = np.random.choice(range(max_mem), self.config.sac_batch_size, 
                                        p=probabilities, replace=False)
        except:
            print(probabilities)
            print(np.isnan(probabilities).any())
            print(np.isnan(probabilities).all())
        #batch = np.random.choice(max_mem, self.config.sac_batch_size)

        externalObservation = self.memory_externalObservation[self.indices]
        newObservation = self.memory_newObservation[self.indices]
        actions = self.memory_actions[self.indices]
        rewards = self.memory_reward[self.indices]
        dones = self.memory_done[self.indices]

        return externalObservation, actions, rewards, newObservation, dones
            
    def update_weights(self, prediction_errors):
        max_error = max(prediction_errors)
        self.max_weight = max(self.max_weight, max_error)
        self.weights[self.indices] = prediction_errors
        
    def _choose_actions (
        self, 
        generatedAction,
    ):
        z = generatedAction == 0.0
        z = z.float() * 1e-8
        generatedAction = generatedAction + z
        acts = torch.zeros((0,1), dtype=torch.int)
        probs = torch.zeros((0,1), dtype=torch.float)
        log_probs = torch.zeros((0,1), dtype=torch.float)
        for ga in generatedAction:
            act_dist = Categorical(ga)
            act = act_dist.sample() 
            
            acts = torch.cat([acts, act.unsqueeze(0).unsqueeze(0).cpu()], dim=0)
            probs = torch.cat([probs, ga[act].unsqueeze(0).unsqueeze(0).cpu()], dim=0)
            log_probs = torch.cat([log_probs, act_dist.log_prob(act).unsqueeze(0).unsqueeze(0).cpu()], dim=0)

        return acts, probs.to(self.device), log_probs.to(self.device)
    
    def internal_reward (
        self,
        action: torch.tensor,
        accepted_action: np.ndarray,
        statePrepares: np.ndarray,
    ):
        rewards = []
        for index, act in enumerate(action):
            #print(act)
            inst_act = int(act % self.actor_model.config.inst_obs_size) 
            ks_act = statePrepares[index].getRealKsAct(int(
                act // self.actor_model.config.inst_obs_size))
            knapSack = statePrepares[index].getKnapsack(ks_act)
            
            eCap = knapSack.getExpectedCap()
            weight = statePrepares[index].getObservedInstWeight(inst_act)
            value = statePrepares[index].getObservedInstValue(inst_act)

            if inst_act < statePrepares[index].pad_len: 
                    rewards.append(0)#-(self.info['VALUE_HIGH']/(5*self.info['WEIGHT_LOW'])))
                    continue
            
            if all(eCap >= weight):
                rewards.append(+(value / np.sum(weight)))
                knapSack.removeExpectedCap(weight)
                accepted_action[index] = np.append(accepted_action[index][1:],
                                                   [[inst_act, ks_act]], 0)
            else:
                rewards.append(0)#-(value / np.sum(weight)))#-self.info['VALUE_LOW'])
        #print(rewards)
        return torch.tensor(rewards, device=self.device).unsqueeze(0)
    
    def make_steps (
        self,
        externalObservation: torch.tensor,
        statePrepares: np.ndarray,
    ):
        accepted_action = np.array([[[-1]*2]*1]*self.main_batch_size, dtype= int)
        generatedAction = self.actor_model.generateOneStep(externalObservation)
        act, _, _ = self._choose_actions(generatedAction)
        internalReward = self.internal_reward(act, accepted_action, statePrepares)
        #print(accepted_action)
        return act, accepted_action, internalReward
    
    def train (
        self, 
    ):
        #print(self._transitions_stored)
        if self._transitions_stored < self.config.sac_batch_size :
            return
        
        externalObservation, actions, rewards, nextObservation, dones = \
            self.sample_step()
        
        externalObservation = torch.tensor(externalObservation, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        nextObservation = torch.tensor(nextObservation, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones).to(self.device)
        
        #critic loss 
        self.critic_optimizer1.zero_grad()
        self.critic_optimizer2.zero_grad()
        
        
        generatedAction = self.actor_model.generateOneStep(nextObservation)
        _, prob, log_prob = self._choose_actions(generatedAction)
        
        next1_values = self.critic_target1.generateOneStep(nextObservation)
        next2_values = self.critic_target2.generateOneStep(nextObservation)
        
        soft_state_values = (prob * (
                    torch.min(next1_values, next2_values) - self.alpha * log_prob
            )).sum(dim=1)
        
        next_q_values = rewards + ~dones * self.config.discount_rate*soft_state_values
        
        soft_q1_values = self.critic_local1.generateOneStep(externalObservation)
        soft_q1_values = soft_q1_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        
        soft_q2_values = self.critic_local2.generateOneStep(externalObservation)
        soft_q2_values = soft_q2_values.gather(1, actions.unsqueeze(1)).squeeze(-1)
        
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
        generatedAction = self.actor_model.generateOneStep(externalObservation)
        _, prob, log_prob = self._choose_actions(generatedAction)
        
        local1_values = self.critic_local1.generateOneStep(externalObservation)
        local2_values = self.critic_local2.generateOneStep(externalObservation)
        
        inside_term = self.alpha * log_prob - torch.min(local1_values, local2_values)
        
        actor_loss = (prob * inside_term).sum(dim=1).mean()
        
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #temperature loss
        self.alpha_optimizer.zero_grad()
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
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