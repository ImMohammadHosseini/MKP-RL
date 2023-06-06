
"""

"""
import torch
import numpy as np
from torch.optim import Adam, Rprop
from typing import List, Optional

from .src.core import (
    set_seed,  
)
from torch.distributions.categorical import Categorical
from src.data_structure.state_prepare import ExternalStatePrepare

from torch.autograd import Variable
from .src.base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig
from os import path, makedirs

class PPOTrainer(BaseTrainer):
    r"""
    this implementation is inspired by: 
        https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_trainer.py and 
        https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    """
    
    def __init__(
        self,
        info: List,
        save_path: str,
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
        
        #set_seed(config.seed)
        
        self.device = device
        self.softmax = torch.nn.Softmax()
        self.info = info
        
        self.actor_model = actor_model
        self.critic_model = critic_model
        self.critic_model = self.critic_model.to(self.device)
        self.external_critic_model = external_critic_model
        self.external_critic_model = self.external_critic_model.to(self.device)
        
        if actor_optimizer is None:
            self.actor_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.actor_model.parameters()), lr=self.config.actor_lr
            )
        else:
            self.actor_optimizer = actor_optimizer
        
        if critic_optimizer is None:
            self.critic_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.critic_model.parameters()), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer = critic_optimizer
            
        if external_critic_optimizer is None:
            self.external_critic_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.external_critic_model.parameters()), lr=self.config.critic_lr
            )
        else:
            self.external_critic_model = external_critic_model
            
        self.savePath = save_path +'/RL/PPOTrainer/'
        if path.exists(self.savePath):
            self.load_models()
            
    '''def _make_distribution (
        self, 
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        generatedInstance = stepGenerate[0].squeeze().cpu().detach().numpy()#[:, :self.actor_model.config.problem_dim].cpu().detach().numpy()
        insts = externalObservation[0][1:self.actor_model.config.inst_obs_size+1,
                                       :-1].cpu().detach().numpy()
        insts = np.append(insts, np.array([[-1]*self.actor_model.config.problem_dim]),0)
        generatedKnapsack = stepGenerate[1].squeeze().cpu().detach().numpy()#[:, self.actor_model.config.problem_dim:].cpu().detach().numpy()
        ks = externalObservation[0][self.actor_model.config.inst_obs_size+2:-1,
                                    :-1].cpu().detach().numpy()
        ks = np.append(ks, np.array([[-1]*self.actor_model.config.problem_dim]),0)
        
        inst_cosin_sim = (generatedInstance @ insts.T)/(np.expand_dims(
            np.linalg.norm(generatedInstance, axis=1),0).T @ np.expand_dims(
                np.linalg.norm(insts, axis=1),0))
        ks_cosin_sim = (generatedKnapsack @ ks.T)/(np.expand_dims(
            np.linalg.norm(generatedKnapsack, axis=1),0).T @ np.expand_dims(
                np.linalg.norm(ks, axis=1),0))
        
        inst_dist = self.softmax(torch.nan_to_num(torch.tensor(inst_cosin_sim)))
        ks_dist = self.softmax(torch.nan_to_num(torch.tensor(ks_cosin_sim)))
        
        return inst_dist, ks_dist'''
    
    def _choose_actions (
        self, 
        generatedInstance: torch.tensor, 
        generatedKnapsack: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        #inst_dist, ks_dist = self._make_distribution(stepGenerate, 
        #                                             externalObservation)
        #inst_act = inst_dist.max(1)[1]#TODO
        #ks_act = ks_dist.max(1)[1]
        
        inst_dist = Categorical(generatedInstance)
        ks_dist = Categorical(generatedKnapsack)
        
        #acts = generated_dist.sample().squeeze()
        inst_act = inst_dist.sample() 
        ks_act = ks_dist.sample()
        inst_log_probs = inst_dist.log_prob(inst_act)
        ks_log_probs = ks_dist.log_prob(ks_act)
        #print(acts.unsqueeze(dim=1).size())
        acts = torch.cat([inst_act,
                          ks_act],dim=1)
        log_probs = inst_log_probs + ks_log_probs
        #log_probs = generated_dist.log_prob(acts)
        
        return acts, log_probs#.squeeze()
    
    def internal_reward (
        self,
        actions: torch.tensor,
        probs: torch.tensor,
        statePrepare: ExternalStatePrepare,
    ):
        #num = 0; num1 = 0
        rewards = []
        accepted_action = np.zeros((0,2), dtype= int)
        accepted_log_prob = np.array([])
        for index, act in enumerate(actions):
            inst_act = int(act[0])
            ks_act = int(act[1])
        #for act_indx in range(0, actions.size(0), 2):
            #inst_act = int(actions[act_indx])
            #ks_act = int(actions[act_indx+1]) - statePrepare.instanceObsSize 
            #if ks_act < 0 or \ or inst_act >= statePrepare.instanceObsSize
            if inst_act < statePrepare.pad_len or inst_act in accepted_action[:,0]: 
                    rewards.append(-self.info['VALUE_LOW'])
                    #rewards.append(0)#-((self.actor_model.config.problem_dim * self.info['WEIGHT_HIGH']) / \
                                     #(self.actor_model.config.problem_dim * self.info['CAP_LOW'])))
                    continue
            #if inst_act == statePrepare.instanceObsSize or \
            #ks_act == statePrepare.knapsackObsSize:
            #    rewards.append(0)
                #num1 += 1
            #elif inst_act < statePrepare.pad_len or \
            #    inst_act in accepted_action[:,0]:
            #    rewards.append(-int(self.info['VALUE_LOW']))
            #else:
            
            knapSack = statePrepare.getKnapsack(ks_act)
            eCap = knapSack.getExpectedCap()
            weight = statePrepare.getObservedInstWeight(inst_act)
            '''if all(weight == 0):
                rewards.append(-int(self.info['VALUE_LOW']))'''
            if all(eCap >= weight):
                value = statePrepare.getObservedInstValue(inst_act)
                rewards.append(value / np.sum(weight))
                #rewards.append(np.sum(weight)/np.sum(eCap)/np.sum(weight))
                knapSack.removeExpectedCap(weight)
                accepted_action = np.append(accepted_action, 
                                            [[inst_act, ks_act]], 0)
                accepted_log_prob = np.append(accepted_log_prob, float(probs[index]))
                #accepted_log_prob = np.append(accepted_log_prob, 
                #                              (float(probs[act_indx])+float(probs[act_indx+1])))
            else:
                #num += 1
                rewards.append(0)#-(self.info['VALUE_HIGH'] / \
                                 #(self.actor_model.config.problem_dim * self.info['WEIGHT_LOW'])))
                #rewards.append(0)#-((self.actor_model.config.problem_dim * self.info['WEIGHT_HIGH']) / \
                                 #(self.actor_model.config.problem_dim * self.info['CAP_LOW'])))
        #print('num, all, num1', (num, len(actions), num1))
        return torch.tensor(rewards, device=self.device), accepted_action, accepted_log_prob
    
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
        done: Optional[torch.tensor] = None,
    ):
        len_act = 0
        externalObservation = torch.cat(
            [externalObservation]*(self.config.internal_batch), 0)#2*
        prompt = None
        step_acts = np.zeros((0,2), dtype= int)
        step_prob = np.array([])
        actions = torch.tensor([], dtype=torch.int, device=self.device)
        probs = torch.tensor([], dtype=torch.float64, device=self.device)
        internalRewards = torch.tensor([], dtype=torch.float64, device=self.device)
        internalObservation = torch.zeros([0, self.config.generat_link_number, #2*
                                           self.actor_model.config.output_dim], 
                                          device = self.device)
        for i in range(0, self.config.generat_link_number):
            generatedInstance, generatedKnapsack, prompt = self.actor_model.generate_step(externalObservation[0].unsqueeze(0), 1, #2
                                                                  self.config.generat_link_number, prompt)
            #internalObservation = torch.cat([internalObservation, prompt[:,-(self.config.generat_link_number+1):-1,:]], 0)#2*
            internalObservation = torch.cat([internalObservation, prompt[:,-(self.config.generat_link_number):,:]], 0)#2*
            
            act, prob = self._choose_actions(generatedInstance, generatedKnapsack, externalObservation)
            len_act += int(act.size(0)) #TODO delete  / 2
            internalReward, accepted_action, accepted_prob = self.internal_reward(
                act, prob, statePrepare)
            
            actions = torch.cat([actions, act], 0)
            probs = torch.cat([probs, prob], 0)
            internalRewards = torch.cat([internalRewards, internalReward],0)
            if len(accepted_action) > 0:
                step_acts = np.append(step_acts, accepted_action, 0)
                step_prob = np.append(step_prob, accepted_prob)
            else:
                prompt = prompt[:,:-1]#-2
            if i % self.config.internal_batch == self.config.internal_batch-1:
                values = self.critic_model(externalObservation.to(self.device), 
                                           internalObservation.to(self.device))
                for _ in range(self.config.ppo_epochs):
                    dones, batches = self._generate_batch(done)
                    self._train_minibatch(self.config.internal_batch, externalObservation, #2*
                                          actions, probs, values.squeeze(), 
                                          internalRewards, batches, 
                                          internalObservation, dones)
                actions = torch.tensor([], dtype=torch.int, device=self.device)
                probs = torch.tensor([], dtype=torch.float64, device=self.device)
                internalRewards = torch.tensor([], dtype=torch.float64, device=self.device)
                internalObservation = torch.zeros([0, self.config.generat_link_number, #2*
                                                   self.actor_model.config.output_dim], 
                                                  device = self.device)
         
        #print('act:', len_act, 'acp_act', len(step_acts))
        return step_acts, step_prob
        
        '''for i in range(0, self.config.generat_link_number+1, self.config.internal_batch):
            stepGenerate, internalObservation, prompt = self.generate_step(
                externalObservation[0].unsqueeze(0), 
                2*self.config.internal_batch, self.config.generat_link_number,
                prompt)
            print(prompt.size())

            #print('gggggggg' )
            #print('step ____any', torch.max(externalObservation))

            action, prob = self._choose_actions(stepGenerate, externalObservation)
            
            values = self.critic_model(externalObservation.to(self.device), 
                                       internalObservation.to(self.device))
            intervalReward, accepted_action = self.internal_reward(
                action, statePrepare)
            step_acts = np.append(step_acts, accepted_action, 0)
            learn_iter += 1
            for _ in range(self.config.ppo_epochs):
                dones, batches = self._generate_batch(done)
                self.train_minibatch(externalObservation, internalObservation, 
                                     action, prob, values.squeeze(), 
                                     intervalReward, dones, batches)
        return step_acts, learn_iter'''
    
    def test_step (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
    ):
        prompt = None
        step_acts = np.zeros((0,2), dtype= int)
        for i in range(0, self.config.generat_link_number):
            stepGenerate, prompt = self.generate_step(externalObservation[0].unsqueeze(0), 2, 
                                                      self.config.generat_link_number, prompt)
            
            act, prob = self._choose_actions(stepGenerate, externalObservation)
            internalReward, accepted_action, accepted_prob = self.internal_reward(
                act, prob, statePrepare)
            
            if len(accepted_action) > 0:
                step_acts = np.append(step_acts, accepted_action, 0)
            else:
                prompt = prompt[:,:-2]
        
        return step_acts
        
    def _generate_batch (
        self, 
        n_states: Optional[int] = None,
        batch_size: Optional[int] = None,
        done: Optional[torch.tensor] = None,
    ):
        if batch_size == None:
            batch_size = self.config.ppo_batch_size
        if n_states == None:
            n_states = self.config.internal_batch#2*
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+batch_size] for i in batch_start]
        
        if done is None:
            dones = torch.tensor([False]*n_states)
        else:
            dones = torch.tensor([done]*n_states).squeeze()
        
        return dones,\
                batches
    
    def _train_minibatch (
        self, 
        n_states: int,
        external_obs: torch.tensor,
        actions: torch.tensor,
        old_log_probs: torch.tensor,
        vals: torch.tensor,
        rewards: torch.tensor,
        batches: List,
        internal_obs: Optional[torch.tensor]=None,
        dones: Optional[torch.tensor]=None,
        mode = "internal",
    ):
        advantage = torch.zeros(n_states, dtype=torch.float32, device=self.device)
        for t in range(n_states-1):
            discount = 1
            a_t = 0
            for k in range(t, n_states-1):
                a_t += discount*(int(rewards[k]) + self.config.gamma*(float(vals[k+1]))*\
                            (1) - (float(vals[k]))) # -int(dones[k])
                discount *= self.config.gamma*self.config.gae_lambda
            advantage[t] = a_t
            
        for batch in batches:
            eo = external_obs[batch]
            if mode == "internal":
                io = internal_obs[batch]
            else: io = None
            olp = old_log_probs[batch]
            #actions = torch.tensor(actions.data, device=torch.device('cpu'))
            acts = actions[batch]
            
            generatedInstance, generatedKnapsack, _ = self.actor_model.generate_step(eo, 1, self.config.generat_link_number, 
                                                             io)
            #print(stepGenerate.size())
            #inst_dist, ks_dist = self._make_distribution(stepGenerate, 
            #                                             eo)
            new_inst_dist = Categorical(generatedInstance)
            new_ks_dist = Categorical(generatedKnapsack)
            #new_dist = Categorical (stepGenerate)
            
            if mode == "internal":
                critic_value = self.critic_model(eo.to(self.device), io.to(self.device))
                new_log_probs = new_inst_dist.log_prob(acts[:,0]) + new_ks_dist.log_prob(acts[:,1])
                #new_log_probs = new_dist.log_prob(acts)

            else:
                critic_value = self.external_critic_model(eo.to(self.device))
                new_log_probs = new_inst_dist.log_prob(acts[:,0]) + new_ks_dist.log_prob(acts[:,1])


            critic_value = torch.squeeze(critic_value)
            
            #new_inst_log_probs = inst_dist.log_prob(acts[:,0])
            #new_ks_log_probs = ks_dist.log_prob(acts[:,1])
            #new_log_probs = new_inst_log_probs + new_ks_log_probs
            
            prob_ratio = (new_log_probs.exp() / olp.exp()).to(self.device)
            
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                        1+self.config.cliprange)*advantage[batch]
            
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch].unsqueeze(dim=1) + vals[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            
            #print('ad', actor_loss)
            #print('ma', critic_loss)
            #actor_loss_leaf = Variable(actor_loss.data, requires_grad=True)
            #critic_loss_leaf = Variable(critic_loss.data, requires_grad=True)
            total_loss = actor_loss + 0.5*critic_loss
            total_loss = Variable(total_loss.data, requires_grad=True)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad() if mode == "internal" else \
                self.external_critic_optimizer.zero_grad()
            total_loss.backward()
            #actor_loss_leaf.backward()
            self.actor_optimizer.step()
            #critic_loss_leaf.backward(retain_graph=True)
            self.critic_optimizer.step() if mode == "internal" else \
                self.external_critic_optimizer.step()
    
    def external_train (
        self,
        externalObservation: torch.tensor,
        actions: torch.tensor,
        old_log_probs: torch.tensor,
        external_reward,
        statePrepare: ExternalStatePrepare,
    ):
        #print(externalObservation.size())
        #print(actions.size())
        #print(old_log_probs.size())
        #print(external_reward.size())
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
    