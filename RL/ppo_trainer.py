
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
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        #inst_dist, ks_dist = self._make_distribution(stepGenerate, 
        #                                             externalObservation)
        #inst_act = inst_dist.max(1)[1]#TODO
        #ks_act = ks_dist.max(1)[1]
        
        generated_dist = Categorical(stepGenerate)
        #ks_dist = Categorical(ks_dist)
        
        acts = generated_dist.sample().squeeze()
        #inst_act = inst_dist.sample() 
        #ks_act = ks_dist.sample()
        #inst_log_probs = inst_dist.log_prob(inst_act)
        #ks_log_probs = ks_dist.log_prob(ks_act)
        #print(acts.unsqueeze(dim=1).size())
        #acts = torch.cat([inst_act.unsqueeze(dim=1),
        #                  ks_act.unsqueeze(dim=1)],dim=1)
        #log_probs = inst_log_probs + ks_log_probs
        log_probs = generated_dist.log_prob(acts)
        
        return acts, log_probs.squeeze()
    
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
        #for inst_act, ks_act in actions:
        for act_indx in range(0, actions.size(0), 2):
            inst_act = int(actions[act_indx])
            ks_act = int(actions[act_indx+1]) - statePrepare.instanceObsSize 
            if ks_act < 0 or \
                inst_act < statePrepare.pad_len or inst_act >= statePrepare.instanceObsSize or \
                inst_act in accepted_action[:,0]: 
                    rewards.append(0)#-self.info['VALUE_LOW'])
                    rewards.append(0)#-((self.actor_model.config.problem_dim * self.info['WEIGHT_HIGH']) / \
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
                rewards.append(np.sum(weight)/np.sum(eCap)/np.sum(weight))
                knapSack.removeExpectedCap(weight)
                accepted_action = np.append(accepted_action, 
                                            [[inst_act, ks_act]], 0)
                accepted_log_prob = np.append(accepted_log_prob, 
                                              (float(probs[act_indx])+float(probs[act_indx+1])))
            else:
                #num += 1
                rewards.append(-(self.info['VALUE_HIGH'] / \
                                 (self.actor_model.config.problem_dim * self.info['WEIGHT_LOW'])))
                rewards.append(-((self.actor_model.config.problem_dim * self.info['WEIGHT_HIGH']) / \
                                 (self.actor_model.config.problem_dim * self.info['CAP_LOW'])))
        #print('num, all, num1', (num, len(actions), num1))
        return torch.tensor(rewards, device=self.device), accepted_action, accepted_log_prob
    
    def generate_step (
        self,
        external_obs: torch.tensor,
        max_tokens_to_generate: int,
        generat_link_number: int,
        promp_tensor: Optional[torch.tensor] = None,   
        
    ):
        encoder_ACT = [0]*self.actor_model.config.input_dim
        encoder_mask = torch.ones_like(external_obs[:,:,0], device=self.device)
        encoder_mask[torch.all(external_obs == torch.tensor(encoder_ACT, 
                                                            device=self.device), 
                               dim=2)] = 0
        encoder_mask = torch.cat([encoder_mask]*self.actor_model.config.nhead , 0)

        decoder_ACT = [0]*self.actor_model.config.output_dim
        if promp_tensor == None:
            start_tokens = [decoder_ACT]
            promp_tensor = torch.tensor(
                self.actor_model.pad_left(
                    sequence=start_tokens,
                    final_length=2 * generat_link_number,#
                    padding_token=decoder_ACT
                    ),
                dtype=torch.long
            )
            promp_tensor = promp_tensor.unsqueeze(dim=0)
            promp_tensor = torch.cat([promp_tensor]*external_obs.size(0), 0)
        
        promp_tensor = promp_tensor.to(self.device)
        internalObservs = []
        generated = []#; generatedKnapsack = []
        for i in range(max_tokens_to_generate):
            internal_obs = promp_tensor[:,-(2*generat_link_number):,:]#
            internalObservs.append(internal_obs)
            #print('transformer ____any', torch.isnan(torch.cat(internalObservs, 0)).any())
            decoder_mask = torch.ones_like(internal_obs[:,:,0])
            decoder_mask[torch.all(internal_obs == torch.tensor(decoder_ACT, 
                                                                device=self.device), 
                                   dim=2)] = 0
            decoder_mask = torch.cat([decoder_mask]*self.actor_model.config.nhead , 0)
            memory_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2).long(), 
                                       encoder_mask.to(torch.device('cpu')).unsqueeze(1).long())
            encoder_mask_sqr = torch.matmul(encoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                            encoder_mask.to(torch.device('cpu')).unsqueeze(1))
            decoder_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                        decoder_mask.to(torch.device('cpu')).unsqueeze(1))
            
            external_obs = external_obs.to(self.device)
            encoder_mask_sqr = encoder_mask_sqr.to(self.device)
            internal_obs = internal_obs.to(self.device)
            decoder_mask = decoder_mask.to(self.device)
            memory_mask = memory_mask.to(self.device)
            next_promp, next_ = self.actor_model(external_obs, encoder_mask_sqr, 
                                             internal_obs, decoder_mask, 
                                             memory_mask)
            #print('transPPPPPP ____max', torch.min(next_promp))
            #print('transPPPPPP ____any', torch.isnan(next_promp).any())
            promp_tensor = torch.cat([promp_tensor, next_promp.unsqueeze(1)], dim=1)#torch.mean(next_, 1)
            #generatedKnapsack.append(next_.unsqueeze(1))
            generated.append(next_.unsqueeze(1))
        return torch.cat(generated,1), promp_tensor #, torch.cat(internalObservs, 0)
        
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
        done: Optional[torch.tensor] = None,
    ):
        externalObservation = torch.cat(
            [externalObservation]*(2*self.config.internal_batch), 0)
        prompt = None
        step_acts = np.zeros((0,2), dtype= int)
        step_prob = np.array([])
        actions = torch.tensor([], dtype=torch.int, device=self.device)
        probs = torch.tensor([], dtype=torch.float64, device=self.device)
        internalRewards = torch.tensor([], dtype=torch.float64, device=self.device)
        internalObservation = torch.zeros([0, 2*self.config.generat_link_number, 
                                           self.actor_model.config.output_dim], 
                                          device = self.device)
        for i in range(0, self.config.generat_link_number):
            stepGenerate, prompt = self.generate_step(externalObservation[0].unsqueeze(0), 2, 
                                                      self.config.generat_link_number, prompt)
            internalObservation = torch.cat([internalObservation, prompt[:,-(2*self.config.generat_link_number+1):-1,:]], 0)
            internalObservation = torch.cat([internalObservation, prompt[:,-(2*self.config.generat_link_number):,:]], 0)
            
            act, prob = self._choose_actions(stepGenerate, externalObservation)
            internalReward, accepted_action, accepted_prob = self.internal_reward(
                act, prob, statePrepare)
            
            actions = torch.cat([actions, act], 0)
            probs = torch.cat([probs, prob], 0)
            internalRewards = torch.cat([internalRewards, internalReward],0)
            if len(accepted_action) > 0:
                step_acts = np.append(step_acts, accepted_action, 0)
                step_prob = np.append(step_prob, accepted_prob)
            else:
                prompt = prompt[:,:-2]
            if i % self.config.internal_batch == self.config.internal_batch-1:
                values = self.critic_model(externalObservation.to(self.device))#, 
                                           #internalObservation.to(self.device))
                for _ in range(self.config.ppo_epochs):
                    dones, batches = self._generate_batch(done)
                    self._train_minibatch(2*self.config.internal_batch, externalObservation, 
                                          actions, probs, values.squeeze(), 
                                          internalRewards, batches, 
                                          internalObservation, dones)
                actions = torch.tensor([], dtype=torch.int, device=self.device)
                probs = torch.tensor([], dtype=torch.float64, device=self.device)
                internalRewards = torch.tensor([], dtype=torch.float64, device=self.device)
                internalObservation = torch.zeros([0, 2*self.config.generat_link_number, 
                                                   self.actor_model.config.output_dim], 
                                                  device = self.device)
                    
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
            n_states = 2*self.config.internal_batch
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
            
            stepGenerate, _ = self.generate_step(eo, 1, self.config.generat_link_number, 
                                                 io)
            #print(stepGenerate.size())
            #inst_dist, ks_dist = self._make_distribution(stepGenerate, 
            #                                             eo)
            #inst_dist = Categorical(inst_dist)
            #ks_dist = Categorical(ks_dist)
            new_dist = Categorical (stepGenerate)
            
            if mode == "internal":
                critic_value = self.critic_model(eo.to(self.device), io.to(self.device))
                new_log_probs = new_dist.log_prob(acts)

            else:
                critic_value = self.external_critic_model(eo.to(self.device))
                new_log_probs = new_dist.log_prob(acts[:,0]) + new_dist.log_prob(acts[:,1])


            #print('cv', critic_value.size())
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
            #print('rt', advantage[batch].unsqueeze(dim=1))
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
    