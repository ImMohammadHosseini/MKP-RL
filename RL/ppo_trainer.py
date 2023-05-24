
"""

"""
import torch
import numpy as np
from torch.optim import Adam
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
        device: torch.device = torch.device("cpu"),
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
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
        self.critic_model.to(self.device)
            
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
            
        self.savePath = save_path +'/RL/PPOTrainer/'
        if path.exists(self.savePath):
            self.load_models()
            
    def generate_step (
        self,
        external_obs: torch.tensor,
        promp_tensor: Optional[torch.tensor] = None,   
        max_tokens_to_generate: Optional[int] = None,
    ):
        if max_tokens_to_generate is None:
            max_tokens_to_generate = self.config.internal_batch
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
                    final_length=self.config.generat_link_number + 1,#
                    padding_token=decoder_ACT
                    ),
                dtype=torch.long
            )
            promp_tensor = promp_tensor.unsqueeze(dim=0)
            promp_tensor = torch.cat([promp_tensor]*external_obs.size(0), 0)
        
        out = promp_tensor.to(self.device)
        internalObservs = []
        for _ in range(max_tokens_to_generate):
            internal_obs = out[:,-(self.config.generat_link_number+1):,:]#
            internalObservs.append(internal_obs)
            
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
            next_ = self.actor_model(external_obs, encoder_mask_sqr, internal_obs, 
                                     decoder_mask, memory_mask)
            out = torch.cat([out, torch.nan_to_num(next_[:,0].unsqueeze(1))], dim=1)#torch.mean(next_, 1)
            
        return out[:,-max_tokens_to_generate:].squeeze(), \
            torch.cat(internalObservs, 0), \
                out
            
    def _make_distribution (
        self, 
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        generatedInstance = stepGenerate[:, :self.actor_model.config.problem_dim].cpu().detach().numpy()
        insts = externalObservation[0][1:self.actor_model.config.inst_obs_size+1,
                                       :-1].cpu().detach().numpy()
        insts = np.append(insts, np.array([[-1]*self.actor_model.config.problem_dim]),0)
        generatedKnapsack = stepGenerate[:, self.actor_model.config.problem_dim:].cpu().detach().numpy()
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
        
        return inst_dist, ks_dist
    
    def _choose_actions (
        self, 
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        inst_dist, ks_dist = self._make_distribution(stepGenerate, 
                                                     externalObservation)
        inst_dist = Categorical(inst_dist)
        ks_dist = Categorical(ks_dist)

        inst_act = inst_dist.sample()
        ks_act = ks_dist.sample()
        inst_log_probs = inst_dist.log_prob(inst_act)
        ks_log_probs = ks_dist.log_prob(ks_act)
        
        acts = torch.cat([inst_act.unsqueeze(dim=1),
                          ks_act.unsqueeze(dim=1)],dim=1)
        log_probs = inst_log_probs + ks_log_probs
        return acts, log_probs
    
    def internal_reward (
        self,
        actions: np.ndarray,
        statePrepare: ExternalStatePrepare,
    ):
        rewards = []; accepted_action = np.zeros((0,2), dtype= int)
        for inst_act, ks_act in actions:
            if inst_act == statePrepare.instanceObsSize or \
            ks_act == statePrepare.knapsackObsSize:
                rewards.append(0)
            elif inst_act < statePrepare.pad_len or \
                inst_act in accepted_action[:,0]:
                rewards.append(-int(self.info['VALUE_LOW']))
            else:
                knapSack = statePrepare.getKnapsack(ks_act)
                eCap = knapSack.getExpectedCap()
                weight = statePrepare.getObservedInstWeight(inst_act)
                '''if all(weight == 0):
                    rewards.append(-int(self.info['VALUE_LOW']))'''
                if all(eCap >= weight):
                    value = statePrepare.getObservedInstValue(inst_act)
                    rewards.append(value / np.sum(weight))
                    knapSack.removeExpectedCap(weight)
                    accepted_action = np.append(accepted_action, 
                                                [[inst_act, ks_act]], 0)
                else:
                    rewards.append(-int(self.info['VALUE_HIGH']))
        
        return torch.tensor(rewards), accepted_action
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
        done: Optional[torch.tensor] = None,
    ):
        externalObservation = torch.cat(
            [externalObservation]*self.config.internal_batch, 0)
        prompt = None
        learn_iter = 0
        step_acts = np.zeros((0,2), dtype= int)
        for i in range(0, self.config.generat_link_number+1, self.config.internal_batch):
            stepGenerate, internalObservation, prompt = self.generate_step(
                externalObservation[0].unsqueeze(0), prompt)
            action, prob = self._choose_actions(stepGenerate, externalObservation)
            
            values = self.critic_model(externalObservation.to(self.device), 
                                       internalObservation.to(self.device))
            intervalReward, accepted_action = self.internal_reward(action, statePrepare)
            step_acts = np.append(step_acts, accepted_action, 0)
            learn_iter += 1
            for _ in range(self.config.ppo_epochs):
                dones, batches = self._generate_batch(done)
                self.train_minibatch(externalObservation, internalObservation, 
                                     action, prob, values, intervalReward, 
                                     dones, batches)
        return step_acts, learn_iter
    
    def test_step (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
    ):
        prompt = None
        stepGenerate, internalObservation, prompt = self.generate_step(
            externalObservation[0].unsqueeze(0), prompt, 
            self.config.generat_link_number)
        #TODO SAMPLE MAX
        inst_dist, ks_dist = self._make_distribution(stepGenerate, 
                                                     externalObservation)
        inst_act = inst_dist.max(1)[1]
        ks_act = ks_dist.max(1)[1]
        acts = torch.cat([inst_act.unsqueeze(dim=1),
                          ks_act.unsqueeze(dim=1)],dim=1)
        _, acts = self.internal_reward(acts, statePrepare)
        
        return acts
        
    def _generate_batch (
        self, 
        done: Optional[torch.tensor] = None,
    ):
        n_states = self.config.internal_batch
        batch_start = np.arange(0, n_states, self.config.ppo_batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.config.ppo_batch_size] for i in batch_start]
        
        if done is None:
            dones = torch.tensor([False]*n_states)
        else:
            dones = torch.tensor([done]*n_states)
        
        return dones,\
                batches
    
    def train_minibatch (
        self, 
        external_obs: torch.tensor,
        internal_obs: torch.tensor,
        actions: torch.tensor,
        old_log_probs: torch.tensor,
        vals: torch.tensor,
        rewards: torch.tensor,
        dones: torch.tensor,
        batches: List,
    ):
        advantage = torch.zeros(self.config.internal_batch, dtype=torch.float32, device=self.device)
        for t in range(self.config.internal_batch-1):
            discount = 1
            a_t = 0
            for k in range(t, self.config.internal_batch-1):
                
                a_t += discount*(int(rewards[k]) + self.config.gamma*(float(vals[k+1][0]))*\
                            (1-int(dones[k])) - (float(vals[k][0])))
                discount *= self.config.gamma*self.config.gae_lambda
            advantage[t] = a_t
            
        for batch in batches:
            eo = external_obs[batch]
            io = internal_obs[batch]
            olp = old_log_probs[batch]
            acts = actions[batch]
            
            stepGenerate, _, _ = self.generate_step(eo, io, 1)
            inst_dist, ks_dist = self._make_distribution(stepGenerate, 
                                                         eo)
            critic_value = self.critic_model(eo.to(self.device), io.to(self.device))
            critic_value = torch.squeeze(critic_value)
            
            new_inst_log_probs = inst_dist.log_prob(acts[:,0])
            new_ks_log_probs = ks_dist.log_prob(acts[:,1])
            new_log_probs = new_inst_log_probs + new_ks_log_probs
            prob_ratio = (new_log_probs.exp() / olp.exp()).to(self.device)
            
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                        1+self.config.cliprange)*advantage[batch]
            
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + vals[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()
            
            actor_loss_leaf = Variable(actor_loss.data, requires_grad=True)
            critic_loss_leaf = Variable(critic_loss.data, requires_grad=True)
            
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            actor_loss_leaf.backward()
            self.actor_optimizer.step()
            critic_loss_leaf.backward(retain_graph=True)
            self.critic_optimizer.step()
            
    def load_models (self):
        file_path = self.savePath + "/" + self.actor_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.actor_model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        file_path = self.savePath + "/" + self.critic_model.name + ".ckpt"
        checkpoint = torch.load(file_path)
        self.critic_model.load_state_dict(checkpoint['model_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
         
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
    