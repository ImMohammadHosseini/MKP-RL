
"""

"""
import torch
import numpy as np
from torch.optim import Adam
from typing import List, Optional

from utils.import_utils import is_torch_greater_2_0
from .src.core import (
    set_seed,  
)
from torch.distributions.categorical import Categorical
from ..src.data_structure.state_prepare import ExternalStatePrepare

from .src.base_trainer import BaseTrainer
from configs.ppo_configs import PPOConfig

class PPOTrainer(BaseTrainer):
    r"""
    this implementation is inspired by: 
        https://github.com/lvwerra/trl/blob/main/trl/trainer/ppo_trainer.py and 
        https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/ppo_torch.py
    """
    
    def __init__(
        self,
        info: List,
        config: Optional[PPOConfig] = None,
        acter_model: Optional[torch.nn.Module] = None,
        critic_model: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        acter_optimizer: Optional[torch.optim.Optimizer] = None,
        critic_optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator=None,
        num_shared_layers: Optional[int] = None,
    ):
        super().__init__(config)
        
        if not isinstance(config, PPOConfig):
            raise ValueError(f"config must be a PPOConfig, got {type(config)}")
        
        set_seed(config.seed)
        
        self.device = device
        self.softmax = torch.nn.Softmax()
        self.info = info
        '''if not isinstance(env, EnvBase):
            raise ValueError(f"env must be a torchrl.envs.EnvBase, got {type(env)}")
        else:
            self.env = env'''
        #self.accelerator
        
        self.acter_model = acter_model
        self.critic_model = critic_model
        
        if acter_optimizer is None:
            self.acter_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.acter_model.parameters()), lr=self.config.acter_lr
            )
        else:
            self.acter_optimizer = acter_optimizer
        
        if critic_optimizer is None:
            self.critic_optimizer = Adam(
                filter(lambda p: p.requires_grad, self.critic_model.parameters()), lr=self.config.critic_lr
            )
        else:
            self.critic_optimizer = critic_optimizer


    def generate_step (
        self,
        external_obs: torch.tensor,
        promp_tensor: Optional[torch.tensor] = None,   
        max_tokens_to_generate: Optional[int] = None,
    ):
        if max_tokens_to_generate is None:
            max_tokens_to_generate = self.config.internal_batch
        encoder_ACT = [0]*self.acter_model.config.input_dim
        encoder_mask = torch.ones_like(external_obs[:,:,0])
        encoder_mask[torch.all(external_obs == torch.tensor(encoder_ACT), dim=2)] = 0
        encoder_mask = torch.cat([encoder_mask]*self.acter_model.config.nhead , 0)

        decoder_ACT = [0]*self.acter_model.config.output_dim
        if promp_tensor == None:
            start_tokens = [decoder_ACT]
            prompt_tensor = torch.tensor(
                self.acter_model.pad_left(
                    sequence=start_tokens,
                    final_length=self.config.internal_batch + 1,
                    padding_token=decoder_ACT
                    ),
                dtype=torch.long
            )
            prompt_tensor = prompt_tensor.unsqueeze(dim=0)
            prompt_tensor = torch.cat([prompt_tensor]*external_obs.size(0), 0)
        
        out = prompt_tensor
        internalObservs = []
        for _ in range(max_tokens_to_generate):
            internal_obs = out[:,-(self.config.internal_batch+1):,:]
            internalObservs.append(internal_obs)
            
            decoder_mask = torch.ones_like(internal_obs[:,:,0])
            decoder_mask[torch.all(internal_obs == torch.tensor(decoder_ACT), dim=2)] = 0
            decoder_mask = torch.cat([decoder_mask]*self.config.nhead , 0)

            memory_mask = torch.matmul(decoder_mask.unsqueeze(2).long(), 
                                       encoder_mask.unsqueeze(1).long())
            encoder_mask_sqr = torch.matmul(encoder_mask.unsqueeze(2), 
                                            encoder_mask.unsqueeze(1))
            decoder_mask = torch.matmul(decoder_mask.unsqueeze(2), 
                                        decoder_mask.unsqueeze(1))

            next_ = self.acter_model(external_obs, encoder_mask_sqr, internal_obs, 
                                     decoder_mask, memory_mask)
            out = torch.cat([out, torch.mean(next_, 1).unsqueeze(1)], dim=1)
            
        return out[self.config.internal_batch+1:].squeeze(), \
            torch.cat(internalObservs, 0), \
                out[:,-(self.config.internal_batch+1):,:]
            
    def _make_distribution (
        self, 
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        generatedInstance = stepGenerate[:, :self.acter_model.config.problem_dim].cpu().detach().numpy()
        insts = externalObservation[0][1:self.instance_observation_size+1, :-1].cpu().detach().numpy()
        insts = np.append(insts, np.array([[-1]*self.acter_model.config.problem_dim]),0)
        generatedKnapsack = stepGenerate[:, self.acter_model.config.problem_dim:].cpu().detach().numpy()
        ks = externalObservation[0][self.instance_observation_size+2:-1, :-1].cpu().detach().numpy()
        ks = np.append(ks, np.array([[-1]*self.acter_model.config.problem_dim]),0)

        inst_cosin_sim = (generatedInstance @ insts.T)/(np.expand_dims(
            np.linalg.norm(generatedInstance, axis=1),0).T @ np.expand_dims(
                np.linalg.norm(insts, axis=1),0))
        ks_cosin_sim = (generatedKnapsack @ ks.T)/(np.expand_dims(
            np.linalg.norm(generatedKnapsack, axis=1),0).T @ np.expand_dims(
                np.linalg.norm(ks, axis=1),0))
        
        inst_dist = self.softmax(torch.tensor(inst_cosin_sim))
        inst_dist = Categorical(inst_dist)
        ks_dist = self.softmax(torch.tensor(ks_cosin_sim))
        ks_dist = Categorical(ks_dist)
        
        return inst_dist, ks_dist
    
    def _choose_actions (
        self, 
        stepGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        inst_dist, ks_dist = self._make_distribution(stepGenerate, 
                                                     externalObservation)
        inst_act = inst_dist.sample()
        ks_act = ks_dist.sample()
        inst_log_probs = inst_dist.log_prob(inst_act)
        ks_log_probs = ks_dist.log_prob(ks_act)
        
        acts = torch.cat([inst_act.unsqueeze(dim=1),
                          ks_act.unsqueeze(dim=1)],dim=1)
        log_probs = inst_log_probs + ks_log_probs
        return acts, log_probs
    
    def _internal_reward (
        self,
        actions: np.ndarray,
        statePrepare: ExternalStatePrepare,
    ):
        rewards = []
        for inst_act, ks_act in actions:
            if inst_act == statePrepare.instanceObsSize or \
            ks_act == statePrepare.knapsackObsSize:
                rewards.append(0)
            else:
                knapSack = statePrepare.getKnapsack(ks_act).getExpectedCap()
                eCap = knapSack.getExpectedCap()
                weight = statePrepare.getObservedInstWeight(inst_act)
                if all(weight == 0):
                    rewards.append(-int(self.info['VALUE_LOW']))
                elif all(eCap >= weight):
                    rewards.append(statePrepare.getObservedInstValue(inst_act))
                    knapSack.addExpectedCap(weight)
                else:
                    rewards.append(-int(self.info['VALUE_HIGH']))
        
        return torch.tensor(rewards)
    
    def steps (
        self,
        externalObservation: torch.tensor,
        statePrepare: ExternalStatePrepare,
    ):
        externalObservation = torch.cat(
            [externalObservation]*self.config.internal_batch, 0)
        prompt = None
        all_acts = np.zeros((0,2))#TODO check all_acts
        for i in (0, self.config.generat_link_number+1, self.config.internal_batch):
            stepGenerate, internalObservation, prompt = self.generate_step(
                externalObservation[0].unsqueeze(0), prompt)
            action, prob = self._choose_actions(stepGenerate, externalObservation)
            values = self.critic_model(externalObservation, 
                                       internalObservation)
            intervalReward = self._internal_reward(action, statePrepare)
            all_acts = np.append(all_acts, action, 0)
            for _ in range(self.config.ppo_epochs):
                self.train_minibatch(externalObservation, internalObservation, 
                                     action, prob, values, intervalReward, 
                                     self._generate_batch())#TODO we are here
                
    def _generate_batch (
        self, 
        dones: Optional[torch.tensor] = None,
    ):
        n_states = self.cofig.internal_batch
        batch_start = np.arange(0, n_states, self.cofig.ppo_batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        if dones is None:
            dones = torch.tensor([False]*n_states)
        
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
        advantage = torch.zeros(self.config.internal_batch, dtype=np.float32)
        for t in range(self.config.internal_batch-1):
            discount = 1
            a_t = 0
            for k in range(t, self.config.internal_batch-1):
                
                a_t += discount*(int(rewards[k]) + self.config.gamma*(float(vals[k+1][0]))*\
                            (1-int(dones[k])) - (float(vals[k][0])))
                discount *= self.config.gamma*self.config.gae_lambda
            advantage[t] = a_t
            
        for batch in batches:
            eo = external_obs[batch]#TODO change in observed and knapsack
            io = internal_obs[batch]
            olp = old_log_probs[batch]
            acts = actions[batch]
            
            stepGenerate, _, _ = self.generate_step(eo, io, 1)
            inst_dist, ks_dist = self._make_distribution(stepGenerate, 
                                                         eo)
            critic_value = self.critic(eo, io)
            critic_value = torch.squeeze(critic_value)
            
            new_inst_log_probs = inst_dist.log_prob(actions[:,0])
            new_ks_log_probs = ks_dist.log_prob(actions[:,1])
            new_log_probs = new_inst_log_probs + new_ks_log_probs
            prob_ratio = new_log_probs.exp() / old_log_probs.exp()
            
            weighted_probs = advantage[batch] * prob_ratio
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.config.cliprange,
                        1+self.config.cliprange)*advantage[batch]
            
            actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
            returns = advantage[batch] + vals[batch]
            critic_loss = (returns-critic_value)**2
            critic_loss = critic_loss.mean()

            total_loss = actor_loss + 0.5*critic_loss
            
            self.actor.optimizer.zero_grad()
            self.critic.optimizer.zero_grad()
            total_loss.backward()
            self.actor.optimizer.step()
            self.critic.optimizer.step()
        
        return #TODO all_acts
        
    def _early_stop(self, policykl):
        pass
    
    def gather_stats(self, stats):
        pass
    
    def batched_forward_pass(
        self,
        #model: PreTrainedModel,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
    ):
        pass
    
    def loss(
        self,
        old_logprobs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor,
        logits: torch.FloatTensor,
        vpreds: torch.FloatTensor,
        logprobs: torch.FloatTensor,
        mask: torch.LongTensor,
    ):
        pass
    
    