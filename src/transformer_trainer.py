
"""
"""
import sys
sys.path.append('MKP-RL/solve_algorithms/')

import torch
import numpy as np
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from torch.distributions.categorical import Categorical
from .models.transformer import TransformerKnapsack
from .data_structure.state_prepare import ExternalStatePrepare
from configs.transformers_model_configs import TransformerKnapsackConfig
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from os import path, makedirs
from solve_algorithms.greedy_select import GreedySelect

class TransformerTrainer:

    def __init__(
        self,
        generate_link_number: int,
        statePrepares: np.ndarray,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ):
        self.generate_link_number = generate_link_number
        self.statePrepares = statePrepares
        self.device = device
        self.model = model 
        
        if optimizer is None:
            self.optimizer = RMSprop(self.model.parameters(), 
                lr=0.001#self.model.config.first_train_lr
            )
        else: self.optimizer = optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        
            
    def train (
        self, 
        num_instance,
    ):
        states, targets = self.data_process(num_instance)
        
        print(states.size())
        print(targets.size())
        cnt_wait=0; best=1e9; loss_list=[]; 
        
        for epoch in range(20):#self.model.config.first_train_epoc):
            losses = []
            for sts, tgt in zip(states, targets):
                print(sts.size())
                print(tgt.size())
                print(self.dd)

            
            
            '''
                rand_indx_instance = np.random.randint(low=1, high=instances_n+1,
                                                       size=(self.model.config.first_train_batch_size,50))
                rand_indx_knapsack = np.random.randint(low=instances_n+2, 
                                                       high=self.model.config.max_length-1,
                                                       size=(self.model.config.first_train_batch_size,50))
                target = np.zeros((self.model.config.first_train_batch_size,100))                                      
                target[:, ::2]=rand_indx_instance
                target[:, 1::2]=rand_indx_knapsack
                start_prompt_length = np.random.randint(0, 101)
                p_loss = []; loss = torch.tensor(0, device=self.device, dtype=torch.float)
                for prompt_length in range(start_prompt_length, 99): 
                    self.model.train()
                    self.optimizer.zero_grad()
                    #print(prompt_length)
                    prompt = torch.cat([obs[0][i][target[i][:prompt_length]].unsqueeze(0) for i in range(
                        self.model.config.first_train_batch_size)], 0)

                    
                    stepGenerate, _ = self.model.generate_step(obs[0], 1, 50, prompt)
                    if prompt_length % 2 == 1:
                        loss_target = target[:,prompt_length+1] - 1
                        #loss = self.get_loss(stepGenerate, torch.tensor(
                        #    loss_target, dtype=torch.long, device=self.device), prompt_length)
                    else:
                        loss_target = target[:,prompt_length+1] - 2
                    loss = self.get_loss(stepGenerate, torch.tensor(
                        loss_target, dtype=torch.long, device=self.device), prompt_length)
                    #loss.requires_grad = True
                    loss.backward()
                    self.optimizer.step()
                    losses.append(float(loss))
                
            epoch_loss = sum(losses) / len(losses)
            loss_list.append(epoch_loss)
            print(f'Epoch: {epoch:03d}, Loss: {epoch_loss:.4f}')
            if epoch_loss < best:
                best = epoch_loss
                cnt_wait = 0
                #self.save_model()
            else:
                cnt_wait += 1
            
            if cnt_wait == 20:
                print('Early stopping!')
                break'''
            
    def get_loss (self, step_generate, target, prompt_length):
        generated_dist = Categorical(step_generate)
        acts = generated_dist.sample().cpu().detach().numpy()#(torch.max(step_generate, 2)[1]).cpu().detach().numpy() 
        print(acts)
        print(target)
        if prompt_length % 2 == 1:
            wrong_idx = np.where(acts >= self.model.config.inst_obs_size)[0]
            
        else :
            wrong_idx = np.where(acts < self.model.config.inst_obs_size)[0]
        #print(wrong_idx)
        #step_generate = step_generate[wrong_idx]
        #target = target[wrong_idx]
        #target = generated_dist.sample().cpu().detach().numpy()
        #target = torch.max(step_generate, 2)[1].cpu().detach().numpy()
        #for i, acts in enumerate(target.T):
            #print(acts)
        #print(step_generate.size())
        #print(target)
        '''if i % 2 == 0:
            wrong_idx = np.where(acts >= self.model.config.inst_obs_size)[0]
            acts[wrong_idx]=np.random.randint(0, self.model.config.inst_obs_size,
                                              len(acts[wrong_idx]))
        elif i % 2 == 1:
            wrong_idx = np.where(acts < self.model.config.inst_obs_size)[0]
            acts[wrong_idx]=np.random.randint(self.model.config.inst_obs_size, 
                                              self.model.config.max_length-3,
                                              len(acts[wrong_idx]))
            #print(acts.shape)'''
        #print(len(wrong_idx))
        
        loss1 = torch.nan_to_num(self.loss_function(step_generate.transpose(1, 2), 
                                torch.tensor(target.unsqueeze(1), device=self.device)))
        loss2 = torch.tensor(float(len(wrong_idx)), device=self.device)
        loss = loss1
        print(loss)

        '''if len(wrong_idx) > 0:
            print (prompt_length)
            print(loss)
            print(acts)'''
        ##print(loss)
        
        return loss        
    def data_process(
        self,
        num_instance,
    ) -> torch.Tensor:
        states = torch.zeros((0, self.model.config.max_length, self.model.config.input_encode_dim), 
                           dtype=torch.float32, device=self.device)
        targets = torch.zeros((0, self.generate_link_number, self.model.config.input_decode_dim), 
                           dtype=torch.float32, device=self.device)
        SOD = np.array([[1.]*self.model.config.input_encode_dim])
        EOD = np.array([[2.]*self.model.config.input_encode_dim])
        decode_PAD = np.array([[0.]*self.model.config.input_decode_dim])
        print('l')
        for statePrepare in self.statePrepares:
            greedy_select = GreedySelect(num_instance, statePrepare)
            _, actss = greedy_select.test_step()
            #for i in range (min(len(actss), self.generate_link_number)):
            #    print(i)
            #    statePrepare.reset()
                #print(actss[:i])
            #    _ = statePrepare.changeNextState(actss[:i])
            #    greedy_select = GreedySelect(num_instance, statePrepare)
            #    _, new_acts = greedy_select.test_step()
                
            stateCaps, stateWeightValues = statePrepare.getObservation()
            state = np.append(np.append(SOD, np.append(stateWeightValues, EOD, 
                                                       axis=0),axis=0), 
                              np.append(stateCaps, EOD, axis=0),axis=0)
            states = torch.cat([states, torch.tensor(state, dtype=torch.float32, 
                                                     device=self.device).unsqueeze(dim=0)], 0)

            state_target = torch.zeros((0, self.model.config.input_decode_dim), 
                                       dtype=torch.float32, device=self.device)
            for index in range(min(len(actss), self.generate_link_number)):
                inst_act = int(actss[index][0])
                ks_act = statePrepare.getRealKsAct(int(actss[index][1]))
                ks = torch.tensor(statePrepare.getObservedKS(ks_act),
                                  device=self.device, dtype=torch.float32).unsqueeze(0)
                inst = torch.tensor(statePrepare.getObservedInst(inst_act), 
                                    device=self.device, dtype=torch.float32).unsqueeze(0)
                state_target = torch.cat([state_target, torch.cat([inst, ks],1)], 0)
            if state_target.size(0) < self.generate_link_number:
                repeat_size = self.generate_link_number - state_target.size(0)
                state_target = torch.cat([state_target, torch.tensor(np.repeat(
                    decode_PAD, repeat_size, axis=0), device=self.device, 
                    dtype=torch.float32)], 0)
            targets = torch.cat([targets, state_target.unsqueeze(0)],0)
        return states, targets
    
    '''def get_data (
        self,
    ):
        
        
        data = torch.zeros((0, self.model.config.max_length, self.model.config.input_dim), 
                           dtype=torch.float32, device=self.device)
        for bs in range(3*self.model.config.first_train_batch_size):
            if variation == 'multipleKnapsack':
                pass
            elif variation == 'multipleKnapsack':
                pass
            elif variation == 'multiObjectiveDimentional':
                c, w, v = multiObjectiveDimentional(self.model.config.problem_dim, 
                                                    kps_n, instances_n, info)
            statePrepare.set_new_problem(c, w, v)
            statePrepare.normalizeData(400, info['VALUE_HIGH'])#info['CAP_HIGH']
            statePrepare.reset()
            caps, weightValues = statePrepare.getObservation()
            ACT = np.zeros((1,self.model.config.input_dim))
            data = torch.cat([data, torch.tensor(np.append(ACT, np.append(np.append(
                weightValues, ACT, axis=0), np.append(caps, ACT, axis=0),axis=0),
                axis=0), dtype=torch.float32, device=self.device).unsqueeze(dim=0)], 0)
        data = DataLoader(TensorDataset(data), 
                          batch_size=self.model.config.first_train_batch_size,
                          shuffle=True)
        return data'''
            
    