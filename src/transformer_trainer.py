
"""
"""

import torch
import numpy as np
from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from torch.distributions.categorical import Categorical
from .models.transformer import TransformerKnapsack
from .data_structure.state_prepare import ExternalStatePrepare
from configs.transformers_model_configs import TransformerKnapsackConfig
from data.dataProducer import multipleKnapSackData, multiObjectiveDimentional
from os import path, makedirs


class TransformerTrainer:

    def __init__(
        self,
        save_path: str,
        model: Optional[torch.nn.Module] = None,
        model_config: Optional[TransformerKnapsackConfig] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ):
        if model is None and model_config is None:
            raise ValueError(f"send config to set new model")
        
        self.device = device
        if model is None:
            self.model = TransformerKnapsack(model_config, self.device)
        else: self.model = model 
        
        if optimizer is None:
            self.optimizer = Adam(self.model.parameters(), 
                lr=self.model.config.first_train_lr
            )
        else: self.optimizer = optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        
        self.savePath = save_path +'/transformer/first_train/'
            
    def train (
        self, 
        variation: str, 
        kps_n: int, 
        instances_n: int, 
        info,
    ):
        data = self.get_data (variation, kps_n, instances_n, info)
        
        cnt_wait=0; best=1e9; loss_list=[]; 
        for epoch in range(self.model.config.first_train_epoc):
            losses = []
            for obs in data:
                rand_indx_instance = np.random.randint(low=1, high=instances_n+1,
                                                       size=(self.model.config.first_train_batch_size,50))
                rand_indx_knapsack = np.random.randint(low=instances_n+2, 
                                                       high=self.model.config.max_length-1,
                                                       size=(self.model.config.first_train_batch_size,50))
                target = np.zeros((self.model.config.first_train_batch_size,100))                                      
                target[:, ::2]=rand_indx_instance
                target[:, 1::2]=rand_indx_knapsack
                start_prompt_length = np.random.randint(0, 101)
                for prompt_length in range(start_prompt_length, 99): 
                    #print(prompt_length)
                    prompt = torch.cat([obs[0][i][target[i][:prompt_length]].unsqueeze(0) for i in range(
                        self.model.config.first_train_batch_size)], 0)

                    self.model.train()
                    self.optimizer.zero_grad()
                    stepGenerate, _ = self.model.generate_step(obs[0], 1, 50, prompt)
                    if prompt_length % 2 == 1:
                        loss_target = target[:,prompt_length+1] - 1
                    else:
                        loss_target = target[:,prompt_length+1] - 2
                    loss = self.get_loss(stepGenerate, torch.tensor(
                        loss_target, dtype=torch.long, device=self.device), prompt_length)
                    loss.backward()
                    self.optimizer.step()
                    #float(loss['loss'])
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
                break
            
    def get_loss (self, step_generate, target, prompt_length):
        generated_dist = Categorical(step_generate)
        acts = generated_dist.sample().cpu().detach().numpy()#(torch.max(step_generate, 2)[1]).cpu().detach().numpy() 
        #print(acts)
        #print(target)
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
        print(len(wrong_idx))
        
        loss = torch.nan_to_num(self.loss_function(step_generate.transpose(1, 2), 
                                torch.tensor(target.unsqueeze(1), device=self.device)))
        '''if len(wrong_idx) > 0:
            print (prompt_length)
            print(loss)
            print(acts)'''
        ##print(loss)
        
        return loss        
    
    def get_data (
        self, 
        variation: str, 
        kps_n: int, 
        instances_n: int, 
        info,
    ):
        statePrepare = ExternalStatePrepare(k_obs_size=self.model.config.knapsack_obs_size, 
                                            i_obs_size=self.model.config.inst_obs_size)
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
            statePrepare.normalizeData(info['CAP_HIGH'], info['VALUE_HIGH'])
            statePrepare.reset()
            caps, weightValues = statePrepare.getObservation()
            ACT = np.zeros((1,self.model.config.input_dim))
            data = torch.cat([data, torch.tensor(np.append(ACT, np.append(np.append(
                weightValues, ACT, axis=0), np.append(caps, ACT, axis=0),axis=0),
                axis=0), dtype=torch.float32, device=self.device).unsqueeze(dim=0)], 0)
        data = DataLoader(TensorDataset(data), 
                          batch_size=self.model.config.first_train_batch_size,
                          shuffle=True)
        return data
            
    def save_model (self):
        if not path.exists(self.savePath):
            makedirs(self.savePath)
        file_path = self.savePath + "/" + self.model.name + ".ckpt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, 
            file_path)
    
    def load_model (self):
        file_path = self.savePath + "/" + self.model.name + ".ckpt"
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
