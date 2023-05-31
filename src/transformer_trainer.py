
"""
"""

import torch
import numpy as np
from torch.optim import Adam, Rprop
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
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
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()), 
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
        
        cnt_wait=0; best=1e9; loss_list=[]; prompt=None
        for epoch in range(self.model.config.first_train_epoc):
            losses = []
            
            for obs in data:
                self.model.train()
                self.optimizer.zero_grad()
                stepGenerate, _ = self.model.generate_step(obs[0], 100, 50, prompt)
                loss = self.get_loss(stepGenerate)
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
            
    def get_loss (self, step_generate):
        generated_dist = Categorical(step_generate)
        target = generated_dist.sample().cpu().detach().numpy()
        #target = torch.max(step_generate, 2)[1].cpu().detach().numpy()
        #print(target.shape)
        for i, acts in enumerate(target.T):
            #print(acts)

            if i % 2 == 0:
                wrong_idx = np.where(acts >= self.model.config.inst_obs_size)[0]
                acts[wrong_idx]=np.random.randint(0, self.model.config.inst_obs_size,
                                                  len(acts[wrong_idx]))
            elif i % 2 == 1:
                wrong_idx = np.where(acts < self.model.config.inst_obs_size)[0]
                acts[wrong_idx]=np.random.randint(self.model.config.inst_obs_size, 
                                                  self.model.config.max_length-3,
                                                  len(acts[wrong_idx]))
            #print(acts.shape)
            #print(len(wrong_idx))
        
        loss = self.loss_function(step_generate.transpose(1, 2), 
                                  torch.tensor(target, device=self.device))
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
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])