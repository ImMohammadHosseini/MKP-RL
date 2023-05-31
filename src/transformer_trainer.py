
"""
"""
import torch
import numpy as np
from torch.optim import Adam, Rprop
from typing import List, Optional
from src.models.transformer import TransformerKnapsack
from data_structure.state_prepare import ExternalStatePrepare
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
        deta = self.get_data (variation, kps_n, instances_n, info)
        for epoch in range(self.model.config.first_train_epoc):
            losses = []
            
    def loss (self):
        pass
    
    def get_data (
        self, 
        variation: str, 
        kps_n: int, 
        instances_n: int, 
        info,
    ):
        statePrepare = ExternalStatePrepare(k_obs_size=self.model.config.knapsack_obs_size, 
                                            i_obs_size=self.model.config.inst_obs_size)
        data = torch.zeros((0, self.model.config.max_length, self.model.config.input_dim))
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
            caps, weightValues = statePrepare.getObservation()
            ACT = np.zeros((1,self.model.config.input_dim))
            data = torch.cat([data, torch.tensor(np.append(ACT, np.append(np.append(
                weightValues, ACT, axis=0), np.append(caps, ACT, axis=0),axis=0),
                axis=0),dtype=torch.float32, device=self.device).unsqueeze(dim=0)], 0)
        return data
            
    def save_model (self):
        pass
    
    def load_model (self):
        pass