
"""
"""
import sys
sys.path.append('MKP-RL/solve_algorithms/')

import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from solve_algorithms.greedy_select import GreedySelect
from solve_algorithms.random_select import RandomSelect

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
            self.optimizer = Adam(self.model.parameters(), 
                lr=0.01#self.model.config.first_train_lr
            )
        else: self.optimizer = optimizer
        self.loss_function = torch.nn.CrossEntropyLoss()
        
            
    def train (
        self, 
        num_instance,
        batch_size=4,
    ):
        data = self.data_process(num_instance, batch_size)
        
        nopeak_mask = torch.tril(torch.ones(self.model.config.nhead*batch_size, 
                                     self.generate_link_number+1, 
                                     self.generate_link_number+1) == 1) 
        nopeak_mask = nopeak_mask.float()
        nopeak_mask = nopeak_mask.masked_fill(nopeak_mask == 0, float('-inf')) 
        nopeak_mask = nopeak_mask.masked_fill(nopeak_mask == 1, float(0.0))
        nopeak_mask = nopeak_mask.to(self.device)
        cnt_wait=0; best=1e9; loss_list=[]; 
        for epoch in range(0):#self.model.config.first_train_epoc):
            losses = []
            for sts, tgt, inst_label, ks_label in data:
                
                self.model.train()
                self.optimizer.zero_grad()
                #print(sts.size())
                #print(tgt.size())
                instances, knapsacks = self.model(sts, tgt, nopeak_mask, 
                                                  mode='transformer_train')
                loss = 0
                for loss_term in range(batch_size):
                    loss += self.get_loss(instances[loss_term], inst_label[loss_term])
                    loss += self.get_loss(knapsacks[loss_term], ks_label[loss_term])
                loss = loss/batch_size
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
                break
            
    def get_loss (self, out, target):
        up_index = min(np.where(target.cpu().detach().numpy() == -1)[0], 
                       default=self.generate_link_number)
        loss = self.loss_function(out[:up_index].to(self.device), 
                                  target[:up_index].type(torch.LongTensor).to(self.device))
        return loss
    
    def data_process(
        self,
        num_instance,
        batch_size,
    ) -> torch.Tensor:
        SOD = np.array([1.]*self.model.config.input_encode_dim)
        SOD_decoder = np.array([[1.]*self.model.config.input_decode_dim])
        EOD = np.array([[2.]*self.model.config.input_encode_dim])
        PAD_decode = np.array([[0.]*self.model.config.input_decode_dim])

        states = torch.zeros((0, self.model.config.max_length, self.model.config.input_encode_dim), 
                           dtype=torch.float32, device=self.device)
        targets = torch.zeros((0, self.generate_link_number+1, self.model.config.input_decode_dim), 
                           dtype=torch.float32, device=self.device)

        all_ks_labels = []
        all_instance_labels = []
        
        for statePrepare in self.statePrepares:
            greedy_select = GreedySelect(num_instance, statePrepare)
            _, test_acts = greedy_select.test_step()
            for i in range(len(test_acts)-3):
                greedy_select.reset()
                _, _ = greedy_select.test_step(i)
                _, acts = greedy_select.test_step()
                
                stateCaps, stateWeightValues = greedy_select.getObservation()
                
                stateWeightValues_encode = np.insert(stateWeightValues, statePrepare.pad_len, 
                                              SOD, axis=0)
                
                state = np.append(np.append(stateWeightValues_encode, EOD, axis=0), 
                                  np.append(stateCaps, EOD, axis=0),axis=0)
                states = torch.cat([states, torch.tensor(state, dtype=torch.float32, 
                                                         device=self.device).unsqueeze(dim=0)], 0)
                
                state_target = torch.tensor(SOD_decoder, dtype=torch.float32, 
                                            device=self.device)

                ks_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                inst_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                for index in range(min(len(acts), self.generate_link_number)):
                    inst_labels[index] = int(acts[index][0])
                    ks_labels[index] = int(acts[index][1])
                    inst_act = int(acts[index][0])
                    ks_act = greedy_select.statePrepare.getRealKsAct(int(acts[index][1]))
                    ks = torch.tensor(stateCaps[ks_act], device=self.device, 
                                      dtype=torch.float32).unsqueeze(0)
                    inst = torch.tensor(stateWeightValues[inst_act], device=self.device, 
                                        dtype=torch.float32).unsqueeze(0)
                    state_target = torch.cat([state_target, torch.cat([inst, ks],1)], 0)
                if state_target.size(0) < self.generate_link_number+1:
                    repeat_size = (self.generate_link_number+1) - state_target.size(0)
                    state_target = torch.cat([state_target, torch.tensor(np.repeat(
                        PAD_decode, repeat_size, axis=0), device=self.device, 
                        dtype=torch.float32)], 0)
                all_instance_labels.append(inst_labels.unsqueeze(0))
                all_ks_labels.append(ks_labels.unsqueeze(0))
                targets = torch.cat([targets, state_target.unsqueeze(0)],0)
        
        for statePrepare in self.statePrepares:
            for _ in range(30):
                random_select = RandomSelect(statePrepare)
                _, test_acts = random_select.test_step_randomInst_ksFirstFit()
                for i in range(len(test_acts)-3):
                    random_select.reset()
                    _, _ = random_select.test_step_randomInst_ksFirstFit(i)
                    _, acts = random_select.test_step_randomInst_ksFirstFit()
                    
                    stateCaps, stateWeightValues = random_select.getObservation()
                    
                    stateWeightValues_encode = np.insert(stateWeightValues, statePrepare.pad_len, 
                                                  SOD, axis=0)
                    
                    state = np.append(np.append(stateWeightValues_encode, EOD, axis=0), 
                                      np.append(stateCaps, EOD, axis=0),axis=0)
                    states = torch.cat([states, torch.tensor(state, dtype=torch.float32, 
                                                             device=self.device).unsqueeze(dim=0)], 0)
                    
                    state_target = torch.tensor(SOD_decoder, dtype=torch.float32, 
                                                device=self.device)
    
                    ks_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                    inst_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                    for index in range(min(len(acts), self.generate_link_number)):
                        inst_labels[index] = int(acts[index][0])
                        ks_labels[index] = int(acts[index][1])
                        inst_act = int(acts[index][0])
                        ks_act = random_select.statePrepare.getRealKsAct(int(acts[index][1]))
                        ks = torch.tensor(stateCaps[ks_act], device=self.device, 
                                          dtype=torch.float32).unsqueeze(0)
                        inst = torch.tensor(stateWeightValues[inst_act], device=self.device, 
                                            dtype=torch.float32).unsqueeze(0)
                        state_target = torch.cat([state_target, torch.cat([inst, ks],1)], 0)
                    if state_target.size(0) < self.generate_link_number+1:
                        repeat_size = (self.generate_link_number+1) - state_target.size(0)
                        state_target = torch.cat([state_target, torch.tensor(np.repeat(
                            PAD_decode, repeat_size, axis=0), device=self.device, 
                            dtype=torch.float32)], 0)
                    all_instance_labels.append(inst_labels.unsqueeze(0))
                    all_ks_labels.append(ks_labels.unsqueeze(0))
                    targets = torch.cat([targets, state_target.unsqueeze(0)],0)
        
        for statePrepare in self.statePrepares:
            for _ in range(30):
                random_select = RandomSelect(statePrepare)
                _, test_acts = random_select.test_step_two_random()
                for i in range(len(test_acts)-3):
                    random_select.reset()
                    _, _ = random_select.test_step_two_random(i)
                    _, acts = random_select.test_step_two_random()
                    
                    stateCaps, stateWeightValues = random_select.getObservation()
                    
                    stateWeightValues_encode = np.insert(stateWeightValues, statePrepare.pad_len, 
                                                  SOD, axis=0)
                    
                    state = np.append(np.append(stateWeightValues_encode, EOD, axis=0), 
                                      np.append(stateCaps, EOD, axis=0),axis=0)
                    states = torch.cat([states, torch.tensor(state, dtype=torch.float32, 
                                                             device=self.device).unsqueeze(dim=0)], 0)
                    
                    state_target = torch.tensor(SOD_decoder, dtype=torch.float32, 
                                                device=self.device)
    
                    ks_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                    inst_labels = torch.tensor([-1]*self.generate_link_number, dtype=torch.int)
                    for index in range(min(len(acts), self.generate_link_number)):
                        inst_labels[index] = int(acts[index][0])
                        ks_labels[index] = int(acts[index][1])
                        inst_act = int(acts[index][0])
                        ks_act = random_select.statePrepare.getRealKsAct(int(acts[index][1]))
                        ks = torch.tensor(stateCaps[ks_act], device=self.device, 
                                          dtype=torch.float32).unsqueeze(0)
                        inst = torch.tensor(stateWeightValues[inst_act], device=self.device, 
                                            dtype=torch.float32).unsqueeze(0)
                        state_target = torch.cat([state_target, torch.cat([inst, ks],1)], 0)
                    if state_target.size(0) < self.generate_link_number+1:
                        repeat_size = (self.generate_link_number+1) - state_target.size(0)
                        state_target = torch.cat([state_target, torch.tensor(np.repeat(
                            PAD_decode, repeat_size, axis=0), device=self.device, 
                            dtype=torch.float32)], 0)
                    all_instance_labels.append(inst_labels.unsqueeze(0))
                    all_ks_labels.append(ks_labels.unsqueeze(0))
                    targets = torch.cat([targets, state_target.unsqueeze(0)],0)
                                
                
        all_instance_labels = torch.cat(all_instance_labels, 0)
        all_ks_labels = torch.cat(all_ks_labels, 0)
        tensorDataset = TensorDataset(states, targets, all_instance_labels, all_ks_labels)
        data = DataLoader(tensorDataset, batch_size=batch_size, shuffle=True, 
                          drop_last=True)
        return data
    