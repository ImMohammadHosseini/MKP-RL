
"""

"""
import torch
import math
import numpy as np
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from .positional_encoding import PositionalEncoding
from typing import Optional


class TransformerKnapsack (nn.Module):
    def __init__(
        self,
        config,
        output_type,
        device: torch.device = torch.device("cpu"),
        name = 'transformer',
    ):
        super().__init__()
        self.config = config
        self.output_type = output_type
        self.name = name
        self.device = device
        self.generate_link_number = self.config.link_number
        
        self.en_embed = nn.Linear(self.config.input_encode_dim, self.config.output_dim, device=self.device)
        self.de_embed = nn.Linear(self.config.input_decode_dim, self.config.output_dim, device=self.device)

        self.en_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     self.config.max_length,
                                                     self.device)
        self.de_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     self.generate_link_number+1,
                                                     self.device)
        
        encoder_layers = TransformerEncoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.encoder = TransformerEncoder(
            encoder_layers, self.config.num_encoder_layers
        ).to(self.device)
        decoder_layers = TransformerDecoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.decoder = TransformerDecoder(
            decoder_layers, self.config.num_decoder_layers
        ).to(self.device)
        
        if self.output_type == 'type1':
            pass
        elif self.output_type == 'type2':  
            self.instance_outer = nn.Linear(self.config.output_dim//2, self.config.inst_obs_size, device=self.device)
            self.knapsack_outer = nn.Linear(self.config.output_dim//2, self.config.knapsack_obs_size, device=self.device)
        elif self.output_type == 'type3':
            self.outer = nn.Linear(self.config.output_dim, 
                                   self.config.inst_obs_size*self.config.knapsack_obs_size, device=self.device)
            
        
        self.softmax = nn.Softmax(dim=-1)
    
    def generateOneStep (
        self,
        external_obs: torch.tensor,
        step: int,
        promp_tensor: Optional[torch.tensor] = None,   
    ):
        SOD1 = [1]*self.config.input_decode_dim
        PAD1 = [0]*self.config.input_decode_dim

        if promp_tensor == None:
            start_tokens = [[SOD1]]*external_obs.size(0)
            promp_tensor = torch.tensor(
                self.padding(
                    sequence=start_tokens,
                    final_length=
                    self.generate_link_number+1, 
                    padding_token=PAD1
                    ),
                dtype=torch.float,
                device=self.device
            )
            
        else: 
            promp_tensor = promp_tensor.to(self.device)
        internal_obs = promp_tensor
        
        tgt_mask = torch.tril(torch.ones(self.config.nhead*internal_obs.size(0), 
                                            self.generate_link_number+1,
                                            self.generate_link_number+1) == 1) 
        tgt_mask = tgt_mask.float() 
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf'))
        tgt_mask = tgt_mask.masked_fill(tgt_mask == 1, float(0.0))
        tgt_mask = tgt_mask.to(self.device)
        
        '''decoder_padding_mask = torch.ones_like(internal_obs[:,:,0])
        decoder_padding_mask[torch.all(internal_obs == torch.tensor(PAD1, 
                                                                    device=self.device), 
                                       dim=2)] = 0
        decoder_padding_mask = decoder_padding_mask.float() 
        decoder_padding_mask = decoder_padding_mask.masked_fill(decoder_padding_mask == 0, float('-inf'))
        decoder_padding_mask = decoder_padding_mask.masked_fill(decoder_padding_mask == 1, float(0.0))
        ''' 
        decoder_padding_mask = None
        
        external_obs = external_obs.to(self.device)
        internal_obs = internal_obs.to(self.device)
        
        
        next_instance, next_ks = self.forward(external_obs,
                                              internal_obs, tgt_mask, 
                                              decoder_padding_mask,
                                              step)
    
        return next_instance, next_ks, promp_tensor
        
    def forward (
        self,
        external_obs: torch.tensor,
        internal_obs: torch.tensor,
        tgt_mask: torch.tensor,
        decoder_padding_mask: Optional[torch.tensor] = None,
        step: Optional[int] = None,
    ):
        external_obs = external_obs.to(torch.float32)
        internal_obs = internal_obs.to(torch.float32)
        
        ext_embedding = self.en_embed(external_obs)* math.sqrt(self.config.output_dim)
        int_embedding = self.de_embed(internal_obs)* math.sqrt(self.config.output_dim)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        encod = self.encoder(self.en_position_encode(ext_embedding))

        transformer_out = self.decoder(self.de_position_encode(int_embedding), 
                                       encod, tgt_mask=tgt_mask, 
                                       tgt_key_padding_mask=decoder_padding_mask)
        
        #print(step)
        #print(self.config.output_dim)
        #print(transformer_out.size())
        pos = torch.cat([step.unsqueeze(0)]*self.config.output_dim,0).T.unsqueeze(1).to(self.device)
        out = transformer_out.gather(1,pos).squeeze(1)
        if self.output_type == 'type1':
            return self._make_distribution(out[:,:self.config.output_dim//2],
                                                     out[:,self.config.output_dim//2:], 
                                                     external_obs)
        elif self.output_type == 'type2':
            return self.softmax(self.instance_outer(out[:,:self.config.output_dim//2])), \
                self.softmax(self.knapsack_outer(out[:,self.config.output_dim//2:]))
                
        elif self.output_type == 'type3':
            return self.softmax(self.outer(out)), None
        
        
        
    def _make_distribution (
        self, 
        instGenerate: torch.tensor, 
        ksGenerate: torch.tensor, 
        externalObservation: torch.tensor,
    ):
        SOD = [1]*self.config.input_encode_dim
        inst_dist=[]; ks_dist=[]
        for index in range(externalObservation.size(0)):
            generatedInstance = np.expand_dims(instGenerate[index].cpu().detach().numpy(),0)
            insts = externalObservation[index][:self.config.inst_obs_size+1,:-1].cpu().detach().numpy()
            insts = insts[int(np.unique(np.where(insts == SOD)[0]))+1:]
            
            generatedKnapsack = np.expand_dims(ksGenerate[index].cpu().detach().numpy(),0)
            ks = externalObservation[index][self.config.inst_obs_size+2:-1,:-1].cpu().detach().numpy()
            inst_cosin_sim = (generatedInstance @ insts.T)/(np.expand_dims(
                np.linalg.norm(generatedInstance),0).T @ np.expand_dims(
                    np.linalg.norm(insts, axis=1),0))
            ks_cosin_sim = (generatedKnapsack @ ks.T)/(np.expand_dims(
                np.linalg.norm(generatedKnapsack),0).T @ np.expand_dims(
                    np.linalg.norm(ks, axis=1),0))
            inst_dist.append(self.softmax(torch.nan_to_num(torch.tensor(inst_cosin_sim))))
            ks_dist.append(self.softmax(torch.nan_to_num(torch.tensor(ks_cosin_sim))))
        
        inst_dist = torch.cat(inst_dist, 0)
        ks_dist = torch.cat(ks_dist, 0)
        return inst_dist, ks_dist
  
    def padding(self, sequence, final_length, padding_token):
        pads = [[padding_token] * (final_length - len(sequence[:][0]))] * len(sequence)
        return [sequence[i]+pads[i] for i in range (len(pads))]
    

