
"""

"""
import torch
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
        device: torch.device = torch.device("cpu"),
    ):
        #torch.autograd.set_detect_anomaly(True)
        super().__init__()
        self.name = 'transformer'
        self.config = config
        self.device = device
        
        self.en_embed = nn.Linear(self.config.input_dim, self.config.output_dim)
        self.de_embed = nn.Linear(self.config.input_dim, self.config.output_dim)

        self.position_encode = PositionalEncoding(self.config.output_dim, 
                                                  self.config.max_length,
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
        )
        decoder_layers = TransformerDecoderLayer(
            d_model= self.config.output_dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.d_hid,
            dropout=self.config.dropout,
            batch_first=self.config.batch_first,
        )
        self.decoder = TransformerDecoder(
            decoder_layers, self.config.num_decoder_layers
        )
        
        self.instance_outer = nn.Linear(self.config.output_dim//2, self.config.inst_obs_size)
        self.knapsack_outer = nn.Linear(self.config.output_dim//2, self.config.knapsack_obs_size-3)

        self.softmax = nn.Softmax()
    
    def generate_step (
        self,
        external_obs: torch.tensor,
        max_tokens_to_generate: int,
        generat_link_number: int,
        promp_tensor: Optional[torch.tensor] = None,   
        
    ):
        encoder_ACT = [0]*self.config.input_dim
        encoder_mask = torch.zeros_like(external_obs[:,:,0], device=self.device)
        encoder_mask[torch.all(external_obs == torch.tensor(encoder_ACT, 
                                                            device=self.device), 
                               dim=2)] = 1
        encoder_mask = torch.cat([encoder_mask]*self.config.nhead , 0)

        decoder_ACT = [0]*self.config.output_dim
        if promp_tensor == None:
            start_tokens = [[decoder_ACT]]*external_obs.size(0)
            nopeak_mask = np.triu(np.ones((self.config.nhead, generat_link_number, 
                                           generat_link_number)), 
                                  k=1).astype('uint8')
        else: 
            start_tokens = promp_tensor.tolist()
            nopeak_mask = np.ones((self.config.nhead, generat_link_number, 
                                   generat_link_number))
        promp_tensor = torch.tensor(
            self.pad_left(
                sequence=start_tokens,
                final_length=generat_link_number,#2 * 
                padding_token=decoder_ACT
                ),
            dtype=torch.float
        )

        nopeak_mask = torch.from_numpy(nopeak_mask) == 0
        #promp_tensor = promp_tensor.unsqueeze(dim=0)
        #promp_tensor = torch.cat([promp_tensor]*external_obs.size(0), 0)

        promp_tensor = promp_tensor.to(self.device)
        internalObservs = []
        generatedInstance = []; generatedKnapsack = []
        for i in range(max_tokens_to_generate):
            internal_obs = promp_tensor[:,-(generat_link_number):,:]#2 * 
            internalObservs.append(internal_obs)

            decoder_mask = torch.zeros_like(internal_obs[:,:,0])
            decoder_mask[torch.all(internal_obs == torch.tensor(decoder_ACT, 
                                                                device=self.device), 
                                   dim=2)] = 1

            decoder_mask = torch.cat([decoder_mask]*self.config.nhead , 0)
            memory_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2).long(), 
                                       encoder_mask.to(torch.device('cpu')).unsqueeze(1).long())
            encoder_mask_sqr = torch.matmul(encoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                            encoder_mask.to(torch.device('cpu')).unsqueeze(1))
            decoder_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                        decoder_mask.to(torch.device('cpu')).unsqueeze(1))
            #print('decoder_mask', decoder_mask.size())
            nopeak_mask = torch.cat([nopeak_mask]*internal_obs.size(0), 0)
            #print('nopeak_mask ', nopeak_mask.size())
            decoder_mask = decoder_mask.to(torch.bool)
            decoder_mask = decoder_mask & nopeak_mask
            
            external_obs = external_obs.to(self.device)
            internal_obs = internal_obs.to(self.device)
            encoder_mask_sqr = encoder_mask_sqr.to(self.device)
            decoder_mask = decoder_mask.to(self.device)
            memory_mask = memory_mask.to(self.device)
            next_promp, next_instance, next_ks = self.forward(external_obs, encoder_mask_sqr, 
                                             internal_obs, decoder_mask, 
                                             memory_mask)
            #print('next', next_promp)
            promp_tensor = torch.cat([promp_tensor, next_promp.unsqueeze(1)], dim=1)#torch.mean(next_, 1)
            generatedInstance.append(next_instance.unsqueeze(1))
            generatedKnapsack.append(next_ks.unsqueeze(1))
        return torch.cat(generatedInstance,1), torch.cat(generatedKnapsack,1), promp_tensor #, torch.cat(internalObservs, 0)
        
    
    def forward (
        self,
        external_obs:torch.tensor,
        encoder_mask:torch.tensor,
        internal_obs:torch.tensor,
        decoder_mask:torch.tensor,
        memory_mask:torch.tensor,
    ):
        external_obs = external_obs.to(torch.float32)
        internal_obs = internal_obs.to(torch.float32)
        encoder_mask = encoder_mask.to(torch.bool)
        decoder_mask = decoder_mask.to(torch.bool)
        memory_mask = memory_mask.to(torch.bool)
        self.en_embed=self.en_embed.to(self.device)
        self.de_embed=self.de_embed.to(self.device)

        ext_embedding = self.en_embed(external_obs)
        #int_embedding = self.de_embed(internal_obs)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        transfer_out = self.decoder(self.position_encode (internal_obs), 
                                    self.encoder(self.position_encode(ext_embedding), encoder_mask), 
                                    decoder_mask, memory_mask)
        self.instance_outer = self.instance_outer.to(self.device)
        self.knapsack_outer = self.knapsack_outer.to(self.device)
        
        #print('nan', torch.isnan(transfer_out[:,0]).all())
        #print('outsize', transfer_out[:,0])
        return torch.nan_to_num(transfer_out[:,0]), \
            self.softmax(self.instance_outer(torch.nan_to_num(transfer_out[:,0,:self.config.output_dim//2]))), \
            self.softmax(self.knapsack_outer(torch.nan_to_num(transfer_out[:,0,self.config.output_dim//2:])))
    
    
    def pad_left(self, sequence, final_length, padding_token):
        pads = [[padding_token] * (final_length - len(sequence[:][0]))] * len(sequence)
        return [pads[i]+sequence[i] for i in range (len(pads))]
    
