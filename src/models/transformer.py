
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

        self.en_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     self.config.max_length,
                                                     self.device)
        self.de_position_encode = PositionalEncoding(self.config.output_dim, 
                                                     100,
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
        self.knapsack_outer = nn.Linear(self.config.output_dim//2, self.config.knapsack_obs_size)

        self.softmax = nn.Softmax()
    
    def generateOneStep (
        self,
        external_obs: torch.tensor,
        generat_link_number: int,
        promp_tensor: Optional[torch.tensor] = None,   
        
    ):
        SOD = [1]*self.config.input_dim
        EOD = [2]*self.config.input_dim
        PAD = [0]*self.config.input_dim
        encoder_padding_mask = torch.zeros_like(external_obs[:,:,0], device=self.device)
        encoder_padding_mask[torch.all(external_obs == torch.tensor(PAD, device=self.device), 
                                       dim=2)] = 1
        
        encoder_mask = torch.zeros_like(external_obs[:,:,0], device=self.device)
        encoder_mask[torch.all(external_obs == torch.tensor(SOD, device=self.device), 
                               dim=2)] = 1
        encoder_mask[torch.all(external_obs == torch.tensor(EOD, device=self.device), 
                               dim=2)] = 1
        encoder_mask = torch.cat([encoder_mask]*self.config.nhead , 0)

        
        if promp_tensor == None:
            start_tokens = [[SOD]]*external_obs.size(0)
            nopeak_mask = np.ones((self.config.nhead, 2*generat_link_number, 
                                   2*generat_link_number))
            nopeak_mask[:,:,-1]=0
            
        else: 
            start_tokens = promp_tensor.tolist()
            nopeak_mask = np.ones((self.config.nhead, 2*generat_link_number, 
                                   2*generat_link_number))
        promp_tensor = torch.tensor(
            self.pad_left(
                sequence=start_tokens,
                final_length=2*generat_link_number, 
                padding_token=PAD
                ),
            dtype=torch.float
        )
        promp_tensor = promp_tensor.to(self.device)

        nopeak_mask = torch.from_numpy(nopeak_mask) == 1
        
        internal_obs = promp_tensor#[:,-(2*generat_link_number):,:]
        decoder_padding_mask = torch.zeros_like(internal_obs[:,:,0])
        decoder_padding_mask[torch.all(internal_obs == torch.tensor(PAD, device=self.device), 
                                       dim=2)] = 1
            
        decoder_mask = torch.zeros_like(internal_obs[:,:,0])
        decoder_mask[torch.all(internal_obs == torch.tensor(EOD, device=self.device), 
                               dim=2)] = 1
        decoder_mask[torch.all(internal_obs == torch.tensor(SOD, device=self.device), 
                               dim=2)] = 1
        decoder_mask = torch.cat([decoder_mask]*self.config.nhead , 0)
        memory_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2).long(), 
                                   encoder_mask.to(torch.device('cpu')).unsqueeze(1).long())
        encoder_mask_sqr = torch.matmul(encoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                        encoder_mask.to(torch.device('cpu')).unsqueeze(1))
        decoder_mask = torch.matmul(decoder_mask.to(torch.device('cpu')).unsqueeze(2), 
                                    decoder_mask.to(torch.device('cpu')).unsqueeze(1))

        nopeak_mask = torch.cat([nopeak_mask]*internal_obs.size(0), 0)
        decoder_mask = decoder_mask.to(torch.bool)
        decoder_mask = decoder_mask & nopeak_mask

        external_obs = external_obs.to(self.device)
        internal_obs = internal_obs.to(self.device)
        encoder_mask_sqr = encoder_mask_sqr.to(self.device)
        decoder_mask = decoder_mask.to(self.device)
        memory_mask = memory_mask.to(self.device)
        encoder_padding_mask = encoder_padding_mask.to(self.device)
        decoder_padding_mask = decoder_padding_mask.to(self.device)
            
        next_promp, next_instance, next_ks = self.forward(external_obs, encoder_mask_sqr, 
                                                          encoder_padding_mask,
                                                          internal_obs, decoder_mask, 
                                                          decoder_padding_mask, 
                                                          memory_mask)
        
        return next_instance.unsqueeze(1), next_ks.unsqueeze(1), promp_tensor
        
    
    def forward (
        self,
        external_obs:torch.tensor,
        encoder_mask:torch.tensor,
        encoder_padding_mask: torch.tensor,
        internal_obs:torch.tensor,
        decoder_mask:torch.tensor,
        decoder_padding_mask: torch.tensor,
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
        int_embedding = self.de_embed(internal_obs)
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        
        #encoder_mask = None
        #decoder_mask = None
        #memory_mask = None
        transfer_out = self.decoder(self.de_position_encode (int_embedding), 
                                    self.encoder(self.en_position_encode(ext_embedding), 
                                                 mask=encoder_mask, 
                                                 src_key_padding_mask=encoder_padding_mask), 
                                    tgt_mask=decoder_mask, memory_mask=memory_mask,
                                    tgt_key_padding_mask=decoder_padding_mask)
        self.instance_outer = self.instance_outer.to(self.device)
        self.knapsack_outer = self.knapsack_outer.to(self.device)
        
        #print('nan', torch.isnan(transfer_out[:,0]).all())
        #print('outsize', transfer_out.size())
        return torch.nan_to_num(transfer_out[:,-1]), \
            self.softmax(self.instance_outer(torch.nan_to_num(transfer_out[:,-1,:self.config.output_dim//2]))), \
            self.softmax(self.knapsack_outer(torch.nan_to_num(transfer_out[:,-1,self.config.output_dim//2:])))
    
    
    def pad_left(self, sequence, final_length, padding_token):
        pads = [[padding_token] * (final_length - len(sequence[:][0]))] * len(sequence)
        return [pads[i]+sequence[i] for i in range (len(pads))]
    
