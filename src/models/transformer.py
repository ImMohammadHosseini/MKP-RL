
"""

"""
import torch
from torch import nn
from torch.nn import (
    TransformerEncoderLayer,
    TransformerEncoder,
    TransformerDecoderLayer,
    TransformerDecoder,
)
from .positional_encoding import PositionalEncoding
from typing import List, Optional


class TransformerKnapsack (nn.Module):
    def __init__(
        self,
        config,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.name = 'transformer'
        self.config = config
        self.device = device
        
        self.embed = nn.Linear(self.config.input_dim, self.config.output_dim)
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
        
        self.outer = nn.Linear(self.config.output_dim, self.config.max_length-3)
        self.softmax = nn.Softmax()
    
    def generate_step (
        self,
        external_obs: torch.tensor,
        max_tokens_to_generate: int,
        generat_link_number: int,
        promp_tensor: Optional[torch.tensor] = None,   
        
    ):
        encoder_ACT = [0]*self.config.input_dim
        encoder_mask = torch.ones_like(external_obs[:,:,0], device=self.device)
        encoder_mask[torch.all(external_obs == torch.tensor(encoder_ACT, 
                                                            device=self.device), 
                               dim=2)] = 0
        encoder_mask = torch.cat([encoder_mask]*self.config.nhead , 0)

        decoder_ACT = [0]*self.config.output_dim
        if promp_tensor == None:
            start_tokens = [decoder_ACT]
            promp_tensor = torch.tensor(
                self.pad_left(
                    sequence=start_tokens,
                    final_length=2 * generat_link_number,#
                    padding_token=decoder_ACT
                    ),
                dtype=torch.long
            )
            promp_tensor = promp_tensor.unsqueeze(dim=0)
            promp_tensor = torch.cat([promp_tensor]*external_obs.size(0), 0)
        
        promp_tensor = promp_tensor.to(self.device)
        internalObservs = []
        generated = []#; generatedKnapsack = []
        for i in range(max_tokens_to_generate):
            internal_obs = promp_tensor[:,-(2*generat_link_number):,:]#
            internalObservs.append(internal_obs)
            #print('transformer ____any', torch.isnan(torch.cat(internalObservs, 0)).any())
            decoder_mask = torch.ones_like(internal_obs[:,:,0])
            decoder_mask[torch.all(internal_obs == torch.tensor(decoder_ACT, 
                                                                device=self.device), 
                                   dim=2)] = 0
            decoder_mask = torch.cat([decoder_mask]*self.config.nhead , 0)
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
            next_promp, next_ = self.forward(external_obs, encoder_mask_sqr, 
                                             internal_obs, decoder_mask, 
                                             memory_mask)
            #print('transPPPPPP ____max', torch.min(next_promp))
            #print('transPPPPPP ____any', torch.isnan(next_promp).any())
            promp_tensor = torch.cat([promp_tensor, next_promp.unsqueeze(1)], dim=1)#torch.mean(next_, 1)
            #generatedKnapsack.append(next_.unsqueeze(1))
            generated.append(next_.unsqueeze(1))
        return torch.cat(generated,1), promp_tensor #, torch.cat(internalObservs, 0)
        
    
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
        
        self.embed=self.embed.to(self.device)
        obs_embeding = self.embed(external_obs)
        positional = self.position_encode(obs_embeding)  
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        transfer_out = self.decoder(self.position_encode (internal_obs), 
                                    self.encoder(positional, encoder_mask), 
                                    decoder_mask, memory_mask)
        self.outer = self.outer.to(self.device)
        '''if torch.isnan(transfer_out[:,0]).any() == True:
            print('trueeeeeeeeeeeeeeeeeeeeee')
            print('max', torch.max(internal_obs))
            print('min', torch.min(internal_obs))
            print('any', torch.isnan(transfer_out[:,0]).any())
            print('all', torch.isnan(transfer_out[:,0]).all())'''

        return torch.nan_to_num(transfer_out[:,0]), \
            self.softmax(self.outer(torch.nan_to_num(transfer_out[:,0])))
    
    def pad_left(self, sequence, final_length, padding_token):
        return [padding_token] * (final_length - len(sequence)) + sequence
    
    
