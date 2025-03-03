import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional
from layers.attention import Attn_Block
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.embedding import Conv_Embedding
from layers.mapper import Mapper
from models import Linear, PTN
from utils.read_cfg import read_cfg
import os

model_dict = {
    'Linear': Linear.Model,
}

class Model(nn.Module):
    def __init__(self,
                 input_len: int,
                 output_len: int,
                 num_channels: int,
                 configs: dict):
        super(Model, self).__init__()
        self.embedding = Conv_Embedding(conv_layers=configs.conv_layers, 
                                        d_model=configs.d_model, 
                                        patch_len=configs.patch_len, 
                                        amp_factor=configs.amp_factor, 
                                        type=configs.type)
        
        # [B, C, L] -> [B, C, L, D]
        attn = Attn_Block(d_model=configs.d_model, 
                          num_heads=configs.num_heads, 
                          norm_type=configs.norm_type, 
                          if_apply_rope=configs.if_apply_rope, 
                          causal_mask=configs.causal_mask, 
                          res_attn=configs.res_attn, 
                          num_layers=configs.attn_layers)
        self.encoder = Encoder(embedding_type=configs.embedding_type, 
                               num_channels=num_channels, 
                               d_model=configs.d_model, 
                               attn_block=attn,
                               patch_len=configs.patch_len, 
                               with_ch=configs.with_ch, 
                               with_tem=configs.with_tem)
        
        # [B, C, l, D] -> [B, C, l, D]

        self.decoder = Decoder(embedding_type=configs.embedding_type, 
                               num_channels=num_channels, 
                               d_model=configs.d_model, 
                               patch_len=configs.patch_len)
        
        aux_model_args_path = os.path.join('configs/models', configs.aux_model + '.yaml')
        aux_model_configs = read_cfg(aux_model_args_path)
        self.aux_model = model_dict[configs.aux_model](input_len=input_len, output_len=output_len, num_channels=num_channels, configs=aux_model_configs)
        self.mapper = Mapper(self.aux_model, configs.task)

    def forward(self, x, y, x_mark, y_mark, **kwargs):
        result = {}
        x_emb = self.embedding(x)
        x_enc = self.encoder(x_emb)
        x_rec = self.decoder(x_enc)

        y_emb = self.embedding(y)
        y_enc = self.encoder(y_emb)
        y_rec = self.decoder(y_enc)

        map_result = self.mapper(x_rec, y)
        y_hat = map_result['y_hat']
        result['y_hat'] = y_hat

        '''
        res_rec = y_rec - y
        res_pred = y_hat - y_rec
        norm_loss = F.mse_loss(F.tanh(res_rec), F.tanh(res_pred))
        '''

        # F.mse_loss(y_hat, y_rec)
        result['predloss'] = map_result['predloss']  # + map_result_stu['predloss']
        # result['normloss'] = torch.zeros_like(norm_loss)
        result['recloss'] = F.l1_loss(x_rec, x, reduction='none').mean(-1) + F.l1_loss(y_rec, y, reduction='none').mean(-1)
        result['loss'] = result['predloss'].mean() + result['normloss'].mean() + result['recloss'].mean()
        return result