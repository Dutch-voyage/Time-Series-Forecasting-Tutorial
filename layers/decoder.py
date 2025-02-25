import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional

class Decoder(nn.Module):
    def __init__(self,
                 embedding_type: str,
                 num_channels: int,
                 d_model: int,
                 patch_len: int,
                 ):
        super(Decoder, self).__init__()
        self.embedding_type = embedding_type
        if self.embedding_type == 'patch':
            d_out = patch_len
        elif self.embedding_type == 'patchedconv':
            d_out = 1
        elif self.embedding_type == 'convpatch':
            d_out = patch_len
        elif self.embedding_type == 'conv':
            d_out = 1
        else:
            d_out = num_channels
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, x_enc):
        x_dec = self.linear(x_enc)
        x_dec = rearrange(x_dec, 'b c l d -> b c (l d)')

        return x_dec