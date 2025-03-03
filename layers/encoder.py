import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple, Optional

class Encoder(nn.Module):
    def __init__(self,
                 embedding_type: str,
                 attn_block: nn.Module,
                 d_model: int,
                 num_channels: int,
                 patch_len: Optional[int],
                 with_ch: bool,
                 with_tem: bool,
                 d_in: Optional[int]=None):
        super(Encoder, self).__init__()
        self.embedding_type = embedding_type
        self.num_channels = num_channels
        # self.attntem = attn_block(d_model=d_model)
        self.attn = attn_block
        self.patch_len = patch_len
        self.with_ch = with_ch
        self.with_tem = with_tem

    def forward(self, x_emb):
        B = x_emb.shape[0]
        C = x_emb.shape[1]
        if not self.with_ch and not self.with_tem:
            return x_emb
        x_enc_tem = torch.zeros_like(x_emb)
        x_enc_ch = torch.zeros_like(x_emb)
        x_emb = rearrange(x_emb, 'b c l d-> (b c) l d')
        if self.with_tem:
            if self.embedding_type == 'patch':
                # x_emb = x_emb.permute(0, 2, 1, 3)
                # x_emb = x_emb.reshape(-1, x_emb.shape[2], x_emb.shape[3])
                x_emb = rearrange(x_emb, 'b c l d-> (b l) c d')
            elif self.embedding_type == 'conv':
                x_enc_tem = self.attn(x_emb)
                x_enc_tem = rearrange(x_enc_tem, '(b c) l d -> b c l d', b=B, c=C)
            elif self.embedding_type == 'patchedconv':
                x_emb = rearrange(x_emb, 'b c (l p) d -> (b c l) p d', p=self.patch_len)
            elif self.embedding_type == 'vanilla':
                pass
            else:
                raise ValueError('Invalid embedding type')
        # tem_x_enc = self.attnch(x_emb)
        # tem_x_enc = rearrange(tem_x_enc, '(b c ) l d -> b c l d', b=B, c=C)
        if self.with_ch:
            if self.embedding_type == 'patch':
                x_enc = rearrange(x_enc, '(b l) c d -> b c l d', b=B)
            elif self.embedding_type == 'conv':
                x_emb = rearrange(x_emb, '(b c) l d -> (b l) c d', b=B, c=C)
                x_enc_ch = self.attn(x_emb)
                x_enc_ch = rearrange(x_enc_ch, '(b l) c d -> b c l d', b=B)
            elif self.embedding_type == 'patchedconv':
                x_enc = rearrange(x_enc, '(b c l) p d -> b c (l p) d', b=B, c=C)
            elif self.embedding_type == 'vanilla':
                pass
            else:
                raise ValueError('Invalid embedding type')
        x_enc = x_enc_tem + x_enc_ch
        return x_enc