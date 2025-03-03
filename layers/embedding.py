import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Dict, Tuple

class Patch_Embedding(nn.Module):
    def __init__(self,
                 patch_len: int,
                 d_model: int,
                 type: str):
        super(Patch_Embedding, self).__init__()
        self.patch_len = patch_len
        self.type = type
        self.embedding = nn.Linear(patch_len, d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = rearrange(x, 'b c (l p) -> b c l p', p=self.patch_len)
        x_emb = self.embedding(x)
        return x_emb

# [B, C, L] -> [B, C, D]
class Conv_Embedding(nn.Module):
    def __init__(self,
                 conv_layers: int,
                 d_model: int,
                 patch_len: int,
                 amp_factor: int,
                 type: str):
        super(Conv_Embedding, self).__init__()
        self.patch_len = patch_len
        self.amp_factor = amp_factor
        self.type = type
        now_channel = 1
        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.activation = nn.GELU()
        for _ in range(conv_layers):
            conv = nn.Conv1d(now_channel * amp_factor, now_channel * 2 * amp_factor, kernel_size=3, stride=2, padding=1) # TODO L = L // 2
            self.convs.append(conv)
            self.norms.append(nn.InstanceNorm1d(now_channel * amp_factor))
            now_channel *= 2

        # self.weight = nn.Parameter(torch.randn(amp_factor * (conv_layers + 1), d_model), requires_grad=True)
        # self.bias = nn.Parameter(torch.randn(d_model), requires_grad=True)
        # nn.init.xavier_normal_(self.weight)
        self.linear = nn.Linear((conv_layers + 1) * amp_factor, d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B*C, 1, L).repeat(1, self.amp_factor, 1)
        xs = [rearrange(x, 'b c l -> b l c')]
        # xs = []
        current_len = L
        for conv, norm in zip(self.convs, self.norms):
            if current_len % 2 != 0:
                x = F.pad(x, (0, 1), 'replicate')
            x = norm(x)
            x = conv(x)
            x = self.activation(x)
            if current_len % 2 != 0:
                x = rearrange(x, 'b (c k) l -> b c (l k)', c=x.shape[1] // 2)[..., :-1]
            xs.append(rearrange(x, 'b (a c) l -> b (l c) a', a=self.amp_factor))
            current_len = (current_len + 1) // 2
        x_emb = torch.stack(xs, dim=-1)
        # x_emb = rearrange(x_emb, '(b c) (l p) a n -> b c l (p a n)', c=C, p=self.patch_len)
        x_emb = rearrange(x_emb, '(b c) l a n -> b c l (a n)', c=C)
        x_emb = self.linear(x_emb)
        # x_emb = torch.einsum('b c l k, k d -> b c l d', x_emb, self.weight[:self.amp_factor * (max_layer + 1), :]) + self.bias
        return x_emb

class PatchedConv_Embedding(nn.Module):
    def __init__(self,
                 conv_layers: int,
                 d_model: int,
                 patch_len: int,
                 amp_factor: int,
                 type: str):
        super(PatchedConv_Embedding, self).__init__()
        self.patch_len = patch_len
        self.amp_factor = amp_factor
        self.type = type
        now_channel = 1
        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.activation = nn.GELU()
        for _ in range(conv_layers):
            conv = nn.Conv1d(now_channel * amp_factor, now_channel * 2 * amp_factor, kernel_size=3, stride=2, padding=1) # , groups=now_channel)
            self.convs.append(conv)
            self.norms.append(nn.InstanceNorm1d(now_channel * amp_factor))
            now_channel *= 2

        self.linear = nn.Linear((conv_layers + 1) * amp_factor, d_model)
        # self.linear = nn.Linear((conv_layers + 1), d_model)

    def forward(self, x):
        B, C, L = x.shape
        x = x.reshape(B*C, 1, L).repeat(1, self.amp_factor, 1)
        xs = [rearrange(x, 'b c l -> b l c')]
        # xs = []
        current_len = L
        for conv, norm in zip(self.convs, self.norms):
            if current_len % 2 != 0:
                x = F.pad(x, (0, 1), 'replicate')
            x = norm(x)
            x = conv(x)
            x = self.activation(x)
            if current_len % 2 != 0:
                x = rearrange(x, 'b (c k) l -> b c (l k)', c=x.shape[1] // 2)[..., :-1]
            xs.append(rearrange(x, 'b (a c) l -> b (l c) a', a=self.amp_factor))
            current_len = (current_len + 1) // 2
        x_emb = torch.stack(xs, dim=-1)
        # x_emb = rearrange(x_emb, '(b c) (l p) a n -> b c l (p a n)', c=C, p=self.patch_len)
        x_emb = rearrange(x_emb, '(b c) l a n -> b c l (a n)', c=C)
        x_emb = self.linear(x_emb)
        return x_emb

class ConvPatch_Embedding(nn.Module):
    def __init__(self,
                 conv_layers: int,
                 d_model: int,
                 patch_len,
                 type: str):
        super(ConvPatch_Embedding, self).__init__()
        self.conv_layers = conv_layers
        self.patch_len = patch_len
        self.type = type

        now_channel = 1
        self.convs = nn.ModuleList([])
        self.norms = nn.ModuleList([])
        self.activation = nn.GELU()
        for _ in range(conv_layers):
            self.convs.append(nn.Conv1d(now_channel, now_channel*2, kernel_size=3, stride=2, padding=1, groups=now_channel))

            self.norms.append(nn.InstanceNorm1d(now_channel))
            now_channel *= 2
        self.linear = nn.Linear((conv_layers + 1) * patch_len, d_model)
    def forward(self, x):
        B, C, L = x.shape
        # assert L % (2 ** self.conv_layers) == 0
        x = x.reshape(B*C, 1, L)
        xs = [x]
        current_len = L
        for conv, norm in zip(self.convs, self.norms):
            if current_len % 2 != 0:
                x = F.pad(x, (0, 1), 'replicate')
            x = norm(x)
            x = conv(x)
            x = self.activation(x)
            # xs.append(rearrange(x, 'b c l -> b l c'))
            xs.append(x)
            current_len = (current_len + 1) // 2
        xs.reverse()
        x_emb = xs[0].unsqueeze(-1)
        for x_l in xs[1:]:
            c = x_l.shape[1]
            x_emb = rearrange(x_emb, 'b (c k) l d -> b c (l k) d',  c=c)
            if x_l.shape[-1] % 2 != 0:
                x_emb = x_emb[:, :, :-1, :]
            x_emb = torch.cat([x_emb, x_l.unsqueeze(-1)], dim=-1)
        # x_emb = torch.stack(xs, dim=-1)
        x_emb = rearrange(x_emb.squeeze(1), '(b c) (l p) d -> b c l (p d)', c=C, p=self.patch_len)
        return self.linear(x_emb)
