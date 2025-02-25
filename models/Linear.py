import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_len, output_len, num_channels, configs):
        super(Model, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_channels = num_channels
        self.linear = nn.Linear(input_len, output_len)

    def forward(self, x, y, x_mark, y_mark, **kwargs):
        x_means = x.mean(dim=-1, keepdim=True)
        x_stds = x.std(dim=-1, keepdim=True) + 1e-5
        x = (x - x_means) / x_stds
        y_hat = self.linear(x)
        y_hat = y_hat * x_stds + x_means

        result = {}
        result['y_hat'] = y_hat
        result['predloss'] = F.l1_loss(y_hat, y)
        result['loss'] = result['predloss'].mean()
        return result

