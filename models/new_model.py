import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_len, output_len, num_channels, configs):
        super(Model, self).__init__()
    