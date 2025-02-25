from trainers.tsf_trainer import Trainer
import numpy as np
import random
import yaml
import warnings
import torch

seed = 1234

if __name__ == '__main__':
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print(torch.cuda.current_device())
    trainer = Trainer('configs/bases/base.yaml')
    trainer.train()
