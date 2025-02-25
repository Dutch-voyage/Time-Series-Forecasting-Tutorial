import os
from models import Linear
import torch
from torch.backends import cudnn
from utils.read_cfg import read_cfg
from omegaconf import OmegaConf
import numpy as np
import random
import argparse
import logging
import time
model_dict = {
    'Linear': Linear,
}

class base_trainer(object):
    def __init__(self, args_path):
        args_base = read_cfg(args_path)
        datasets_args_path = os.path.join(args_base.config_path, 'datasets', args_base.dataset + '.yaml')
        if not os.path.exists(datasets_args_path):
            raise FileNotFoundError('Dataset config file not found.')
        model_args_path = os.path.join(args_base.config_path, 'models', args_base.model + '.yaml')
        if not os.path.exists(model_args_path):
            raise FileNotFoundError('Model config file not found.')
        task_args_path = os.path.join(args_base.config_path, 'tasks', args_base.task + '.yaml')
        if not os.path.exists(task_args_path):
            raise FileNotFoundError('Task config file not found.')

        args_dataset = read_cfg(datasets_args_path)
        args_model = read_cfg(model_args_path)
        args_task = read_cfg(task_args_path)

        args_sync = OmegaConf.merge(args_dataset.sync, args_task.sync, args_base.sync)
        args_base.sync = args_sync
        args_dataset.sync = args_sync
        args_model.sync = args_sync
        args_task.sync = args_sync

        OmegaConf.save(args_base, args_path)
        OmegaConf.save(args_dataset, datasets_args_path)
        OmegaConf.save(args_model, model_args_path)
        OmegaConf.save(args_task, task_args_path)
        if 'aux_models' in args_model:
            for aux_model in args_model.aux_models:
                aux_model_args_path = os.path.join(args_base.config_path, 'models', aux_model + '.yaml')
                if not os.path.exists(aux_model_args_path):
                    raise FileNotFoundError('Model config file not found.')
                args_aux_model = read_cfg(aux_model_args_path)
                args_aux_model.sync = args_sync
                OmegaConf.save(args_aux_model, aux_model_args_path)

        self.configs = args_base
        self.configs.dataset = args_dataset
        self.configs.model = args_model
        self.configs.task = args_task
        self._set_seed()

    def _set_seed(self):
        np.random.seed(self.configs.sync.seed)
        random.seed(self.configs.sync.seed)
        torch.manual_seed(self.configs.sync.seed)

        # cudnn.deterministic = True
        # cudnn.benchmark = True

        torch.backends.cuda.enable_flash_sdp(enabled=True)

    def _get_logger(self):
        LOG_FORMAT = "%(asctime)s  %(message)s"
        DATE_FORMAT = "%m/%d %H:%M"

        console_handler = logging.StreamHandler()  # 输出到控制台
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)
        self.logger = logging.getLogger(__name__)   
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        # self.logger.addHandler(console_handler)

        if self.configs.log_path is not None:
            log_path = os.path.dirname(self.configs.log_path)
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_name = self.configs.log_path + time.strftime(
                '%Y-%m-%d-%H-%M-%S') + f'_{self.configs.dataset.name[0]}_{self.configs.model.name}_.log'

            file_handler = logging.FileHandler(log_name)  # 输出到文件
            self.logger.addHandler(file_handler)
            self.logger.info(self.configs)
            self.logger.info(self.configs.model)
            self.logger.info(self.configs.dataset)

    def _build_model(self):
        model = model_dict[self.configs.model.name].Model(input_len = self.configs.sync.input_len,
                                                          output_len = self.configs.sync.output_len,
                                                          num_channels = self.configs.sync.n_channels,
                                                          configs = self.configs.model)
        return model

    def _acquire_device(self):
        device = self.configs.sync.device
        return device

    def _get_data(self, flag):
        pass

    def _sava_model(self):
        pass
