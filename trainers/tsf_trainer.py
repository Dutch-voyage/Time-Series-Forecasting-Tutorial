from trainers.base_trainer import base_trainer
from data_providers.data_provider import data_provider
from utils.metrics import masked_mae_np, masked_mse_np, masked_mape_np
import torch
import numpy as np
import os
from tqdm import tqdm


class Trainer(base_trainer):
    def __init__(self, configs):
        super(Trainer, self).__init__(configs)
        self._set_seed()
        self._get_logger()
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        self.train_data, self.train_loader = self._get_data('train')
        self.val_data, self.val_loader = self._get_data('val')
        self.test_data, self.test_loader = self._get_data('test')
        if self.configs.resume:
            self._resume_()
        else:
            self.best_vali_metrics = {'mae': np.inf, 'mse': np.inf, 'mape': np.inf}
    
    def _resume_(self):
        path = os.path.join(self.configs.ckpts_path, self.configs.model.name, self.configs.dataset.name[0])
        path = os.path.join(path, 'checkpoint.pth')
        ckpt = torch.load(path)
        self.model.load_state_dict(ckpt['model'])
        self.best_vali_metrics = ckpt['best_metric']

    def _save_model(self):
        path = os.path.join(self.configs.ckpts_path, self.configs.model.name, self.configs.dataset.name[0])
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint = {
            'model': self.model.state_dict(),
            'best_metric': self.best_vali_metrics
        }
        torch.save(checkpoint, os.path.join(path, f'checkpoint.pth'))
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.configs.dataset, flag)
        return data_set, data_loader

    def train(self):
        self.model.train()
        for epoch in range(self.configs.epochs):
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(self.train_loader)):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_x_mark = {key: value.long().to(self.device) for key, value in batch_x_mark.items()}
                batch_y_mark = {key: value.long().to(self.device) for key, value in batch_y_mark.items()}
                
                batch_x = self.train_data.transform(batch_x)
                batch_y = self.train_data.transform(batch_y)

                self.optim.zero_grad()
                results = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = results["loss"]
                loss.backward()
                self.optim.step()
        
            self.logger.info("Epoch: {} ".format(epoch + 1))
            vali_metrics = self.eval(self.val_loader)
            test_metrics = self.eval(self.test_loader)

            self.logger.info(f"On Valid Set, MAE:{vali_metrics['mae']}, MSE:{vali_metrics['mse']}, MAPE:{vali_metrics['mape']}")
            self.logger.info(f"On Test Set, MAE:{test_metrics['mae']}, MSE:{test_metrics['mse']}, MAPE:{test_metrics['mape']}")

            if vali_metrics['mse'] < self.best_vali_metrics['mse']:
                self.best_vali_metrics = vali_metrics
                self._save_model()
                self.logger.info('best model saved')
    
    def eval(self, data_loader):
        self.model.eval()
        y_pred = []
        y_true = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(data_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = {key: value.long().to(self.device) for key, value in batch_x_mark.items()}
                batch_y_mark = {key: value.long().to(self.device) for key, value in batch_y_mark.items()}

                batch_x = self.train_data.transform(batch_x)
                batch_y = self.train_data.transform(batch_y)

                result = self.model(batch_x, batch_y, batch_x_mark, batch_y_mark)
                outputs = result['y_hat']
                y_pred.append(outputs)
                y_true.append(batch_y)
            y_pred = torch.cat(y_pred, dim=0)
            y_true = torch.cat(y_true, dim=0)
            mae = masked_mae_np(y_true, y_pred, torch.nan)
            mse = masked_mse_np(y_true, y_pred, torch.nan)
            mape = masked_mape_np(y_true, y_pred, torch.nan)

        return {'mae': mae, 'mse': mse, 'mape': mape}

                
