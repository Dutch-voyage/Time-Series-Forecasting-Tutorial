from torch.utils.data import Dataset, DataLoader
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import random
import numpy as np
from utils.read_cfg import read_cfg
from tqdm import tqdm
import time
import torch
import torch.nn.functional as F
from einops import rearrange

data_dict = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
    'PEMS03': 'PEMS03/PEMS03.csv',
    'PEMS04': 'PEMS04/PEMS04.csv',
    'PEMS07': 'PEMS07/PEMS07.csv',
    'PEMS08': 'PEMS08/PEMS08.csv',
    'weather': 'weather/weather.csv',
    'traffic': 'traffic/traffic.csv',
    'electricity': 'electricity/electricity.csv',
    'illness': 'illness/national_illness.csv',
    }


def data_provider(args, flag='train', para=False):
    shuffle = True if flag == 'train' else False
    batch_size = args.sync.batch_size
    dataset = CustomDataset(args, flag)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    return dataset, dataloader

class CustomDataset(Dataset):
    def __init__(self, args, flag='train'):
        self.args = args
        self.task = self.args.task
        self.scale = self.args.scale
        self.scaler = StandardScaler()
        self.n_channels = self.args.sync.n_channels
        if self.task == 'forecasting':
            self.pred_type = self.args.pred_type
            self.his_len = self.args.sync.input_len
            self.pred_len = self.args.sync.output_len
        set_types = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = set_types[flag]
        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.args.root_dir + data_dict[self.args.name[0]])
        if self.args.name[0] == 'ETTh1' or self.args.name[0] == 'ETTh2' or self.args.name[0] == 'ETTm1' or self.args.name[0] == 'ETTm2':
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            df_raw['day'] = df_raw.date.apply(lambda row: row.day, 1)
            # df_raw = df_raw.drop(df_raw[df_raw['day'] == 31].index)
            # df_raw = df_raw.drop(df_raw[(df_raw['date'] >= '2017-07-23 21:00') & (df_raw['date'] <= '2017-07-29 17:00')].index)
            df_raw = df_raw.drop(labels='day', axis=1)
        cols = list(df_raw.columns)
        if self.pred_type == 'm2m' or self.pred_type == 'm2u':
            df_data = df_raw[cols[1:]]
        elif self.pred_type == 'u2u':
            df_data = df_raw[cols[-1]]

        if self.args.name[0] == 'ETTh1' or self.args.name[0] == 'ETTh2':
            data_length = 20 * 30 * 24
        elif self.args.name[0] == 'ETTm1' or self.args.name[0] == 'ETTm2':
            data_length = 20 * 30 * 24 * 4
        else:
            data_length = len(df_data)
        if "PEMS" in self.args.name[0]:
            df_data = df_data.fillna(0)
        num_train = int(data_length * self.args.train_ratio)
        num_val = int(data_length * self.args.val_ratio)
        num_test = int(data_length * self.args.test_ratio)

        border1s = [0, num_train - self.his_len, num_train + num_val - self.his_len]
        border2s = [num_train, num_train + num_val, data_length]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        train_data = df_data[border1s[0]:border2s[0]]
        self.scaler.fit(train_data.values)
        self.means = self.scaler.mean_
        self.stds = self.scaler.scale_

        data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) - 1
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1) - 1
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute / 15, 1)
        data_stamp = df_stamp.drop(labels='date', axis=1).values

        data = torch.tensor(data[border1:border2], device=self.args.sync.device)
        if self.set_type == 0 and self.scale:
            data = (data - self.means) / self.stds
        self.data_x = data
        self.data_y = data

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        x_mark, y_mark = {}, {}
        index = index + self.his_len
        x = self.data_x[index - self.his_len:index, ...].transpose(-1, -2).float()
        y = self.data_y[index:index + self.pred_len, ...].transpose(-1, -2).float()
        x_mark['time_stamp'] = self.data_stamp[index - self.his_len:index]
        x_mark['pos_stamp'] = torch.arange(0, self.his_len)
        y_mark['time_stamp'] = self.data_stamp[index:index + self.pred_len]
        y_mark['pos_stamp'] = torch.arange(0, self.pred_len)

        x_mark['channel'] = torch.arange(0, self.n_channels)
        y_mark['channel'] = torch.arange(0, self.n_channels)
        return x, y, x_mark, y_mark

    def __len__(self):
        return self.data_x.shape[0] - self.his_len - self.pred_len + 1

    def transform(self, data):
        B, C, L = data.shape
        data = rearrange(data, 'b c l -> (b l) c')
        out = (data - torch.Tensor(self.means).to(data)) / torch.Tensor(self.stds).to(data)
        out = rearrange(out, '(b l) c -> b c l', l=L)
        return out

    def inverse_transform(self, data):
        B, C, L = data.shape
        data = rearrange(data, 'b c l -> (b l) c')
        out = data * (torch.Tensor(self.stds).to(data)) + torch.Tensor(self.means).to(data)
        out = rearrange(out, '(b l) c -> b c l', l=L)
        return out

