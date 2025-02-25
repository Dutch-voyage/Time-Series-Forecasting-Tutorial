import numpy as np
import torch
from fastdtw import fastdtw
from tqdm import tqdm

def mask_np(array, null_val):
    return torch.not_equal(array, null_val).float()

def masked_mape_np(y_true, y_pred, null_val=torch.nan, reduction='mean'):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mape = torch.abs((y_pred - y_true) / y_true)
    mape = mask * mape
    if reduction == 'mean':
        return torch.mean(mape) * 100
    elif reduction == 'sum':
        return torch.sum(mape) * 100
    elif reduction == 'none':
        return mape * 100
    else:
        raise ValueError('reduction should be mean, sum or none')


def masked_rmse_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return torch.sqrt(torch.mean(mask * mse))


def masked_mse_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return torch.mean(mask * mse)


def masked_mae_np(y_true, y_pred, null_val=torch.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = torch.abs(y_true - y_pred)
    return torch.mean(mask * mae)

def get_dtw(y_true, y_pred, null_val=torch.nan):
    y_true = y_true.reshape(y_true.shape[-1], y_true.shape[-2])
    y_pred = y_pred.reshape(y_pred.shape[-1], y_pred.shape[-2])
    dtw_list = []
    manhattan_distance = lambda x, y: np.abs(x - y)
    for i in range(y_pred.shape[0]):
        x = y_pred[i].reshape(-1, 1)
        y = y_true[i].reshape(-1, 1)
        d, _, = fastdtw(x, y, dist=manhattan_distance)
        dtw_list.append(d)
    dtw = np.array(dtw_list).mean()
    return dtw

