from .base import Problem
from ..data.thor.dataset import ThorDataset, collate

import torch
import torch.nn.functional as F


class Thor(Problem):
    def __init__(self, data_path, valid_path, downsample=None):
        self.train_dataset = ThorDataset(data_path, downsample_pointclouds=downsample)
        self.valid_dataset = ThorDataset(valid_path, downsample_pointclouds=downsample)
        self.collate_fn = collate

    def loss(self, item, pred):
        actual = item['pos'].unsqueeze(-2)
        mask = item['mask'].unsqueeze(-1)
        pred, actual = torch.broadcast_tensors(pred, actual)
        loss = F.mse_loss(pred, actual, reduction='none').sum(-1)
        loss *= mask
        return loss.mean()
