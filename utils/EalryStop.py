# -*- coding = utf-8 -*-
# @Time: 2025/10/20 15:13
# @Author: wisehone
# @File: EalryStop.py
# @SoftWare: PyCharm
import shutil

import numpy as np
import torch
import os
from utils.tools import convert_size, delete_txt_files_in_folder

class EarlyStop:
    """
    early stop according to val loss
    """
    def __init__(self,
                 patience: int = 7,
                 verbose: bool = False,
                 dataset_name: str = '',
                 delta: float = 0.,
                 save_every_epoch: bool=False,
                 monitor: str='val_loss',
                 mode: str='min'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = np.inf if mode == 'min' else -np.inf

        self.delta = delta
        self.save_every_epoch = save_every_epoch
        self.monitor = monitor
        self.mode = mode

    def __call__(self, val_loss, val_metric, model, path, epoch=None):
        if self.monitor == 'val_loss':
            current_value = val_loss
        elif self.monitor == 'val_metric':
            current_value = val_metric
        else:
            raise ValueError('monitor must be val_loss or val_metric')

        if self.mode == 'min':
            score = -current_value
        elif self.mode == 'max':
            score = current_value
        else:
            raise ValueError('mode must be min or max')

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_metric, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, val_metric, model, path, epoch)
            self.counter = 0
            self.early_stop = False

    def save_checkpoint(self, val_loss, val_metric, model, path, epoch=None):
        file_path = os.path.join(path, 'checkpoint.pth')
        # TODO: address this issue：RuntimeError: File ./checkpoints\exp_detect\checkpoint.pth cannot be opened.
        torch.save(model.state_dict(), file_path)
        file_size = convert_size(os.path.getsize(file_path))
        print(f'Model saved ({self.monitor} improved). Size: {file_size}')

        delete_txt_files_in_folder(path)
        if val_loss is None:
            val_loss = 0.0
        if val_metric is None:
            val_metric = 0.0
        with open(os.path.join(path, f'Epoch_{epoch}.txt'), 'w') as f:
            f.write(f'Epoch {epoch} | val_loss={val_loss:.5f} | val_metric={val_metric:.5f}')

        if self.save_every_epoch and epoch is not None:
            suffix = f'_{self.monitor}_{getattr(self, "mode")}_{epoch:d}.pth'
            shutil.copy(file_path, os.path.join(path, f'checkpoint{suffix}'))