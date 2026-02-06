# -*- coding = utf-8 -*-
# @Time: 2025/10/23 16:50
# @Author: wisehone
# @File: exp_basic.py
# @SoftWare: PyCharm
import sys
sys.path.append('..')
sys.path.append('.')

import os
import torch
from models import SEA_FGA, SEA_FGT, Info_FGT

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'SEA_FGA': SEA_FGA,
            'SEA_FGT': SEA_FGT,
            'Info_FGT': Info_FGT
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
