# -*- coding = utf-8 -*-
# @Time: 2025/10/24 15:31
# @Author: wisehone
# @File: threshold.py
# @SoftWare: PyCharm
import sys

sys.path.append('..')
sys.path.append('.')

import numpy as np
import torch
import torch.nn.functional as F

from utils.pot import POTThresholdDetector
from utils.spot import SPOT

def prob_normalize(x, method="softmax", eps=1e-8, temperature=1.0):
    if method == "softmax":
        return torch.softmax(x / temperature, dim=-1)

    elif method == "l1":
        x = x.clamp_min(eps)
        return x / x.sum(dim=-1, keepdim=True)

    else:
        raise ValueError("Unknown normalization method")


def kl_div(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1)


@torch.no_grad()
def symmetric_kl_scores(y1, y2, temperature=1.0, dim_feat=-1, norm="softmax"):
    if dim_feat != -1:
        y1 = y1.movedim(dim_feat, -1)
        y2 = y2.movedim(dim_feat, -1)

    p = prob_normalize(y1, norm, temperature=temperature)
    q = prob_normalize(y2, norm, temperature=temperature)

    return kl_div(p, q) + kl_div(q, p)

def unified_threshold(train_scores, test_scores, k=0.1):
    """
    Compute unified percentile threshold over all datasets.
    """
    all_scores = np.concatenate([train_scores, test_scores])
    threshold = np.percentile(all_scores, 100.0 - k)
    return threshold

class ThresholdPolicy:
    def __init__(self, mode: str, k_percent: float=5.0,
                 spot_q: float=1e-5, spot_level: float=0.02, spot_scale: float=3.0):
        self.mode = mode
        self.kp = float(k_percent)
        self.spot_q = spot_q
        self.spot_level = spot_level
        self.spot_scale = spot_scale
        self.threshold_ = 1e-5
        self.detector_ = None

    def fit(self, train_scores_1d: np.ndarray=None, ref_score_1d: np.ndarray=None):
        if self.mode == 'percentile_val':
            assert ref_score_1d is not None
            self.threshold_ = float(np.percentile(ref_score_1d, 100.0 - self.kp))
        elif self.mode == 'percentile_train':
            assert train_scores_1d is not None
            self.threshold_ = float(np.percentile(train_scores_1d, 100.0 - self.kp))
        elif self.mode == 'spot':
            assert train_scores_1d is None
            self.detector_ = POTThresholdDetector(q=self.spot_q, level=self.spot_level, scale_factor=self.spot_scale)
            self.detector_.fit(train_scores_1d, ref_score_1d)
            self.threshold_ = float(self.detector_.threshold)
        elif self.mode == 'unified':
            self.threshold_ = unified_threshold(train_scores_1d, ref_score_1d, k=self.kp)
        else:
            raise ValueError(f'Unknown mode: {self.mode}')
        return self

    def predict(self, scores_1d: np.ndarray) -> np.ndarray:
        return (scores_1d > self.threshold_).astype(np.int32)
    
    def get_shreshold(self):
        return self.threshold_