# -*- coding = utf-8 -*-
# @Time: 2025/10/19 15:24
# @Author: wisehone
# @File: PositionEmbedding.py
# @SoftWare: PyCharm
import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        """
        dim == d_model
        :param dim:
        :param max_seq_len:
        """
        super(RotaryPositionEmbedding, self).__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        # precal cos and sin
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: [B, H, T, D] / [B, T, D]
        :return:
        """
        """
        seq_len = x.size(-2)
        ori_shape = x.shape

        if seq_len > self.max_seq_len:
            self._extend_caches(seq_len)

        # suitable for 2D or 3D
        if x.dim() == 2:  # [T, D]
            x = x.unsqueeze(0)  # [1, T, D]
        elif x.dim() == 3:  # [B, T, D]
            x = x.unsqueeze(1)  # [B, 1, T, D] 

        # get cos and sin
        cos = self.cos_cached[:, :, :seq_len, ...].to(x.device)
        sin = self.sin_cached[:, :, :seq_len, ...].to(x.device)

        # apply
        x1, x2 = x[..., 0::2], x[..., 1::2]
        rotated_x = torch.cat((-x2, x1), dim=-1)

        pe = x*cos + rotated_x*sin

        if len(ori_shape) == 2:  # [T, D]
            pe = pe.squeeze(0)  
        elif len(ori_shape) == 3:  # [B, T, D]
            pe = pe.squeeze(1)  

        return pe
        """
        B, H, T, D = x.shape
        assert D == self.dim

        # frequencies
        position = torch.arange(T, dtype=x.dtype, device=x.device).unsqueeze(1)  # [T, 1]
        dim_index = torch.arange(D, dtype=x.dtype, device=x.device).unsqueeze(0)  # [1, D]

        # θ = 10000^{-2i/d}
        freqs = self.max_seq_len ** (-2 * (dim_index // 2) / D)  # [1, D]

        angles = position * freqs  # [T, D]

        # split into even / odd
        sin = angles.sin()[None, None, :, :]  # [1,1,T,D]
        cos = angles.cos()[None, None, :, :]  # [1,1,T,D]

        x1 = x[..., ::2]
        x2 = x[..., 1::2]

        # rotate
        x_rot = torch.stack([x1 * cos[..., ::2] - x2 * sin[..., ::2],
                             x1 * sin[..., ::2] + x2 * cos[..., ::2]], dim=-1)

        # merge last two dims
        return x_rot.flatten(-2)

    def _extend_caches(self, seq_len: int):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        t = torch.arange(seq_len).type_as(inv_freq)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]
        self.max_seq_len = seq_len

class SinusoidalPositionEmbedding(nn.Module):
     """
     Sinusoidal positional embedding (Transformer-style)
     """
     def __init__(self, dim: int, max_seq_len: int = 2048):
         super(SinusoidalPositionEmbedding, self).__init__()
         position = torch.arange(max_seq_len).unsqueeze(1)
         div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
         pe = torch.zeros(max_seq_len, dim)
         pe[:, 0::2] = torch.sin(position * div_term)
         pe[:, 1::2] = torch.cos(position * div_term)
         self.register_buffer('pe', pe.unsqueeze(0)) # [1, T, D]

     def forward(self, x: torch.Tensor) -> torch.Tensor:
         seq_len = x.size(-2)
         return self.pe[:, :seq_len, ...]


class Time2Vec(nn.Module):
    """
    """

    def __init__(self, dim: int):
        super(Time2Vec, self).__init__()
        self.w0 = nn.Parameter(torch.randn(1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w1 = nn.Parameter(torch.randn(dim - 1))
        self.b1 = nn.Parameter(torch.randn(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.w0 * x + self.b0
        periodic = torch.sin(self.w1 * x + self.b1)
        return torch.cat([linear, periodic], dim=-1)