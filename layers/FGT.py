# -*- coding = utf-8 -*-
# @Time: 2025/11/3 8:13
# @Author: wisehone
# @File: FGT.py
# @SoftWare: PyCharm
import sys
sys.path.append('..')
sys.path.append('.')

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PositionEmbedding import RotaryPositionEmbedding

# Frequency-Guided Transformer
class FGT(nn.Module):
    """
    Frequency-Guided Transformer (FGT)
    Input:  x [B, N, T]
    Output: h [B, T, D]
    """
    def __init__(self,
                 num_channels: int,
                 T: int,
                 d_model: int,
                 n_heads: int = 8,
                 n_layers: int = 3,
                 dropout: float = 0.1,
                 ffn_dim: int = 2048,
                 position_embedding: str = 'rotate'):
        super(FGT, self).__init__()

        self.num_channels = num_channels
        self.T = T
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # spectral entropy
        self.spectral_gate = SpectralGate(num_channels)

        # time embedding
        self.temporal_embedding = TemporalEmbedding(T, num_channels, d_model)

        # position embedding
        self.position_embedding = None
        # if position_embedding is not None and position_embedding != 'rotate':
        #     self.position_embedding = RotaryPositionEmbedding(d_model)
        

        # encoder stack
        self.encoder_layers = nn.ModuleList([
            FGTEncoderLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # output
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, num_channels)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, T]
        :return: [B, T, D]
        """
        B, N, T = x.shape

        # 1. spectral gate
        channel_gate = self.spectral_gate(x)
        # print('channel_gate.shape: ', channel_gate.shape)
        # print('x.shape: ', x.shape)
        x = x * channel_gate.unsqueeze(-1)

        # 2. temporal embedding
        z = self.temporal_embedding(x)

        # 3. position embedding
        if self.position_embedding is not None:
            pos_emb = self.position_embedding(z)
            z = z + pos_emb

        # 4. transformer encoder
        for encoder in self.encoder_layers:
            z = encoder(z)

        # 5.projection
        # TODO: whether to add norm and dropout
        h = self.layer_norm(z)
        h = self.output_projection(z)

        return z

# Spectral Gate
class SpectralGate(nn.Module):
    """
    Compute spectral entropy per channel -> gate weight
    """
    def __init__(self, num_channels: int):
        super(SpectralGate, self).__init__()
        self.num_channels = num_channels
        # TODO: suspective
        self.proj = nn.Sequential(
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels),
            nn.Sigmoid() # why sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, N, T]
        :return: gate [B, N]
        """
        B, N, T = x.shape
        x_fft = torch.fft.fft(x, dim=-1)
        # x_fft = torch.fft.rfft(x, dim=-1)
        psd = torch.abs(x_fft) ** 2

        psd_norm = psd / (psd.sum(dim=-1, keepdim=True) + 1e-8)
        # print('psd_norm.shape: ', psd_norm.shape)
        entropy = - (psd_norm * torch.log(psd_norm + 1e-8)).sum(dim=-1) # [B, N]
        entropy = entropy / math.log(T)
        # print('entropy.shape: ', entropy.shape)

        # TODO: check
        gate = self.proj(entropy)
        self.last_entropy = entropy
        self.last_gate = gate
        # gate = self.proj(entropy.unsqueeze(-1)).squeeze(-1) # [B, N]
        # print('gate.shape: ', gate.shape)
        return gate

# Temporal embedding
class TemporalEmbedding(nn.Module):
    def __init__(self, T: int, num_channels: int, d_model: int):
        super(TemporalEmbedding, self).__init__()
        self.linear_proj = nn.Linear(num_channels, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
       :param x: [B, N, T]
       :return: [B, T, D]
       """
        x_t = x.transpose(1, 2) # [B, T, N]
        z = self.linear_proj(x_t)  # [B, T, D]
        return z

class SpectralEntropyFFN(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float=0.1):
        super(SpectralEntropyFFN, self).__init__()
        # main ffn
        self.fc_expand = nn.Linear(d_model, ffn_dim)
        self.fc_reduce = nn.Linear(ffn_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # spectral entropy gate
        self.freq_gate_proj = nn.Linear(d_model, d_model)
        self.freq_gate_bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_in: [B, T, D]
        :return: [B, T, D]
        """
        B, T, D = x.shape
        # print('x.shape: ', x.shape)

        freq_repr = torch.fft.rfft(x, dim=1)
        psd = torch.abs(freq_repr) ** 2
        psd_norm = psd / (psd.sum(dim=1, keepdim=True) + 1e-8)

        spec_entropy = - (psd_norm * torch.log(psd_norm + 1e-8)).sum(dim=1) # [B, D]
        spec_entropy = spec_entropy / math.log(T)

        # print('spec_entropy.shape: ', spec_entropy.shape)
        freq_gate = torch.sigmoid(self.freq_gate_proj(spec_entropy.unsqueeze(1)) + self.freq_gate_bias)
        # print('freq_gate.shape: ', freq_gate.shape)

        hidden_in = freq_gate * x
        hidden_mid = F.relu(self.fc_expand(hidden_in))
        # print('hidden_mid.shape: ', hidden_mid.shape)
        # print('freq_gate.shape: ', freq_gate.shape)
        # hidden_mid = hidden_mid * freq_gate.unsqueeze(1)

        hidden_out = self.fc_reduce(self.dropout(hidden_mid))
        hidden_out = self.norm(hidden_in + self.dropout(hidden_out))
        # hidden_out = self.norm(x + self.dropout(hidden_out))
        return hidden_out



class FGTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super(FGTEncoderLayer, self).__init__()
        self.self_attention = FGTMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)

        self.freq_ffn = SpectralEntropyFFN(d_model, ffn_dim, dropout)
        """
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        """

        self.ffn_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B, T, D]
        :return: [B, T, D]
        """
        attn_out = self.self_attention(x)
        # TODO: check
        attn_out_norm = self.attn_norm(x + self.dropout(attn_out))

        # ff_out = self.feed_forward(attn_out_norm)
        ff_out = self.freq_ffn(attn_out_norm)
        # ff_out = self.freq_ffn(x)
        output = self.ffn_norm(attn_out_norm + self.dropout(ff_out))
        return output

# multihead  attention
class FGTMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float=0.1, RoPE: bool=True):
        super(FGTMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        self.pos_embedding = None
        if RoPE:
            self.pos_embedding = RotaryPositionEmbedding(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        Q = self.w_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.pos_embedding is not None:
            Q_rope = self.pos_embedding(Q)
            K_rope = self.pos_embedding(K)
        else:
            Q_rope = Q
            K_rope = K

        attn = torch.matmul(Q_rope, K_rope.transpose(-2, -1) / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        self.last_attn = attn  
        attn = self.dropout(attn)


        output = torch.matmul(attn, V)
        output = output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.w_o(output)
        return output

class DualFGT(nn.Module):
    def __init__(self,
                 num_channels: int,
                 T: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 dropout: float = 0.1,
                 ffn_dim: int = 2048,
                 position_embedding: str = 'rotate'):
        super(DualFGT, self).__init__()
        self.shared_fgt = FGT(
            num_channels=num_channels,
            T=T,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ffn_dim=ffn_dim,
            position_embedding=position_embedding
        )

    def forward(self, x_ori: torch.Tensor, x_aug: torch.Tensor):
        h_ori = self.shared_fgt(x_ori)
        h_aug = self.shared_fgt(x_aug)
        return h_ori, h_aug
