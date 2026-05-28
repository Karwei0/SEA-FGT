# -*- coding = utf-8 -*-
# @Time: 2025/10/21 14:53
# @Author: wisehone
# @File: SEA_FGA.py
# @SoftWare: PyCharm
import sys
sys.path.append("..")
sys.path.append('.')

import torch
import torch.nn as  nn
import torch.nn.functional as F

from typing import Dict, Any, List, Literal
from layers.CCE import CCE
from layers.FGT import DualFGT
from layers.SEA import SEA
from layers.Transformer import DualTransformer
from layers.PositionEmbedding import *

# for generic
class Model(nn.Module):
    """
    input [B, T, N]
    X -> CCE -> SEA ->X_aug -> FGT |
                |                  |-——> loss
              X_ori ---------> FGT |
    args:
        T: int, num_channels: int, # basic parameters
         bin_size: int, k_sparse: int, use_laplacian: bool, # CCE parameters
         expert_configs: List[Dict[str, Any]], topk: int, temperature: float,
         capacity_factor: float, noise_std: float, prob_threshold: float, use_residual: bool, # SEA parameters
         d_model: int, n_heads: int, n_layers: int, dropout: float, ffn_dim: int, position_embedding: str, stat_mode: Literal['avg', r'max'] = 'avg', # FGA
         use_cce: bool = True, use_sea: bool = True, use_fga: bool = True
    """
    def __init__(self, args):
        super(Model, self).__init__()

        self.num_channels = args.num_channels
        self.T = args.T
        self.d_model = args.d_model
        self.use_cce = args.use_cce
        self.use_sea = args.use_sea
        self.use_fgt = args.use_fgt
        self.num_experts = len(args.expert_configs)
        self.cnt = 0

        if self.use_cce:
            self.cce = CCE(
                num_channels=self.num_channels,
                T=args.T, bin_size=args.bin_size,
                k_sparse=args.k_sparse, use_laplacian=args.use_laplacian
            )

        if self.use_sea:
            if args.expert_configs is None:
                args.expert_configs = [{'type': 'conv', 'hidden_channels': 32, 'dilation': 2} for _ in range(8)]
            self.sea = SEA(
                T=args.T, num_channels=args.num_channels, expert_configs=args.expert_configs, topk=args.topk,
                temperature=args.temperature,
                capacity_factor=args.capacity_factor, noise_std=args.noise_std, prob_threshold=args.prob_threshold,
            )

        if self.use_fgt:
            self.fgt = DualFGT(
                T=args.T, num_channels=args.num_channels,
                d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=args.dropout,
                ffn_dim=args.ffn_dim, position_embedding=args.position_embedding,
            )
        else:
            self.simp_proj = DualTransformer(
                T=args.T, num_channels=args.num_channels,
                d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=args.dropout,
                ffn_dim=args.ffn_dim, position_embedding=args.position_embedding,
            )
            
            # nn.Linear(args.num_channels, args.d_model)

        # save all parameters
        self.config = {'T': args.T, 'num_channels': args.num_channels, 'bin_size': args.bin_size, 'k_sparse': args.k_sparse,
                       'use_laplacian': args.use_laplacian, 'expert_configs': args.expert_configs, 'topk': args.topk,
                       'temperature': args.temperature, 'capacity_factor': args.capacity_factor, 'noise_std': args.noise_std,
                       'prob_threshold': args.prob_threshold, 'use_residual': args.use_residual, 'd_model':args.d_model,
                       'n_heads': args.n_heads, 'n_layers': args.n_layers, 'dropout': args.dropout, 'ffn_dim': args.ffn_dim,
                       'position_embedding': args.position_embedding, 'use_cce': args.use_cce, 'use_sea': args.use_sea,
                       'use_fgt': args.use_fgt}

    def forward(self, x: torch.Tensor):
        """

        :param x: [B, T, N]
        :return:
        """
        B, T, N = x.shape
        device = x.device

        assert T == self.T, f'T must be equal to {self.T}, but got {T}'
        assert N == self.num_channels, f'N must be equal to {self.num_channels}, but got {N}'

        x_ = x.transpose(2, 1)
        coherence_matrix = None
        smooth_p = None

        if self.use_cce:
            x_, coherence_matrix = self.cce(x_) # [B, T, N], [B, N, N]

        if self.use_sea:
            x_aug, smooth_p = self.sea(x_)
            x_ori = x_
        else:
            pass

        if self.use_fgt:
            y_ori, y_aug = self.fgt(x_ori, x_aug) # the proj is already in the FGA
        else:
            pass

        if not self.use_sea:
            return y_ori, y_aug, coherence_matrix, smooth_p, None, None, None

        return y_ori, y_aug, coherence_matrix, smooth_p, None, None, self.sea.get_SE()

