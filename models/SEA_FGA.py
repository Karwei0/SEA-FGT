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
from layers.SEA import SEA
from layers.PositionEmbedding import *

# for generic
class Model(nn.Module):
    """
    input [B, T, N]
    X -> CCE -> SEA ->X_aug -> FGA |
                |                  |-——> loss
              X_ori ---------> FGA |
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
        self.use_fga = args.use_fga
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
                expert_configs = [{'type': 'conv', 'hidden_channels': 32, 'dilation': 2} for _ in range(8)]
            self.sea = SEA(
                T=args.T, num_channels=args.num_channels, expert_configs=args.expert_configs, topk=args.topk,
                temperature=args.temperature,
                capacity_factor=args.capacity_factor, noise_std=args.noise_std, prob_threshold=args.prob_threshold,
            )

        # save all parameters
        self.config = {'T': args.T, 'num_channels': args.num_channels, 'bin_size': args.bin_size, 'k_sparse': args.k_sparse,
                       'use_laplacian': args.use_laplacian, 'expert_configs': args.expert_configs, 'topk': args.topk,
                       'temperature': args.temperature, 'capacity_factor': args.capacity_factor, 'noise_std': args.noise_std,
                       'prob_threshold': args.prob_threshold, 'use_residual': args.use_residual, 'd_model':args.d_model,
                       'n_heads': args.n_heads, 'n_layers': args.n_layers, 'dropout': args.dropout, 'ffn_dim': args.ffn_dim,
                       'position_embedding': args.position_embedding, 'use_cce': args.use_cce, 'use_sea': args.use_sea,
                       'use_fga': args.use_fga}

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
        expert_utilization = None
        gate_ori = None
        gate_aug = None

        if self.use_cce:
            x_, coherence_matrix = self.cce(x_) # [B, T, N], [B, N, N]

        if self.use_sea:
            x_aug, expert_utilization = self.sea(x_)
            x_ori = x_
        else:
            x_ori = x_aug = x_
            expert_utilization = torch.zeros(B, N, self.num_experts, device=device)

        if self.use_fga:
            y_ori, y_aug = self.fga(x_ori, x_aug) # the proj is already in the FGA

            # TODO: get the gate
            gate_ori = self.fga.shared_fga.fg(x_ori)
            gate_aug = self.fga.shared_fga.fg(x_aug)
            # gate_ori, gate_aug = self.fga.get_gate() get fre gate from FGA !!undergoing
        else:
            simp_proj = nn.Linear(N, self.d_model).to(device)
            y_ori = simp_proj(x_ori.transpose(1, 2)) # []
            y_aug = simp_proj(x_aug.transpose(1, 2))

        return y_ori, y_aug, coherence_matrix, expert_utilization, gate_ori, gate_aug, self.sea.get_SE()

