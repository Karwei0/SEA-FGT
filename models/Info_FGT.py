import sys
sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn

from layers.Auto import AutoAUG
from layers.CCE import CCE
from layers.FGT import DualFGT


class Model(nn.Module):
     def __init__(self, args):
          super(Model, self).__init__()

          self.num_channels = args.num_channels
          self.T = args.T
          self.d_model = args.d_model
          self.use_cce = args.use_cce
          self.use_fgt = args.use_fgt
          self.use_info = args.use_sea
     
          if self.use_cce:
               self.cce = CCE(
                num_channels=self.num_channels,
                T=args.T, bin_size=args.bin_size,
                k_sparse=args.k_sparse, use_laplacian=args.use_laplacian
               )

          if self.use_info:
               self.infots = AutoAUG(aug_p1=0.4)
          
          if self.use_fgt:
               self.fgt = DualFGT(
                T=args.T, num_channels=args.num_channels,
                d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, dropout=args.dropout,
                ffn_dim=args.ffn_dim, position_embedding=args.position_embedding,
               )
          else:
               self.simp_proj = nn.Linear(args.num_channels, args.d_model)

          self.config = {'T': args.T, 'num_channels': args.num_channels, 'bin_size': args.bin_size, 'k_sparse': args.k_sparse,
                       'use_laplacian': args.use_laplacian, 'expert_configs': args.expert_configs, 'topk': args.topk,
                       'temperature': args.temperature, 'capacity_factor': args.capacity_factor, 'noise_std': args.noise_std,
                       'prob_threshold': args.prob_threshold, 'use_residual': args.use_residual, 'd_model':args.d_model,
                       'n_heads': args.n_heads, 'n_layers': args.n_layers, 'dropout': args.dropout, 'ffn_dim': args.ffn_dim,
                       'position_embedding': args.position_embedding, 'use_cce': args.use_cce, 'use_sea': args.use_sea,
                       'use_fgt': args.use_fgt}
          
     def forward(self, x: torch):
          """
          Docstring for forward
          x:[B, T, N]
          :param self: Description
          :param x: Description
          :type x: torch
          """
          B, T, N = x.shape
          device = x.device

          assert T == self.T, f'T must be equal to {self.T}, but got {T}'
          assert N == self.num_channels, f'N must be equal to {self.num_channels}, but got {N}'

          x_ = x.transpose(1, 2)
          conherence_matrix = smooth_p = None
          gate_ori = None
          gate_aug = None

          if self.use_cce:
               x_, coherence_matrix = self.cce(x_)
          
          if self.use_info:
               x_ori, x_aug = self.infots(x_)
          else:
               x_ori = x_aug = x_
          
          smooth_p = torch.zeros(B, N, self.num_experts, device=device)

          if self.use_fgt:
               y_ori, y_aug = self.fgt(x_ori, x_aug) # the proj is already in the FGA

          else:
               y_ori = self.simp_proj(x_ori.transpose(1, 2))
               y_ori = self.simp_proj(x_aug.transpose(1, 2))
          
          return y_ori, y_aug, conherence_matrix, smooth_p, x_ori, x_aug, None