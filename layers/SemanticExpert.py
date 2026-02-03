# -*- coding = utf-8 -*-
# @Time: 2025/10/18 19:54
# @Author: wisehone
# @File: SemanticExpert.py.py
# @SoftWare: PyCharm
import sys
from typing import Dict, Any, List

sys.path.append('..')
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from abc import ABC, abstractmethod

"""
 more sophisticated
"""


class SemanticExpert(nn.Module, ABC):
    """abstract sematic epxert network"""

    def __init__(self, T: int):
        super(SemanticExpert, self).__init__()
        self.T = T

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_config(self) -> Dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'T': self.T
        }

class ConvExpert(SemanticExpert):
    """Convolutional semantic expert"""

    def __init__(
        self,
        T: int,
        hidden_channels: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.1,
        use_residual: bool = False,
    ):
        super().__init__(T)
        self.use_residual = use_residual

        pad = (kernel_size - 1) // 2

        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size, padding=pad, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=pad, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return (x + y) if self.use_residual else y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"expert_type": "conv"})
        return cfg


class MLPExpert(SemanticExpert):
    """MLP semantic expert"""

    def __init__(
        self,
        T: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__(T)
        self.use_residual = use_residual

        self.mlp = nn.Sequential(
            nn.Linear(T, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, T),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        x_flat = x.squeeze(1)          # [B, T]
        y = self.mlp(x_flat).unsqueeze(1)  # [B, 1, T]
        return (x + y) if self.use_residual else y

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"expert_type": "mlp"})
        return cfg


class SimpleConvExpert(SemanticExpert):
    """Simple convolutional semantic expert"""

    def __init__(self,
                 T: int,
                 hidden_channels: int = 32,
                 kernel_size: int = 5,
                 num_layers: int = 2):
        super(SimpleConvExpert, self).__init__(T)

        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        conv_layers = []
        for i in range(num_layers):
            in_channels = 1 if i == 0 else hidden_channels
            out_channels = 1 if i == num_layers - 1 else hidden_channels

            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2
                )
            )
            if i < num_layers - 1:
                conv_layers.append(nn.ReLU())

        # 添加池化
        # self.pool = nn.AdaptiveAvgPool1d(1)
        # self.conv_net = nn.Sequential(*conv_layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_net(x)
        out2 = self.pool(out1).squeeze(-1)
        output = out2.unsqueeze(1)
        return output

    def get_config(self) -> Dict[str, Any]:
        base_config = super().get_config()
        base_config.update({
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "expert_type": "simple_conv"
        })
        return base_config

class Conv_MLP_Expert(SemanticExpert):
    """Convolutional MLP semantic expert"""
    def __init__(self, T: int,
                 hidden_channels: int = 32,
                 kernel1: int = 3,
                 kernel2: int = 5,
                 kernel3: int = 7,
                 dilation1: int = 1,
                 dilation2: int = 3,
                 dilation3: int = 5,
                 mlp_hidden: int = 128,
                 drop_out: float = 0.2):
        super(Conv_MLP_Expert, self).__init__(T)
        
        self.T = T
        self.num_channels = hidden_channels
        
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=hidden_channels,
            kernel_size=kernel1, padding=(kernel1 - 1) // 2 * dilation1, 
            dilation=dilation1)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel2, padding=(kernel2 - 1) // 2 * dilation2, 
            dilation=dilation2)
        self.conv3 = nn.Conv1d(
            in_channels=hidden_channels, out_channels=hidden_channels,
            kernel_size=kernel3, padding=(kernel3 - 1) // 2 * dilation3, 
            dilation=dilation3)
        
        # out is a mlp
        # [B, C, T]
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_channels, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop_out),
            nn.Linear(mlp_hidden, 1)
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [B, 1, T]
        h = F.gelu(self.conv1(x))
        h = F.gelu(self.conv2(h))
        h = F.gelu(self.conv3(h))   # [B, hidden_channels, T]

        # transpose to [B, T, C] for Linear to operate on channel dim
        h_perm = h.permute(0, 2, 1)  # [B, T, C]
        h_perm = h_perm.contiguous()
        out = self.channel_mlp(h_perm)  # [B, T, 1]

        # transpose back to [B, 1, T]
        out = out.permute(0, 2, 1)
        out = out.contiguous()
        return out
    def get_config(self) -> Dict[str, Any]:
        """
        Return configuration dictionary of this expert, including convolution and MLP parameters.
        Useful for logging, checkpoint metadata, and model reconstruction.
        """
        base_config = super().get_config()
        base_config.update({
            "expert_type": "conv_mlp",
            "hidden_channels": self.hidden_channels,
            "kernel1": getattr(self, "conv1", None).kernel_size[0] if hasattr(self, "conv1") else None,
            "kernel2": getattr(self, "conv2", None).kernel_size[0] if hasattr(self, "conv2") else None,
            "kernel3": getattr(self, "conv3", None).kernel_size[0] if hasattr(self, "conv3") else None,
            "dilation1": getattr(self, "conv1", None).dilation[0] if hasattr(self, "conv1") else None,
            "dilation2": getattr(self, "conv2", None).dilation[0] if hasattr(self, "conv2") else None,
            "dilation3": getattr(self, "conv3", None).dilation[0] if hasattr(self, "conv3") else None,
            "mlp_hidden": getattr(self, "channel_mlp", None)[0].out_features
                        if hasattr(self, "channel_mlp") and isinstance(self.channel_mlp[0], nn.Linear)
                        else None,
        })
        return base_config
