# -*- coding = utf-8 -*-
# @Time: 2025/10/18 11:17
# @Author: wisehone
# @File: config.py
# @SoftWare: PyCharm

# TODO: consider more
import copy


dataset2path ={
    'MSL': 'datasets/MSL',
    'SMD': 'datasets/SMD',
    'SMAP': 'datasets/SMAP',
    'SWAT': 'datasets/SWAT',
    'PSM': 'datasets/PSM'
}

dataset2channels = {
    'MSL': 55,
    'SMAP': 25,
    'PSM': 25,
    'SWAT': 51,
    'SMD': 38
}


"""
hidden_channels: int = 32,
kernel1: int = 3,
kernel2: int = 5,
kernel3: int = 7,
dilation1: int = 1,
dilation2: int = 3,
dilation3: int = 5,
"""
"""
one_expert_config = {"type": "conv_mlp", "hidden_channels": 16,
                    'kernel1': 3,'kernel2': 5, 'kernel3': 7, 
                    "dilation1": 1, "dilation2": 3, "dilation3": 5,
                    'mlp_hidden': 128, "dropout": 0.2}
"""

conv_one_expert_config = {
    "type": "conv",
    "hidden_channels": 8,
    "kernel_size": 5,
    "dropout": 0.1,
    "use_residual": True,
}


mlp_one_expert_config = {
    "type": "mlp",
    "hidden_dim": 512,
    "dropout": 0.1,
    "use_residual": True,
}



conv_mlp_one_expert_config = {
    "type": "conv_mlp", "hidden_channels": 8,
    'kernel1': 3,'kernel2': 5, 'kernel3': 7, 
    "dilation1": 1, "dilation2": 3, "dilation3": 5,
    'mlp_hidden': 522, "dropout": 0.2
}
expert_configs=[
    copy.deepcopy(mlp_one_expert_config) for _ in range(10)
]
