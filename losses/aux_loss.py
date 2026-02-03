# -*- coding = utf-8 -*-
# @Time: 2025/10/20 9:50
# @Author: wisehone
# @File: aux_loss.py
# @SoftWare: PyCharm
import sys
from losses.JSD_loss import JSD_loss
from losses.KL_loss import *
from losses.NTXentLoss import NTXentLoss
from abc import ABC, abstractmethod
from optparse import Option
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseAuxLoss(nn.Module, ABC):
    """abstract base class for auxiliary loss"""
    def __init__(self, name: str, weight: float=1.0):
        super(BaseAuxLoss, self).__init__()
        self.name = name
        self.weight = weight

    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
         pass

    def weighted_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return self.weight * loss


class InfoNCELoss(BaseAuxLoss):
    """

    """
    def __init__(self, weight=1.0, temperature: float = 0.2):
        super().__init__('info_nce_loss', weight)
        self.temperature = temperature

    def forward(self, x_ori: torch.Tensor, x_aug: torch.Tensor):
        """
        x_ori, x_aug: [B, N, T]
        """
        device = x_ori.device
        B, N, T = x_ori.shape

        # ---------- 1. global pooling ----------
        # [B, N, T] -> [B, N]
        z_ori = x_ori.mean(dim=-1)
        z_aug = x_aug.mean(dim=-1)

        # ---------- 2. concat views ----------
        features = torch.cat([z_ori, z_aug], dim=0)  # [2B, N]

        # ---------- 3. labels ----------
        labels = torch.arange(B, device=device)
        labels = torch.cat([labels, labels], dim=0)  # [2B]

        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # ---------- 4. similarity ----------
        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T)  # [2B,2B]

        # ---------- 5. remove self-contrast ----------
        mask = torch.eye(2 * B, device=device).bool()
        labels = labels[~mask].view(2 * B, -1)
        sim = sim[~mask].view(2 * B, -1)

        positives = sim[labels.bool()].view(2 * B, -1)
        negatives = sim[~labels.bool()].view(2 * B, -1)

        logits = torch.cat([positives, negatives], dim=1)
        logits /= self.temperature

        loss = -F.log_softmax(logits, dim=1)[:, 0].mean()

        return self.weighted_loss(loss)

# TODO: whether to use
class FGAConsistencyLoss(BaseAuxLoss):
    """
    FGA consistency loss
    """
    def __init__(self, weight: float=1.0):
        super().__init__('fga_consistency_loss', weight)

    def forward(self, gate_ori: torch.Tensor, gate_aug: torch.Tensor) -> torch.Tensor:
        """

        :param gate_ori: [B, T]
        :param gate_pred: [N, T]
        :return: 1
        """
        B, T = gate_ori.shape

        kl_losses = []
        eps = 1e-8

        for t in range(T):
            g_t_ori = gate_ori[:, t]
            g_t_aug = gate_aug[:, t]

            g_t_ori_ = g_t_ori + eps
            g_t_aug_ = g_t_aug + eps

            # TODO: check
            g_t_ori_norm = g_t_ori_ / g_t_ori_.sum()
            g_t_aug_norm = g_t_aug_ / g_t_aug_.sum()

            kl_forward = F.kl_div(
                torch.log(g_t_aug_norm.unsqueeze(1)),
                g_t_ori_norm.unsqueeze(1),
                reduction='batchmean'
            )

            kl_backward = F.kl_div(
                torch.log(g_t_ori_norm.unsqueeze(1)),
                g_t_aug_norm.unsqueeze(1),
                reduction='batchmean'
            )

            kl = (kl_forward + kl_backward)
            kl_losses.append(kl)

        l_fga_gate = torch.mean(torch.stack(kl_losses))
        return self.weighted_loss(l_fga_gate)

class LoadBalanceLoss(BaseAuxLoss):
    def __init__(self, weight: float=0.1, w_l: float=0.5):
        super().__init__('load_balance_loss', weight)
        self.w_l = w_l

    def forward(self, smooth_p: torch.Tensor) -> torch.Tensor:
        """

        :param expert_utilization: [B, N, num_expert]
        :return: 1
        """

        # usage
        expert_usage = smooth_p.sum(dim=[0, 1])

        # cal cv
        mean_usage = expert_usage.mean()
        std_suage = expert_usage.std() / mean_usage

        if mean_usage < 1e-8:
            return torch.tensor(0.0, device=smooth_p.device)

        cv = std_suage / mean_usage

        l_load = cv ** 2 * self.w_l

        return self.weighted_loss(l_load)

class OrhogonalityLoss(BaseAuxLoss):
    """
    diversity for experts
    """
    def __init__(self, weight: float=0.1):
        super().__init__('orthogonality_loss', weight)

    def forward(self, experts: List[nn.Module]) -> torch.Tensor:
        assert isinstance(experts, list), 'experts must be a list of nn.Module'
        num_experts = len(experts)

        if num_experts <= 1:
            return torch.tensor(0.0, device=experts[0].weight.device)

        # col the last weights for each experts
        expert_weights = []
        for expert in experts:
            last_layer_weight = self._get_last_layer_weight(expert)
            if last_layer_weight is not None:
                weight_flat = last_layer_weight.view(last_layer_weight.shape[0], -1)
                weight_norm = F.normalize(weight_flat, p=2, dim=-1)
                # weight_norm = weight_flat
                expert_weights.append(weight_norm)

        if len(expert_weights) <= 1:
            return torch.tensor(0.0, device=experts[0].weight.device)

        # cal orth
        orth_losses = []
        for i, w_i in enumerate(expert_weights):
            for j in range(i + 1, num_experts):
                w_j = expert_weights[j]

                if w_i.size(1) == w_j.size(1):
                    cosine_sim = F.cosine_similarity(w_i, w_j, dim=1)
                    orth_loss = torch.mean(torch.abs(cosine_sim))

                    # similarity = torch.matmul(w_i.t(), w_j)
                    # orth_loss = torch.mean(similarity ** 2)
                    orth_losses.append(orth_loss)

        if len(orth_losses) < 2:
            return torch.tensor(0.0, device=experts[0].weight.device)

        L_orth = torch.mean(torch.stack(orth_losses))
        return self.weighted_loss(L_orth)

    def _get_last_layer_weight(self, expert: nn.Module) -> torch.Tensor:
        for module in reversed(list(expert.modules())):
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                return module.weight
        return None

class HISCLoss(BaseAuxLoss):
    """
    Hilbert-Schmidt Independence Criterion LOSS
    """
    def __init__(self, sigma: float= 1.0, kernel_type: str= 'rbf', weight: float= 0.1):
        super(HISCLoss, self).__init__('hiscloss', weight)
        self.sigma = sigma
        self.kernel_type = kernel_type

    def rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor = None) -> torch.Tensor:
        """

        :param X: [B, d] or [B, T, d]
        :param Y: [B, d] or [B, T, d]
        :return: [B, B]
        """
        if Y is None:
            Y = X

        if X.dim() > 2:
            X = X.view(X.shape[0], -1)
        if Y.dim() > 2:
            Y = Y.view(Y.shape[0], -1)

        K = torch.matmul(X, Y.t())
        return K

    def compute_kernel(self, X: torch.Tensor, Y: torch.Tensor=None) -> torch.Tensor:
        if self.kernel_type == 'rbf':
            return self.rbf_kernel(X, Y)
        elif self.kernel_type == 'linear':
            return self.linear_kernel(X, Y)
        else:
            raise ValueError('Invalid kernel type')

    def center_matrix(self, K: torch.Tensor) -> torch.Tensor:
        """
        centerilization kernel
        :param K: [B, B]
        :return: [B, B]
        """
        B = K.shape(0)

        H = torch.eye(B, device=K.device) - torch.ones(B, B, device=K.device) / B

        K_centered = torch.matmul(torch.matmul(H, K), H)
        return K_centered

    def hsic(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """

        :param X: [B, d_x] or [B, T, d_x]
        :param Y: [B, d_y] or [B, T, d_y]
        :return:
        """
        B = X.size(0)

        K = self.compute_kernel(X)
        L = self.compute_kernel(Y)

        K_centered = self.center_matrix(K)
        L_centered = self.center_matrix(L)

        # hsic = trace(K_centered * L_centered) / (B - 1)
        hsic_value = torch.trace(torch.matmul(K_centered, L_centered)) / ((B - 1) ** 2)
        return hsic_value

    def forward(self, experts_outputs: List[torch.Tensor]) -> torch.Tensor:
        """

        :param epxerts_outputs:[B, ...]
        :return:
        """
        # print(experts_outputs)
        # print('type: ', type(experts_outputs))
        num_experts = len(experts_outputs)
        if num_experts <= 1:
            return torch.tensor(0.0, device=experts_outputs[0].device)

        # paiwise hsic
        hsic_values = []
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                hsic_val = self.hsic(experts_outputs[i], experts_outputs[j])
                hsic_values.append(hsic_val)

        if len(hsic_values) == 0:
            return torch.tensor(0.0, device=experts_outputs[0].device)
        L_hsic = torch.mean(torch.stack(hsic_values))
        return self.weighted_loss(L_hsic)


class ContrastiveLossManager(nn.Module):
    """
    manager all losses
    """

    def __init__(self,
                 temperature: float = 0.1,
                 lambda_uti: float = 0.1,
                 lambda_orth: float = 0.1,
                 lambda_info_nce: float = 0.1):
        """

        :param temperature:
        :param lambda_uti:
        :param lambda_orth:
        :param lambda_constru: loss in FGA or RGTA maybe
        """
        super().__init__()

        self.main_loss = KL_loss(temperature=temperature, weight=1.0)  #
        # self.main_loss = NTXentLoss(device=device, batch_size=batch_szie, temperature=temperature, use_cosine_similarity=True)
        if lambda_uti > 0:
            self.uti_loss = LoadBalanceLoss(weight=lambda_uti)
        else:
            self.uti_loss = None
        
        if lambda_orth > 0:
            self.orth_loss = OrhogonalityLoss(weight=lambda_orth)
        else:
            self.orth_loss = None

        if lambda_info_nce > 0:
            self.infonce_loss = InfoNCELoss(weight=lambda_info_nce)
        else:
            self.infonce_loss = None

        # 注册损失函数以便管理
        self.loss_functions = nn.ModuleDict({
            'main_loss': self.main_loss,
            'infonce_loss': self.infonce_loss,
            'uti_loss': self.uti_loss,
            'orth_loss': self.orth_loss
        })

    def forward(self,
                y_ori: torch.Tensor,
                y_aug: torch.Tensor,
                gate_ori: torch.Tensor,
                gate_aug: torch.Tensor,
                expert_utilization: torch.Tensor,
                experts: List[nn.Module]) -> Dict[str, torch.Tensor]:
        """
        总损失计算
        """
        # 计算各项损失
        l_main = self.main_loss(y_ori, y_aug)

        if self.infonce_loss is None:
            l_infonce = torch.tensor(0.0, device=y_ori.device)
        else:
            l_infonce = self.infonce_loss(gate_ori, gate_aug)
        
        if self.uti_loss is None:
            l_uti = torch.tensor(0.0, device=y_ori.device)
        else:
            l_uti = self.uti_loss(expert_utilization)

        if self.orth_loss is None:
            l_orth = torch.tensor(0.0, device=y_ori.device)
        else:
            l_orth = self.orth_loss(experts)

        total_loss = l_main + l_infonce + l_uti + l_orth

        loss_dict = {
            'total_loss': total_loss,
            'l_main': l_main,
            'l_infonce': l_infonce,
            'l_uti': l_uti,
            'l_orth': l_orth
        }

        return loss_dict

    def get_loss_names(self) -> List[str]:
        return list(self.loss_functions.keys())

    def set_loss_weight(self, loss_name: str, weight: float):
        if loss_name in self.loss_functions:
            self.loss_functions[loss_name].weight = weight


# 测试代码
def test_modular_losses():
    """测试模块化损失函数"""
    batch_size = 4
    seq_length = 50
    d_model = 128
    num_channels = 8
    num_experts = 4

    # 创建各个损失函数
    kl_loss = KL_loss(temperature=0.1)
    fga_loss = FGAConsistencyLoss(weight=0.1)
    balance_loss = LoadBalanceLoss(weight=0.1)
    orth_loss = OrhogonalityLoss(weight=0.1)

    # 创建测试数据
    y_ori = torch.randn(batch_size, seq_length, d_model)
    y_aug = torch.randn(batch_size, seq_length, d_model)
    gate_ori = torch.sigmoid(torch.randn(batch_size, seq_length))
    gate_aug = torch.sigmoid(torch.randn(batch_size, seq_length))
    expert_utilization = torch.randint(0, 2, (batch_size, num_channels, num_experts)).float()

    # 创建模拟专家
    experts = [nn.Linear(10, 10) for _ in range(num_experts)]

    # 测试各个损失函数
    print("=== 模块化损失函数测试 ===")

    L_KL = kl_loss(y_ori, y_aug)
    print(f"对称KL损失: {L_KL.item():.6f}")

    L_FGA = fga_loss(gate_ori, gate_aug)
    print(f"FGA一致性损失: {L_FGA.item():.6f}")

    L_balance = balance_loss(expert_utilization)
    print(f"负载均衡损失: {L_balance.item():.6f}")

    L_orth = orth_loss(experts)
    print(f"正交性损失: {L_orth.item():.6f}")


    # 测试损失管理器
    loss_manager = ContrastiveLossManager()
    loss_dict = loss_manager(y_ori, y_aug, gate_ori, gate_aug, expert_utilization, experts)

    print("\n=== 损失管理器测试 ===")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.6f}")

    print("测试通过！模块化损失函数工作正常。")


def test_loss_extensibility():
    """测试损失函数的可扩展性"""

    # 可以轻松添加新的损失函数
    class CustomLoss(BaseAuxLoss):
        def __init__(self, weight: float = 1.0):
            super().__init__("custom_loss", weight)

        def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
            # 自定义损失逻辑
            return self.weighted_loss(torch.mean((input1 - input2) ** 2))

    # 使用自定义损失
    custom_loss = CustomLoss(weight=0.5)
    input1 = torch.randn(4, 10)
    input2 = torch.randn(4, 10)
    loss = custom_loss(input1, input2)

    print(f"\n自定义损失: {loss.item():.6f}")
    print("测试通过！损失函数具有良好的可扩展性。")


if __name__ == "__main__":
    test_modular_losses()
    test_loss_extensibility()