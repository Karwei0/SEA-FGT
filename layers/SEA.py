# -*- coding = utf-8 -*-
# @Time: 2025/10/18 19:54
# @Author: wisehone
# @File: SEA.py.py
# @SoftWare: PyCharm
import sys
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')
from layers.SemanticExpert import *

# Semantic Expert Ensemble
class SEE(nn.Module):
    """
    Semantic Expert Ensemble.
    - Takes routed expert indices and weights from SEA.
    - Only computes outputs for valid (compute_mask=True) routes.
    - Passthrough routes copy input directly.
    - Keeps gradients for expert parameters (no @torch.no_grad()).
    - Avoids in-place ops to prevent autograd issues.
    """
    def __init__(self, expert_configs: List[Dict[str, Any]], T: int):
        super().__init__()
        self.num_experts = len(expert_configs)
        self.T = T
        self.route_scale = nn.Parameter(torch.tensor(1.2), requires_grad=True)

        # === Build expert ensemble ===
        self.experts = nn.ModuleList()
        for cfg in expert_configs:
            etype = cfg.get("type", "conv")
            if etype == "conv":
                self.experts.append(
                    ConvExpert(
                        T=T,
                        hidden_channels=cfg.get("hidden_channels", 32),
                        kernel_size=cfg.get("kernel_size", 5),
                        dropout=cfg.get("dropout", 0.1),
                        use_residual=cfg.get("use_residual", False),
                    )
                )
            elif etype == "mlp":
                # TODO: think about the T and the dim of mlp ??
                self.experts.append(
                    MLPExpert(
                        T=T,
                        hidden_dim=cfg.get("hidden_dim", 128),
                        dropout=cfg.get("dropout", 0.1),
                        use_residual=cfg.get("use_residual", True),
                    )
                )
            elif etype == "simple_conv":
                self.experts.append(
                    SimpleConvExpert(
                        T=T,
                        hidden_channels=cfg.get("hidden_channels", 32),
                        kernel_size=cfg.get("kernel_size", 5),
                        num_layers=cfg.get("num_layers", 2),
                    )
                )
            elif etype == "conv_mlp":
                self.experts.append(
                    Conv_MLP_Expert(
                        T=T,
                        hidden_channels=cfg.get("hidden_channels", 32),
                        kernel1=cfg.get("kernel1", 3),
                        kernel2=cfg.get("kernel2", 5),
                        kernel3=cfg.get("kernel3", 7),
                        dilation1=cfg.get("dilation1", 1),
                        dilation2=cfg.get("dilation2", 3),
                        dilation3=cfg.get("dilation3", 5),
                        mlp_hidden=cfg.get("mlp_hidden", 128),
                        drop_out=cfg.get("drop_out", 0.2),
                    )
                )
            else:
                raise ValueError(f"Unsupported expert type: {etype}")

    def forward(
        self,
        x: torch.Tensor,           # [B, N, T]
        chosen_idx: torch.Tensor,  # [B, N, K]
        chosen_w: torch.Tensor,    # [B, N, K]
        compute_mask: torch.Tensor # [B, N, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of expert ensemble.
        Only active (compute_mask=True) routes go through experts.
        Others are passthrough (copy input).
        """
        device = x.device
        B, N, T = x.shape
        E = self.num_experts
        K = chosen_idx.size(-1)
        BN = B * N

        # === Flatten input for easier indexing ===
        x_flat = x.contiguous().view(BN, 1, T)
        idx_flat = chosen_idx.view(BN, K)
        w_flat = chosen_w.view(BN, K)
        cm_flat = compute_mask.view(BN, K)

        # === Initialize tensors ===
        # Start with passthrough copy; will replace compute positions later
        enhanced_contrib = x_flat.unsqueeze(1).repeat(1, K, 1, 1)
        final_expert_util = torch.zeros(BN, E, device=device)

        # === Compute outputs for each expert ===
        for e in range(E):
            # Find samples routed to this expert
            sel_mask = (idx_flat == e) & cm_flat
            if not sel_mask.any():
                continue

            # Extract active batch indices and route positions
            bn_idx, k_idx = torch.nonzero(sel_mask, as_tuple=True)

            # Forward through expert (keep grad)
            x_in = x_flat[bn_idx]
            y = self.experts[e](x_in)
            if y.dim() == 2:
                y = y.unsqueeze(1)

            # Safely assign results (no in-place)
            enhanced_contrib = enhanced_contrib.index_put(
                (bn_idx, k_idx), y, accumulate=False
            )

            # Update utilization count
            util_add = torch.zeros_like(final_expert_util)
            util_add[bn_idx, e] = 1.0
            final_expert_util = final_expert_util + util_add

        # === Weighted combination ===
        w_sum = w_flat.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        norm_w = w_flat / w_sum

        # Weighted average over K selected routes
        mixed = (enhanced_contrib * norm_w.view(BN, K, 1, 1)).sum(dim=1)
        enhanced_output = (mixed * self.route_scale).view(B, N, T)
        final_expert_util = final_expert_util.view(B, N, E)

        return enhanced_output, final_expert_util


# =========================================================
# Semantic Gate
# =========================================================
class SG(nn.Module):
    """
    Semantic Gate network.
    - Produces logits and softmax probabilities over experts.
    - Supports temperature annealing and Gaussian noise for exploration.
    - Returns both full probabilities and top-k selections.
    """
    def __init__(self,
                 T: int,
                 num_channels: int,
                 num_experts: int,
                 temperature: float = 1.0,
                 noise_std: float = 0.1,
                 prob_threshold: float = 0.05,
                 auto_anneal: bool = True,
                 t_max: float = 10.0,
                 t_min: float = 0.2,
                 total_steps: int = 10000,
                 anneal_mode: str = 'cosine'):
        super().__init__()
        self.T = T
        self.num_experts = num_experts
        self.num_channels = num_channels

        # Gating and annealing settings
        self.noise_std = noise_std
        self.prob_threshold = prob_threshold
        self.auto_anneal = auto_anneal
        self.t_max, self.t_min = t_max, t_min
        self.temperature = temperature
        self.anneal_mode = anneal_mode
        self.total_steps = max(1, total_steps)
        self._auto_step = 0

        # Simple MLP-based gate
        self.gate_net = nn.Linear(self.T, self.num_experts)
        # self.gate_net = nn.Sequential(
        #     nn.Linear(self.T, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.num_experts)
        # )

    def _update_temperature(self):
        """Cosine or linear decay of temperature over training steps."""
        if not self.auto_anneal:
            return
        p = min(1.0, self._auto_step / float(self.total_steps))
        if self.anneal_mode == 'cosine':
            self.temperature = self.t_min + 0.5 * (self.t_max - self.t_min) * (1.0 + math.cos(math.pi * p))
        else:
            self.temperature = self.t_max * (1.0 - p) + self.t_min * p
        self._auto_step += 1

    def add_gumbel_noise(self, logits: torch.Tensor):
        """Add Gaussian noise to encourage exploration."""
        if self.training and self.noise_std > 0:
            return logits + torch.randn_like(logits) * self.noise_std
        return logits

    def forward(self, x: torch.Tensor, topk: int = 2):
        """
        Forward pass of the gate.
        Returns:
            - expert_weights: full distribution over experts [B, N, E]
            - topk_indices: selected expert indices [B, N, K]
            - topk_weights_norm: normalized top-k probabilities [B, N, K]
        """
        self._update_temperature()
        temp = max(1e-6, float(self.temperature))

        B, N, T = x.shape
        logits = self.gate_net(x.contiguous().view(B * N, T))
        logits = self.add_gumbel_noise(logits) / temp

        expert_weights = F.softmax(logits, dim=-1)
        K = min(topk, expert_weights.size(-1))
        topk_weights, topk_indices = torch.topk(expert_weights, k=K, dim=-1)
        topk_weights_norm = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-12)
        
        kth_val = topk_weights[:, -1].unsqueeze(-1)
        sigma = self.noise_std if self.noise_std > 0 else 1e-6
        smooth_p = 0.5 * (1 + torch.erf((logits - kth_val) / (sigma * math.sqrt(2.0))))

        return (
            expert_weights.view(B, N, self.num_experts),
            topk_indices.view(B, N, K),
            topk_weights_norm.view(B, N, K),
            smooth_p.view(B, N, self.num_experts)
        )


# =========================================================
# Semantic Expert Augmentation
# =========================================================
class SEA(nn.Module):
    """
    Semantic Expert Augmentation.
    - Combines gating, routing, capacity management, and re-sampling.
    - Routes inputs to experts based on gate output and capacity.
    - Re-samples when probability < threshold or expert is full.
    - Avoids duplicate experts per input.
    """
    def __init__(self,
                 num_channels: int,
                 T: int,
                 expert_configs: List[Dict[str, Any]],
                 topk: int = 2,
                 temperature: float = 1.0,
                 capacity_factor: float = 1.5,
                 noise_std: float = 0.1,
                 prob_threshold: float = 0.05,
                 use_residual: bool = True,
                 t_max: float = 10.0,
                 t_min: float = 0.2):
        super().__init__()

        self.num_channels = num_channels
        self.T = T
        self.topk = topk
        self.num_experts = len(expert_configs)
        self.capacity_factor = capacity_factor
        self.prob_threshold = prob_threshold
        self.use_residual = use_residual

        # Gating and expert ensemble
        self.sg = SG(
            T=T,
            num_channels=num_channels,
            num_experts=self.num_experts,
            temperature=temperature,
            noise_std=noise_std,
            prob_threshold=prob_threshold,
            t_max=t_max,
            t_min=t_min,
        )
        self.expert_ensemble = SEE(expert_configs, T)
        self.expert_utils = torch.ones(self.num_experts)

    def calculate_expert_capacity(self, batch_size: int) -> List[int]:
        """
        Compute per-expert capacity.
        Only decremented when an expert actually processes data.
        """
        S = batch_size * self.num_channels
        cap = int(math.ceil(self.capacity_factor * (self.topk * S) / max(self.num_experts, 1)))
        return [cap for _ in range(self.num_experts)]

    @torch.no_grad()
    def _sample_from_masked_probs(self, probs: torch.Tensor, valid_mask: torch.Tensor) -> int:
        """Sample a new expert index from masked probs; return -1 if none valid."""
        p = probs * valid_mask.to(probs.dtype)
        s = p.sum()
        if s <= 0:
            return -1
        p = p / s
        return int(torch.multinomial(p, 1).item())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass:
        1. Compute gating probabilities.
        2. Route inputs to experts based on capacity and prob threshold.
        3. Only valid routes execute expert forward.
        """
        B, N, T = x.shape
        if N == self.T and T == self.num_channels:
            x = x.transpose(1, 2)

        # 1) Compute expert capacity (Python list)
        cap_left = self.calculate_expert_capacity(B)

        # 2) Gating output: full probabilities + top-k selection
        full_probs, topk_indices, topk_weights, smooth_p = self.sg(x, topk=self.topk)
        B_, N_, E = full_probs.shape
        K = topk_indices.size(-1)
        BN = B_ * N_

        probs_flat = full_probs.view(BN, E)
        idx_flat = topk_indices.view(BN, K).clone()
        w_flat = topk_weights.view(BN, K)

        device = x.device
        picked_mask = torch.zeros(BN, E, dtype=torch.bool, device=device)
        compute_mask = torch.zeros(BN, K, dtype=torch.bool, device=device)

        # 3) Assign routes (capacity-aware and prob-threshold-aware)
        for r in range(BN):
            for k in range(K):
                e = int(idx_flat[r, k])
                prob = float(w_flat[r, k])

                # Need re-route if low prob, full capacity, or already picked
                need_reroute = (prob < self.prob_threshold) or (cap_left[e] <= 0) or picked_mask[r, e]

                if not need_reroute:
                    compute_mask[r, k] = True
                    picked_mask[r, e] = True
                    cap_left[e] = max(0, cap_left[e] - 1)
                    continue

                # Mask out used or full experts
                valid = (~picked_mask[r]) & torch.tensor([c > 0 for c in cap_left], device=device)
                new_e = self._sample_from_masked_probs(probs_flat[r], valid)

                if new_e >= 0:
                    # Found a valid new expert
                    idx_flat[r, k] = new_e
                    compute_mask[r, k] = True
                    picked_mask[r, new_e] = True
                    cap_left[new_e] = max(0, cap_left[new_e] - 1)
                # else: passthrough (no compute)

        # 4) Reshape to original dimensions
        chosen_idx = idx_flat.view(B, N, K)
        chosen_w = w_flat.view(B, N, K)
        compute_mask_ = compute_mask.view(B, N, K)

        # 5) Compute experts' outputs
        enhanced_output, util = self.expert_ensemble(
            x=x,
            chosen_idx=chosen_idx,
            chosen_w=chosen_w,
            compute_mask=compute_mask_,
        )

        # 6) Optional residual connection
        if self.use_residual:
            # enhanced_output = 0.2 * x + 0.8 * enhanced_output
            enhanced_output = 0.2 * (enhanced_output + x)
            # enhanced_output = enhanced_output

        self.expert_utils = util
        return enhanced_output, smooth_p

        # helpers
    def get_expert_info(self) -> List[Dict[str, Any]]:
        return self.expert_ensemble.get_expert_info()

    def get_SE(self) -> List[nn.Module]:
        return list(self.expert_ensemble.experts)

    def get_SE_state_dicts(self) -> List[Dict[str, torch.Tensor]]:
        return [e.state_dict() for e in self.expert_ensemble.experts]

    def get_expert_util(self) -> torch.Tensor:
        return self.expert_utils

# 测试代码
def test_enhanced_sea():
    batch_size = 64
    num_channels = 80
    seq_length = 2

    # 创建专家配置
    expert_configs = [
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
        {"type": "conv", "hidden_channels": 1, "dilation": 1},
    ]

    sea = SEA(
        num_channels=num_channels,
        T=seq_length,
        expert_configs=expert_configs,
        topk=2,
        capacity_factor=1.25,  # 容量因子
        prob_threshold=0.05 # 概率阈值
    )

    x = torch.randn(batch_size, num_channels, seq_length)
    x_aug, smooth_p = sea(x)
    
    expert_utilization = sea.get_expert_util()


    print(f"输入形状: {x.shape}")
    print(f"输出形状: {x_aug.shape}")
    print(f"平滑专家利用率形状: {expert_utilization.shape}")
    # 计算每个专家被使用的次数
    expert_usage = expert_utilization.sum(dim=[0, 1])
    print(f"每个专家被使用次数: {expert_usage}")

    expert_utils = sea.get_expert_util()
    expert_utils = expert_utils.sum(dim=[0, 1])
    print(f"每个专家实际被使用次数: {expert_utils}")

    # 验证输出形状
    assert x_aug.shape == (batch_size, num_channels, seq_length)
    assert expert_utilization.shape == (batch_size, num_channels, len(expert_configs))

    print("测试通过！增强版SEA模块（专家容量和概率化路由）工作正常。")


if __name__ == "__main__":
    test_enhanced_sea()