import sys

import torch
from torch import nn
import torch.nn.functional as F

sys.path.append('..')
sys.path.append('.')

class KL_loss(nn.Module): 
    def __init__(self, temperature: float = 0.1, **kwargs): 
        super().__init__() 
        self.temperature = temperature 
    def _normalize(self, x: torch.Tensor) -> torch.Tensor: 
            # x_copy = x.clamp_min(1e-8) 
            # return x_copy / x_copy.sum(dim=-1, keepdim=True) 
            return torch.softmax(x / self.temperature, dim=-1)
    def _kl(self, p: torch.Tensor, q: torch.Tensor, eps: float=1e-4) -> torch.Tensor: 
        """ KL(p || q), p/q shape: [B, D] """ 
        res = p * (torch.log(p + eps) - torch.log(q + eps))
        return torch.mean(res, dim=-1)
    

    def forward(self, p_ori: torch.Tensor, q_aug: torch.Tensor) -> torch.Tensor: 
        """ p_ori, q_aug: [B, T, D] """
        B, T, D = p_ori.shape 
        p_ori = self._normalize(p_ori) 
        q_aug = self._normalize(q_aug)
        kl_losses = [] 
        
        for t in range(T): 
            p_t = p_ori[:, t, :] 
            q_t = q_aug[:, t, :]

            # === DCdetector-style stopgrad symmetric KL === # 
            # update p, stop q 
            kl_p = self._kl(p_t, q_t.detach())
            # update q, stop p 
            kl_q = self._kl(q_t, p_t.detach())
            kl_losses.append(0.5 * (kl_p + kl_q)) 
        return torch.mean(torch.stack(kl_losses))
