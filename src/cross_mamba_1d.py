# cross_mamba_1d.py
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class CrossMamba1D(nn.Module):
    """
    输入: x1 (B,L,D)  x2 (B,L,D)  已由 patch_embed 展平成序列
    输出: y (B,L,D)  残差已加
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.):
        super().__init__()
        # 两套参数，保持“交叉”味道
        self.ssm1 = BiMamba1D(d_model, d_state, d_conv, expand)
        self.ssm2 = BiMamba1D(d_model, d_state, d_conv, expand)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x1, x2):
        # x1,x2: (B, L, D)
        y1 = self.ssm1(x1)
        y2 = self.ssm2(x2)
        y = self.proj(self.norm(y1 + y2))
        return x1 + self.drop(y)   # 残差


class BiMamba1D(nn.Module):
    """极简双向：正向 + 逆向，结果相加"""
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        self.forward_mamba = Mamba(d_model, d_state, d_conv, expand)
        self.backward_mamba = Mamba(d_model, d_state, d_conv, expand)

    def forward(self, x):
        # x: (B, L, D)
        y1 = self.forward_mamba(x)
        y2 = self.backward_mamba(x.flip(1)).flip(1)
        return y1 + y2