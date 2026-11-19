"""Manuscript U-Net port, retaining archived toehold checkpoint keys."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(
            torch.arange(half, device=timestep.device) * (-math.log(10000) / (half - 1))
        )
        angles = timestep[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class AttentionBlock(nn.Module):
    """Single-head self-attention over sequence positions."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, 3, channels, length)
        query, key, value = qkv.unbind(1)
        attention = torch.einsum("bcl,bck->blk", query, key) * channels**-0.5
        attention = attention.softmax(dim=-1)
        attended = torch.einsum("blk,bck->bcl", attention, value)
        return x + self.out(attended)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(x)))
        hidden = hidden + self.time(time_embedding)[:, :, None]
        hidden = F.silu(self.norm2(self.conv2(hidden)))
        return hidden + self.shortcut(x)


class UpDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, down: bool):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_dim)
        self.attention = AttentionBlock(out_channels)
        self.sample = (
            nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
            if down
            else nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        return self.sample(self.attention(self.residual(x, time_embedding)))


class UNet1D(nn.Module):
    def __init__(self, base_channels: int = 64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embedding = nn.Sequential(
            TimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv1d(4, base_channels, 3, padding=1)
        self.down1 = UpDownBlock(base_channels, base_channels * 2, time_dim, down=True)
        self.down2 = UpDownBlock(base_channels * 2, base_channels * 4, time_dim, down=True)
        self.mid1 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.mid_attention = AttentionBlock(base_channels * 4)
        self.mid2 = ResidualBlock(base_channels * 4, base_channels * 4, time_dim)
        self.up1 = UpDownBlock(base_channels * 8, base_channels * 2, time_dim, down=False)
        self.up2 = UpDownBlock(base_channels * 4, base_channels, time_dim, down=False)
        self.output_residual = ResidualBlock(base_channels * 2, base_channels, time_dim)
        self.output = nn.Conv1d(base_channels, 4, 1)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timestep)
        x0 = self.input(x)
        x1 = self.down1(x0, time_embedding)
        x2 = self.down2(x1, time_embedding)
        middle = self.mid2(self.mid_attention(self.mid1(x2, time_embedding)), time_embedding)
        up = self.up1(torch.cat((middle, x2), dim=1), time_embedding)
        if up.shape[2] != x1.shape[2]:
            up = F.interpolate(up, size=x1.shape[2], mode="linear", align_corners=False)
        up = self.up2(torch.cat((up, x1), dim=1), time_embedding)
        if up.shape[2] != x0.shape[2]:
            up = F.interpolate(up, size=x0.shape[2], mode="linear", align_corners=False)
        return self.output(self.output_residual(torch.cat((up, x0), dim=1), time_embedding))
