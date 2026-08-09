"""Neural network modules shared by RNADGG experiments."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device) * -scale)
        emb = t[:, None].float() * freqs[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=1)


class AttentionBlock(nn.Module):
    """Lightweight 1D self-attention block."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        qkv = self.qkv(self.norm(x)).view(batch, 3, channels, length)
        q, k, v = qkv.unbind(1)
        attn = torch.einsum("bcl,bck->blk", q, k) * (channels ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("blk,bck->bcl", attn, v)
        return x + self.out(out)


class ResidualBlock(nn.Module):
    """Time-conditioned residual block for 1D sequence tensors."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = h + self.time_mlp(t)[:, :, None]
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UpDownBlock(nn.Module):
    """Residual-attention block followed by upsampling or downsampling."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int, down: bool):
        super().__init__()
        self.residual = ResidualBlock(in_channels, out_channels, time_dim)
        self.attention = AttentionBlock(out_channels)
        if down:
            self.sample = nn.Conv1d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.sample = nn.ConvTranspose1d(out_channels, out_channels, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.sample(self.attention(self.residual(x, t)))


class UNet1D(nn.Module):
    """1D U-Net denoiser for one-hot-like regulatory RNA sequence tensors."""

    def __init__(self, alphabet_size: int = 4, channels: int = 64):
        super().__init__()
        time_dim = channels * 4
        self.time_embedding = nn.Sequential(
            TimeEmbedding(channels),
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv1d(alphabet_size, channels, 3, padding=1)
        self.down1 = UpDownBlock(channels, channels * 2, time_dim, down=True)
        self.down2 = UpDownBlock(channels * 2, channels * 4, time_dim, down=True)
        self.mid1 = ResidualBlock(channels * 4, channels * 4, time_dim)
        self.mid_attention = AttentionBlock(channels * 4)
        self.mid2 = ResidualBlock(channels * 4, channels * 4, time_dim)
        self.up1 = UpDownBlock(channels * 8, channels * 2, time_dim, down=False)
        self.up2 = UpDownBlock(channels * 4, channels, time_dim, down=False)
        self.out = nn.Conv1d(channels * 2, alphabet_size, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        x0 = self.in_conv(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.down2(x1, t_emb)
        xm = self.mid1(x2, t_emb)
        xm = self.mid_attention(xm)
        xm = self.mid2(xm, t_emb)
        u1 = self.up1(torch.cat([xm, x2], dim=1), t_emb)
        u1 = F.interpolate(u1, size=x1.shape[2])
        u2 = self.up2(torch.cat([u1, x1], dim=1), t_emb)
        u2 = F.interpolate(u2, size=x0.shape[2])
        return self.out(torch.cat([u2, x0], dim=1))


class OracleCNN(nn.Module):
    """CNN sequence-to-function predictor used for gradient guidance."""

    def __init__(self, sequence_length: int, alphabet_size: int = 4, dropout: float = 0.3):
        super().__init__()
        pooled_length = max(sequence_length // 4, 1)
        self.features = nn.Sequential(
            nn.Conv1d(alphabet_size, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(128 * pooled_length, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))
