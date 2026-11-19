"""GRU generator and CNN discriminator recovered from the UTR comparison driver."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..utils import set_seed


class RNNSeqGenerator(nn.Module):
    def __init__(self, sequence_length: int, z_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.L = sequence_length
        self.h0 = nn.Linear(z_dim, hidden)
        self.in_proj = nn.Linear(z_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.lm = nn.Linear(hidden, 4)

    def forward(self, z):
        initial = torch.tanh(self.h0(z)).unsqueeze(0)
        inputs = self.in_proj(z).unsqueeze(1).expand(z.shape[0], self.L, -1)
        hidden, _ = self.gru(inputs, initial)
        return self.lm(hidden).permute(0, 2, 1)


class SeqDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(4, 64, 7, padding=3), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 5, padding=2), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_gan(features, device, seed, epochs=40, batch_size=128):
    set_seed(seed + 99)
    generator = RNNSeqGenerator(features.shape[-1]).to(device)
    discriminator = SeqDiscriminator().to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loader = DataLoader(TensorDataset(features), batch_size=batch_size, shuffle=True, drop_last=True)
    if len(loader) == 0:
        raise ValueError("GAN training needs at least one full minibatch.")
    for epoch in range(epochs):
        for (real,) in loader:
            real = real.to(device)
            z = torch.randn(real.shape[0], 32, device=device)
            fake = generator(z).softmax(dim=1)
            discriminator_optimizer.zero_grad()
            real_loss = F.binary_cross_entropy_with_logits(
                discriminator(real), torch.full((len(real),), 0.9, device=device)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                discriminator(fake.detach()), torch.full((len(real),), 0.1, device=device)
            )
            (real_loss + fake_loss).backward()
            discriminator_optimizer.step()
            generator_optimizer.zero_grad()
            fake = generator(z).softmax(dim=1)
            generator_loss = F.binary_cross_entropy_with_logits(
                discriminator(fake), torch.ones(len(real), device=device)
            )
            generator_loss.backward()
            generator_optimizer.step()
        print(f"[gan] epoch {epoch + 1}/{epochs}", flush=True)
    return generator.eval(), discriminator.eval()


@torch.no_grad()
def sample_gan(generator, count, batch_size, device):
    batches = []
    for start in range(0, count, batch_size):
        size = min(batch_size, count - start)
        tokens = generator(torch.randn(size, 32, device=device)).argmax(dim=1)
        batches.append(F.one_hot(tokens, num_classes=4).float().permute(0, 2, 1).cpu())
    return torch.cat(batches)
