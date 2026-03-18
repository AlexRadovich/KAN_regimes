import torch
import torch.nn as nn
import torch.nn.functional as F

class RegimeDetector(nn.Module):
    def __init__(self, input_dim: int, n_regimes: int = 3, tau: float = 0.5):
        super().__init__()
        self.n_regimes = n_regimes
        self.tau = tau
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_regimes)
        )

    def forward(self, x):
        logits = self.encoder(x)
        if self.training:
            # Gumbel-Softmax: differentiable discrete sampling
            gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
            regime_probs = F.softmax((logits + gumbel) / self.tau, dim=-1)
        else:
            regime_probs = F.softmax(logits, dim=-1)
        regime_hard = regime_probs.argmax(dim=-1)
        return regime_probs, regime_hard