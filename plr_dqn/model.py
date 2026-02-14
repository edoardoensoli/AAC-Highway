"""
Dueling DQN Q-Network for highway-env Kinematics observations.

Observation: flattened (vehicles_count × features) = 70-dim vector
  10 vehicles × 7 features (presence, x, y, vx, vy, cos_h, sin_h)
Action space: Discrete(5) — LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER

Architecture (Dueling, Wang et al. 2016):
  features : Linear(70, 256) → ReLU → Linear(256, 256) → ReLU
  value    : Linear(256, 1)
  advantage: Linear(256, 5)
  Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def init_weights(module: nn.Module, gain: float = np.sqrt(2)):
    """Orthogonal weight init."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class QNetwork(nn.Module):
    """
    Dueling Double DQN Q-Network.

    Parameters
    ----------
    obs_shape : tuple
        Flattened observation shape, e.g. (70,).
    num_actions : int
        Number of discrete actions.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(self, obs_shape: tuple, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        obs_dim = int(np.prod(obs_shape))

        self.features = nn.Sequential(
            init_weights(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            init_weights(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.value_head = init_weights(nn.Linear(hidden_dim, 1), gain=1.0)
        self.advantage_head = init_weights(nn.Linear(hidden_dim, num_actions), gain=0.01)

        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions: (B, num_actions)."""
        feat = self.features(obs)
        value = self.value_head(feat)
        advantage = self.advantage_head(feat)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    @torch.no_grad()
    def select_action(self, obs: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """Epsilon-greedy action selection. Returns (B, 1) long tensor."""
        if torch.rand(1).item() < epsilon:
            return torch.randint(
                0, self.num_actions, (obs.shape[0], 1), device=obs.device
            )
        return self.forward(obs).argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Max Q-value as state value estimate. Returns (B, 1)."""
        return self.forward(obs).max(dim=-1, keepdim=True)[0]
