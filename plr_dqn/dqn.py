"""
DQN — Double Deep Q-Network with experience replay.

Implements:
  - Circular replay buffer with uniform sampling
  - Double DQN (van Hasselt et al., 2016): online net selects action,
    target net evaluates it
  - Huber (smooth L1) loss
  - Hard or soft target network updates
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    """
    Fixed-size circular replay buffer stored as contiguous tensors.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_shape : tuple
        Observation shape (flattened).
    device : str
        Tensor device.
    """

    def __init__(self, capacity: int, obs_shape: tuple, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, *obs_shape, device=device)
        self.actions = torch.zeros(capacity, 1, dtype=torch.long, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, *obs_shape, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        """Add a single transition (tensors, no batch dim)."""
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.dones[self.pos] = done
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a random mini-batch."""
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size


class DQN:
    """
    Double DQN agent.

    Parameters
    ----------
    q_net : nn.Module
        Online Q-network.
    target_net : nn.Module
        Target Q-network (periodically synced with online).
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    max_grad_norm : float
        Gradient clipping threshold.
    """

    def __init__(
        self,
        q_net: nn.Module,
        target_net: nn.Module,
        lr: float = 5e-4,
        gamma: float = 0.99,
        max_grad_norm: float = 10.0,
    ):
        self.q_net = q_net
        self.target_net = target_net
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.optimizer = optim.Adam(q_net.parameters(), lr=lr)

        # Initial sync
        self.hard_update_target()

    def update(self, replay_buffer: ReplayBuffer, batch_size: int) -> dict:
        """
        One gradient step of Double DQN.

        Returns dict with loss and mean Q-value.
        """
        obs, actions, rewards, next_obs, dones = replay_buffer.sample(batch_size)

        # Current Q(s, a)
        current_q = self.q_net(obs).gather(1, actions)

        # Double DQN target: online net SELECTS action, target net EVALUATES
        with torch.no_grad():
            next_actions = self.q_net(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_obs).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
        }

    def hard_update_target(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def soft_update_target(self, tau: float = 0.005):
        """Polyak-average the target network."""
        for p, tp in zip(self.q_net.parameters(), self.target_net.parameters()):
            tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)
