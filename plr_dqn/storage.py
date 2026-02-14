"""
Rollout Storage — experience buffer with per-step level tracking.

Faithful adaptation of ``facebookresearch/dcd/algos/storage.py``.
Tracks: obs, actions, action_log_probs, action_log_dist (full logits),
value_preds, rewards, masks, bad_masks, level_seeds.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T: int, N: int, tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(T * N, *tensor.shape[2:])


class RolloutStorage:
    """
    Fixed-size rollout buffer for on-policy algorithms.

    Shapes: ``(num_steps+1, num_processes, ...)`` for obs/values/masks
            ``(num_steps, num_processes, ...)`` for actions/rewards/logprobs

    Parameters
    ----------
    num_steps : int
        Rollout horizon (number of environment steps per update).
    num_processes : int
        Number of parallel environments.
    obs_shape : tuple
        Observation shape (e.g. (4, 18, 6) for OccupancyGrid).
    action_space : gym.spaces.Space
        The environment's action space.
    """

    def __init__(
        self,
        num_steps: int,
        num_processes: int,
        obs_shape: tuple,
        action_space,
        device: str = "cpu",
    ):
        self.device = device
        self.num_steps = num_steps
        self.num_processes = num_processes

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)

        # Full logit vector for PLR scoring (policy entropy, min_margin, etc.)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
            num_actions = action_space.n
            self.action_log_dist = torch.zeros(
                num_steps, num_processes, num_actions
            )
        else:
            action_shape = action_space.shape[0]
            num_actions = action_shape
            self.action_log_dist = torch.zeros(num_steps, num_processes, 1)

        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()

        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.bad_masks = torch.ones(num_steps + 1, num_processes, 1)

        self.level_seeds = torch.zeros(
            num_steps, num_processes, 1, dtype=torch.int
        )

        self.step = 0

    def to(self, device: str):
        self.device = device
        self.obs = self.obs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.action_log_dist = self.action_log_dist.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        self.bad_masks = self.bad_masks.to(device)
        self.level_seeds = self.level_seeds.to(device)
        return self

    def insert(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        action_log_dist: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        bad_masks: torch.Tensor,
        level_seeds: torch.Tensor | None = None,
    ):
        """Insert one step of experience for all processes."""
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.action_log_dist[self.step].copy_(action_log_dist)
        self.value_preds[self.step].copy_(value_preds)
        if len(rewards.shape) == 3:
            rewards = rewards.squeeze(2)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.bad_masks[self.step + 1].copy_(bad_masks)

        if level_seeds is not None:
            self.level_seeds[self.step].copy_(level_seeds)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        """Copy last obs/mask to position 0 for the next rollout."""
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(
        self,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ):
        """Compute GAE returns in-place."""
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages, num_mini_batch):
        """Yield mini-batches for PPO updates."""
        num_steps, num_processes = self.rewards.size()[:2]
        batch_size = num_processes * num_steps

        assert batch_size >= num_mini_batch, (
            f"PPO requires num_processes ({num_processes}) * num_steps ({num_steps}) "
            f"= {batch_size} >= num_mini_batch ({num_mini_batch})"
        )
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)),
            mini_batch_size,
            drop_last=True,
        )

        for indices in sampler:
            obs_batch = self.obs[:-1].reshape(-1, *self.obs.shape[2:])[indices]
            actions_batch = self.actions.reshape(-1, self.actions.shape[-1])[indices]
            value_preds_batch = self.value_preds[:-1].reshape(-1, 1)[indices]
            return_batch = self.returns[:-1].reshape(-1, 1)[indices]
            masks_batch = self.masks[:-1].reshape(-1, 1)[indices]
            old_action_log_probs_batch = self.action_log_probs.reshape(-1, 1)[indices]

            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages.reshape(-1, 1)[indices]

            yield (
                obs_batch,
                actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
            )
