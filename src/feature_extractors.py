"""
Custom Feature Extractors for Highway-env PLR
================================================

Provides architecture upgrades from the default MlpPolicy flattening:

1. **OccupancyGridCNN** — Spatial CNN for OccupancyGrid observations (4, 18, 6).
   Learns spatial filters (e.g. "obstacle ahead") that generalise across PLR
   level configurations regardless of vehicle list ordering.

2. **PermutationInvariantExtractor** — PointNet-style encoder for kinematics
   observations (list of vehicles).  Guarantees that the order of vehicles in
   the state vector does not affect the policy output — essential for PLR where
   the agent must recognise equivalent states during replay.

3. **AttentionExtractor** — Self-attention (Transformer) encoder for vehicle
   observations.  More expressive than PointNet for modelling pairwise
   vehicle–vehicle interactions (e.g. two cars forming a gap).

Usage with Stable-Baselines3
----------------------------
>>> from feature_extractors import OccupancyGridCNN
>>> model = PPO(
...     "CnnPolicy",          # or "MlpPolicy" — both work with custom extractor
...     env,
...     policy_kwargs=dict(
...         features_extractor_class=OccupancyGridCNN,
...         features_extractor_kwargs=dict(features_dim=256),
...         net_arch=dict(pi=[256, 256], vf=[256, 256]),
...     ),
... )
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# ──────────────────────────────────────────────────────────────────────────────
# 1.  OccupancyGridCNN  —  spatial CNN for (C, H, W) grid observations
# ──────────────────────────────────────────────────────────────────────────────
class OccupancyGridCNN(BaseFeaturesExtractor):
    """
    CNN feature extractor for OccupancyGrid observations.

    Default input shape: ``(batch, 4, 18, 6)`` where

    * ``C = 4``  — feature channels (presence, vx, vy, on_road)
    * ``H = 18`` — longitudinal cells (90 m / 5 m per cell)
    * ``W = 6``  — lateral cells (30 m / 5 m per cell)

    Uses small 3×3 kernels appropriate for 5 m cell resolution.
    One stride-2 layer reduces spatial dims while keeping the receptive
    field large enough to cover merging / braking situations.

    As shown in the CarRacing PLR experiments (Jiang et al., 2021),
    a CNN-based image embedding is critical for generalising across
    OOD conditions — the same principle applies to the OccupancyGrid's
    2-D topology.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Must have shape ``(C, H, W)`` with 3 dimensions.
    features_dim : int
        Dimensionality of the output feature vector (default 256).
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        n_channels = observation_space.shape[0]  # e.g. 4

        self.cnn = nn.Sequential(
            # (C, H, W) → (32, H, W)
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # (32, H, W) → (64, ⌈H/2⌉, ⌈W/2⌉)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            # (64, ⌈H/2⌉, ⌈W/2⌉) → (64, ⌈H/2⌉, ⌈W/2⌉)
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically compute the flattened size (robust to grid changes)
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            flat_size = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flat_size, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  PermutationInvariantExtractor  —  PointNet-style for kinematics obs
# ──────────────────────────────────────────────────────────────────────────────
class PermutationInvariantExtractor(BaseFeaturesExtractor):
    """
    PointNet-style permutation-invariant feature extractor.

    Input shape: ``(batch, N_vehicles, F_per_vehicle)``
    e.g. ``(batch, 5, 5)`` for 5 nearest vehicles with 5 features each.

    Architecture
    ~~~~~~~~~~~~
    1. Per-vehicle MLP φ : ℝ^F → ℝ^D
    2. Max-pool across vehicles : ℝ^{N×D} → ℝ^D
    3. Global MLP ψ : ℝ^D → ℝ^{features_dim}

    The output is **invariant** to the ordering of vehicles in the
    observation — essential for PLR because the agent must recognise
    the "state" of a level during replay regardless of the (arbitrary)
    vehicle list order returned by highway-env.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Shape ``(N, F)`` (2-D) or ``(N*F,)`` (1-D, auto-reshaped).
    features_dim : int
        Output feature vector size.
    hidden_dim : int
        Width of the per-vehicle and global MLPs.
    n_vehicles : int
        Number of vehicles when obs is flattened (1-D).  Ignored if the
        observation space is already 2-D.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        hidden_dim: int = 128,
        n_vehicles: int = 5,
    ):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) == 2:
            self._n_vehicles, self._n_features = obs_shape
        else:
            # Flattened kinematics: use supplied n_vehicles
            self._n_features = obs_shape[0] // n_vehicles
            self._n_vehicles = n_vehicles

        # Per-vehicle encoder φ
        self.phi = nn.Sequential(
            nn.Linear(self._n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Global encoder ψ  (after max-pool)
        self.psi = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        if observations.dim() == 2:
            # Reshape from (batch, N*F) → (batch, N, F)
            observations = observations.view(
                batch_size, self._n_vehicles, self._n_features
            )

        # Per-vehicle encoding: (batch, N, F) → (batch, N, D)
        encoded = self.phi(observations)

        # Permutation-invariant pooling: (batch, N, D) → (batch, D)
        pooled, _ = encoded.max(dim=1)

        # Global encoding: (batch, D) → (batch, features_dim)
        return self.psi(pooled)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  AttentionExtractor  —  Transformer encoder for vehicle observations
# ──────────────────────────────────────────────────────────────────────────────
class AttentionExtractor(BaseFeaturesExtractor):
    """
    Self-attention (Transformer) feature extractor for vehicle observations.

    More expressive than PointNet: multi-head self-attention captures
    *pairwise interactions* between vehicles (e.g. two cars forming a gap
    the ego can squeeze through), while mean-pooling across the vehicle
    dimension preserves permutation invariance.

    Parameters
    ----------
    observation_space : gym.spaces.Box
        Shape ``(N, F)`` (2-D) or ``(N*F,)`` (1-D, auto-reshaped).
    features_dim : int
        Output feature vector size.
    d_model : int
        Internal embedding dimension per vehicle.
    n_heads : int
        Number of attention heads (must divide ``d_model``).
    n_layers : int
        Number of Transformer encoder layers.
    n_vehicles : int
        Number of vehicles when obs is flattened.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        n_vehicles: int = 5,
    ):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space.shape
        if len(obs_shape) == 2:
            self._n_vehicles, self._n_features = obs_shape
        else:
            self._n_features = obs_shape[0] // n_vehicles
            self._n_vehicles = n_vehicles

        # Project vehicle features → d_model
        self.input_proj = nn.Linear(self._n_features, d_model)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        if observations.dim() == 2:
            observations = observations.view(
                batch_size, self._n_vehicles, self._n_features
            )

        # Project: (batch, N, F) → (batch, N, d_model)
        x = self.input_proj(observations)

        # Self-attention: (batch, N, d_model) → (batch, N, d_model)
        x = self.transformer(x)

        # Mean pooling (permutation-invariant): (batch, N, d_model) → (batch, d_model)
        x = x.mean(dim=1)

        # Output: (batch, d_model) → (batch, features_dim)
        return self.output_proj(x)
