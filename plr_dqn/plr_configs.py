"""
PLR Environment Configurations — Seed-Based Parametric Levels
=============================================================

Self-contained config module for the plr/ training pipeline.
Maps integer seeds deterministically to highway-env factors of variation
with **lane-dependent correlated ranges** to prevent trivially easy or
impossibly hard levels.

Factor ranges (correlated by lanes):
  • lanes_count       ∈ {2, 3, 4, 5}
  • vehicles_count    ∈ per-lane range (overall 10–50)
  • vehicles_density  ∈ per-lane range (overall 0.8–2.0)
  • POLITENESS        ∈ [0.0, 1.0]

Lane-dependent ranges:
  lanes=2 → vehicles ∈ [10, 20], density ∈ [0.8, 1.2]
  lanes=3 → vehicles ∈ [15, 30], density ∈ [0.8, 1.5]
  lanes=4 → vehicles ∈ [20, 40], density ∈ [1.0, 1.8]
  lanes=5 → vehicles ∈ [25, 50], density ∈ [1.0, 2.0]

This ensures 2-lane levels never get packed impossibly dense, and
5-lane levels are never trivially empty.
"""

from __future__ import annotations

from typing import Dict

import gymnasium as gym
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Factor ranges — correlated by lane count
# ─────────────────────────────────────────────────────────────────────────────
LANES_OPTIONS = [2, 3, 4, 5]
POLITENESS_RANGE = (0.0, 1.0)

# Per-lane ranges for vehicles_count and vehicles_density.
# Keys = lanes_count, values = (vehicles_min, vehicles_max), (density_min, density_max)
LANE_DEPENDENT_RANGES = {
    2: {"vehicles": (10, 20), "density": (0.8, 1.2)},
    3: {"vehicles": (15, 30), "density": (0.8, 1.5)},
    4: {"vehicles": (20, 40), "density": (1.0, 1.8)},
    5: {"vehicles": (25, 50), "density": (1.0, 2.0)},
}


# Shared observation / action / reward (same across all levels)
DURATION = 60
POLICY_FREQUENCY = 2

OBSERVATION_CFG = {
    "type": "Kinematics",
    "vehicles_count": 7,
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,
    "order": "sorted",
    "normalize": True,
    "clip": True,
    "see_behind": True,
    "observe_intentions": False,
}

ACTION_CFG = {"type": "DiscreteMetaAction"}

REWARD_CFG = {
    "collision_reward": -10,
    "right_lane_reward": 0,
    "high_speed_reward": 0.3,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
}



# Seed → Factor mapping  (lane-dependent correlated sampling)
def seed_to_factors(seed: int) -> Dict:
    """
    Deterministically map an integer seed to highway env factors.

    Vehicle count and density are sampled from lane-dependent ranges
    so that narrow roads (2 lanes) never get impossibly dense traffic,
    and wide roads (5 lanes) always have enough cars to be non-trivial.
    """
    rng = np.random.RandomState(seed)

    lanes = int(rng.choice(LANES_OPTIONS))
    lr = LANE_DEPENDENT_RANGES[lanes]

    vehicles_count = int(rng.randint(lr["vehicles"][0], lr["vehicles"][1] + 1))
    vehicles_density = round(float(rng.uniform(*lr["density"])), 2)
    politeness = round(float(rng.uniform(*POLITENESS_RANGE)), 3)

    return {
        "lanes_count": lanes,
        "vehicles_count": vehicles_count,
        "vehicles_density": vehicles_density,
        "politeness": politeness,
    }


def seed_to_config(seed: int) -> Dict:
    """
    Build a complete highway-v0 config dict from a seed.

    Includes all parameters needed for ``gym.make()`` plus metadata keys
    (_seed, _politeness) consumed by :class:`HighwayLevelWrapper`.
    """
    factors = seed_to_factors(seed)
    return {
        "env_id": "highway-fast-v0",
        "observation": OBSERVATION_CFG,
        "action": ACTION_CFG,
        "lanes_count": factors["lanes_count"],
        "vehicles_count": factors["vehicles_count"],
        "vehicles_density": factors["vehicles_density"],
        "duration": DURATION,
        "policy_frequency": POLICY_FREQUENCY,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "_politeness": factors["politeness"],
        "_seed": seed,
        **REWARD_CFG,
    }


def describe_level(seed: int) -> str:
    """One-line description of a level: seed → factors."""
    f = seed_to_factors(seed)
    return (
        f"seed={seed}  L={f['lanes_count']}  "
        f"V={f['vehicles_count']}  D={f['vehicles_density']:.2f}  "
        f"P={f['politeness']:.3f}"
    )


# Disjoint seed sets for train and test
TRAIN_SEEDS = list(range(0, 50))
TEST_SEEDS  = list(range(1000, 1020))



# Gymnasium Wrapper — applies POLITENESS to NPC vehicles
_META_KEYS = frozenset({"env_id", "_politeness", "_seed"})


def config_for_gym(cfg: Dict) -> Dict:
    """Strip metadata keys that are not valid gym.make kwargs."""
    return {k: v for k, v in cfg.items() if k not in _META_KEYS}


class HighwayLevelWrapper(gym.Wrapper):
    """
    Wrap *highway-v0* to apply POLITENESS to NPC vehicles.

    The MOBIL lane-change model coefficient is set per-instance on every
    ``reset()`` and ``step()`` to handle mid-episode vehicle spawns
    without leaking state between wrapper instances.
    """

    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self._level_seed = seed
        self._factors = seed_to_factors(seed)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        pol = self._factors["politeness"]
        for v in self.unwrapped.road.vehicles:
            if v is not self.unwrapped.vehicle:
                v.POLITENESS = pol
        info["level_seed"] = self._level_seed
        info["politeness"] = pol
        self._needs_reset = False
        return obs, info

    def step(self, action):
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "HighwayLevelWrapper.step() called after set_level() "
                "without an intervening reset().  Call env.reset() first."
            )
        obs, reward, done, truncated, info = super().step(action)
        pol = self._factors["politeness"]
        for v in self.unwrapped.road.vehicles:
            if v is not self.unwrapped.vehicle:
                v.POLITENESS = pol
        return obs, reward, done, truncated, info

    def set_level(self, seed: int):
        """Change the level for the next ``reset()``."""
        self._level_seed = seed
        self._factors = seed_to_factors(seed)
        cfg = self.unwrapped.config
        cfg["lanes_count"] = self._factors["lanes_count"]
        cfg["vehicles_count"] = self._factors["vehicles_count"]
        cfg["vehicles_density"] = self._factors["vehicles_density"]
        # Ensure observation / action configs are never lost
        cfg["observation"] = OBSERVATION_CFG
        cfg["action"] = ACTION_CFG
        cfg["duration"] = DURATION
        cfg["policy_frequency"] = POLICY_FREQUENCY
        # Also update reward config in case env was created with different values
        for k, v in REWARD_CFG.items():
            cfg[k] = v
        self._needs_reset = True


# Quick-print when run directly
if __name__ == "__main__":
    print("Lane-dependent factor ranges:")
    for lanes, r in sorted(LANE_DEPENDENT_RANGES.items()):
        print(f"  {lanes} lanes → V∈{r['vehicles']}, D∈{r['density']}")

    print(f"\nReward config: {REWARD_CFG}")

    print(f"\nSample levels:")
    for seed in [0, 42, 100, 999, 1000, 5000]:
        print(f"  {describe_level(seed)}")
