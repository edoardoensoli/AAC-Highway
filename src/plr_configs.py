"""
PLR Environment Configurations — Seed-Based Parametric Levels
=============================================================

Maps integer seeds deterministically to highway-env factors of variation:

  • lanes_count       ∈ {2, 3, 4, 5}
  • vehicles_count    ∈ [20, 40]
  • vehicles_density  ∈ [1.0, 1.5]   (capped to prevent unsolvable levels)
  • POLITENESS        ∈ [0.0, 1.0]   (MOBIL lane-change model coefficient)

Observation space:
  • OccupancyGrid (W×H×F) — ego-centric spatial grid encoding presence,
    velocity, and road structure around the agent.

The MOBIL model in highway-env controls lane-change decisions.
POLITENESS ∈ [0, 1] weights consideration for neighbouring vehicles'
acceleration when deciding whether to change lane:
  POLITENESS = 0.0  →  purely egoistic (aggressive)
  POLITENESS = 1.0  →  maximally considerate (polite)

Training levels (Λ_train) and test levels (Λ_test) use **disjoint** seed
ranges so that held-out evaluation measures true zero-shot generalisation
rather than overfitting to training levels.

highway-env does NOT support setting POLITENESS through the config dict;
HighwayLevelWrapper applies it by setting **per-instance** attributes only
(no class-level mutation) on every ``reset()`` and ``step()`` to handle
mid-episode vehicle spawns without leaking state between wrapper instances.
"""

from __future__ import annotations

from typing import Dict, List

import gymnasium as gym
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Factor ranges
# ─────────────────────────────────────────────────────────────────────────────
LANES_OPTIONS = [2, 3, 4, 5]
VEHICLES_RANGE = (20, 40)
DENSITY_RANGE = (1.0, 1.5)       # capped at 1.5 to prevent unsolvable levels
POLITENESS_RANGE = (0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Shared observation / action / reward  (same across all levels)
# ─────────────────────────────────────────────────────────────────────────────
_OBSERVATION_CFG = {
    "type": "OccupancyGrid",
    "features": ["presence", "vx", "vy", "on_road"],
    "features_range": {
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "grid_size": [[-30, 60], [-15, 15]],   # 90 m × 30 m, asymmetric forward
    "grid_step": [5, 5],                    # 5 m per cell  →  (4, 18, 6)
    "absolute": False,
    "align_to_vehicle_axes": True,
    "clip": True,
    "as_image": False,
}
# Obs shape: (F, W, H) = (4, 18, 6) = 432 features after flatten


_ACTION_CFG = {"type": "DiscreteMetaAction"}

_REWARD_CFG = {
    "collision_reward": -20,         # harsh crash penalty (survives normalize_reward)
    "right_lane_reward": 0.1,
    "high_speed_reward": 0.4,
    "lane_change_reward": -0.1,
    "reward_speed_range": [20, 30],
    "normalize_reward": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# Seed → Factor mapping
# ─────────────────────────────────────────────────────────────────────────────
def seed_to_factors(seed: int) -> Dict:
    """
    Deterministically map an integer seed to highway env factors.

    Uses a seeded PRNG so the same seed always produces the same
    (lanes_count, vehicles_count, vehicles_density, politeness) tuple.
    """
    rng = np.random.RandomState(seed)
    return {
        "lanes_count": int(rng.choice(LANES_OPTIONS)),
        "vehicles_count": int(rng.randint(VEHICLES_RANGE[0], VEHICLES_RANGE[1] + 1)),
        "vehicles_density": round(float(rng.uniform(*DENSITY_RANGE)), 2),
        "politeness": round(float(rng.uniform(*POLITENESS_RANGE)), 3),
    }


def seed_to_config(seed: int) -> Dict:
    """
    Build a complete highway-v0 config dict from a seed.

    Includes all parameters needed for ``gym.make()`` plus metadata keys
    (_seed, _politeness) consumed by :class:`HighwayLevelWrapper`.
    """
    factors = seed_to_factors(seed)
    return {
        "env_id": "highway-v0",
        "observation": _OBSERVATION_CFG,
        "action": _ACTION_CFG,
        "lanes_count": factors["lanes_count"],
        "vehicles_count": factors["vehicles_count"],
        "vehicles_density": factors["vehicles_density"],
        "duration": 40,
        "policy_frequency": 5,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        # Metadata — consumed by wrapper, stripped before gym.make
        "_politeness": factors["politeness"],
        "_seed": seed,
        **_REWARD_CFG,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Disjoint seed sets:  Λ_train  and  Λ_test
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SEEDS = list(range(0, 50))        # |Λ_train| = 50
TEST_SEEDS  = list(range(1000, 1020))   # |Λ_test|  = 20, fully disjoint


# ─────────────────────────────────────────────────────────────────────────────
# Gymnasium Wrapper — applies POLITENESS to NPC vehicles
# ─────────────────────────────────────────────────────────────────────────────
class HighwayLevelWrapper(gym.Wrapper):
    """
    Wrap *highway-v0* to apply:

    1. **POLITENESS** — seed-based MOBIL coefficient for NPC lane-change
       aggressiveness, set at both class and instance level on ``reset()``.

    The wrapper injects ``level_seed`` and ``politeness`` into the ``info`` dict.
    """

    def __init__(
        self,
        env: gym.Env,
        seed: int = 0,
    ):
        super().__init__(env)
        self._level_seed = seed
        self._factors = seed_to_factors(seed)

    # ── Gymnasium API ────────────────────────────────────────────────────

    def reset(self, **kwargs):
        # NOTE: We do NOT touch IDMVehicle.POLITENESS (the class variable).
        # Mutating the class leaks state between parallel wrapper instances.
        obs, info = super().reset(**kwargs)

        # Set per-instance POLITENESS for every NPC after reset
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

        # Re-apply per-instance POLITENESS for any newly spawned NPCs
        pol = self._factors["politeness"]
        for v in self.unwrapped.road.vehicles:
            if v is not self.unwrapped.vehicle:
                v.POLITENESS = pol

        return obs, reward, done, truncated, info

    def set_level(self, seed: int):
        """Change the level for the next ``reset()``.

        .. important:: You **must** call ``reset()`` before the next
           ``step()`` — no environment steps may be taken between
           ``set_level()`` and ``reset()``.
        """
        self._level_seed = seed
        self._factors = seed_to_factors(seed)
        cfg = self.unwrapped.config
        cfg["lanes_count"] = self._factors["lanes_count"]
        cfg["vehicles_count"] = self._factors["vehicles_count"]
        cfg["vehicles_density"] = self._factors["vehicles_density"]
        self._needs_reset = True  # guard: forbid step() before reset()


# ─────────────────────────────────────────────────────────────────────────────
# Level Factory
# ─────────────────────────────────────────────────────────────────────────────
class HighwayLevelFactory:
    """
    Unified interface for creating training and test environments.

    Environments are created from seeds with all factors of variation
    (including POLITENESS) applied via :class:`HighwayLevelWrapper`.
    """

    def __init__(self, train_seeds=None, test_seeds=None):
        self.train_seeds = train_seeds or TRAIN_SEEDS
        self.test_seeds  = test_seeds or TEST_SEEDS
        # Pre-compute full config dicts
        self.train_configs = [seed_to_config(s) for s in self.train_seeds]
        self.test_configs  = [seed_to_config(s) for s in self.test_seeds]

    @property
    def num_train_levels(self) -> int:
        return len(self.train_seeds)

    @property
    def num_test_levels(self) -> int:
        return len(self.test_seeds)

    # ── env creation ─────────────────────────────────────────────────────────

    def make_env(self, seed: int, render: bool = False) -> gym.Env:
        """Create a single wrapped environment from a seed."""
        config = seed_to_config(seed)
        env_config = _config_for_gym(config)
        render_mode = "human" if render else None
        env = gym.make("highway-v0", config=env_config, render_mode=render_mode)
        return HighwayLevelWrapper(env, seed=seed)

    def make_train_env(self, level_idx: int, render: bool = False) -> gym.Env:
        """Create a wrapped environment for training level ``level_idx``."""
        return self.make_env(self.train_seeds[level_idx], render=render)

    def make_test_env(self, test_idx: int, render: bool = False) -> gym.Env:
        """Create a wrapped environment for test level ``test_idx``."""
        return self.make_env(self.test_seeds[test_idx], render=render)

    # ── descriptions ─────────────────────────────────────────────────────────

    def describe_level(self, level_idx: int) -> str:
        f = seed_to_factors(self.train_seeds[level_idx])
        return (
            f"seed={self.train_seeds[level_idx]}  "
            f"L={f['lanes_count']}  V={f['vehicles_count']}  "
            f"D={f['vehicles_density']:.1f}  P={f['politeness']:.2f}"
        )

    def describe_test_level(self, test_idx: int) -> str:
        f = seed_to_factors(self.test_seeds[test_idx])
        return (
            f"seed={self.test_seeds[test_idx]}  "
            f"L={f['lanes_count']}  V={f['vehicles_count']}  "
            f"D={f['vehicles_density']:.1f}  P={f['politeness']:.2f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────
_META_KEYS = frozenset({"env_id", "_politeness", "_seed"})


def _config_for_gym(cfg: Dict) -> Dict:
    """Strip metadata keys that are not valid gym.make kwargs."""
    return {k: v for k, v in cfg.items() if k not in _META_KEYS}


# ─────────────────────────────────────────────────────────────────────────────
# Backward-compatible API  (used by dqn_plr.py, ppo_baseline.py, etc.)
# ─────────────────────────────────────────────────────────────────────────────
def generate_env_configs() -> List[Dict]:
    """Return training configs.  Backward-compatible alias."""
    return [seed_to_config(s) for s in TRAIN_SEEDS]


def generate_test_configs() -> List[Dict]:
    """Return held-out test configs.  Backward-compatible alias."""
    return [seed_to_config(s) for s in TEST_SEEDS]


def config_without_env_id(cfg: Dict) -> Dict:
    """Return config with internal/meta keys removed (for gym.make)."""
    return {k: v for k, v in cfg.items() if k not in _META_KEYS}


def get_env_id(cfg: Dict) -> str:
    """Extract env_id, defaulting to highway-v0."""
    return cfg.get("env_id", "highway-v0")


def describe_config(cfg: Dict) -> str:
    """One-line human-readable summary of a config dict."""
    lanes = cfg.get("lanes_count", "?")
    density = cfg.get("vehicles_density", "?")
    veh = cfg.get("vehicles_count", "?")
    pol = cfg.get("_politeness", "?")
    seed = cfg.get("_seed", "?")
    return f"highway  L={lanes}  V={veh}  D={density}  P={pol}  (seed={seed})"


# ─────────────────────────────────────────────────────────────────────────────
# Quick-print when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    factory = HighwayLevelFactory()

    print(f"Training levels (Λ_train): {factory.num_train_levels}")
    print(f"Test levels     (Λ_test) : {factory.num_test_levels}")
    print(f"\n{'─' * 64}")
    print("  Λ_train")
    print(f"{'─' * 64}")
    for i in range(factory.num_train_levels):
        print(f"  [{i:2d}] {factory.describe_level(i)}")

    print(f"\n{'─' * 64}")
    print("  Λ_test  (held-out, never seen by PLR)")
    print(f"{'─' * 64}")
    for i in range(factory.num_test_levels):
        print(f"  [{i:2d}] {factory.describe_test_level(i)}")
