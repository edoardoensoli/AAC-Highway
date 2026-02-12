"""
PLR Environment Configurations — Highway-v0, 15 Difficulty Levels
===================================================================

Hand-crafted curriculum of 15 highway configs with increasing difficulty.
Difficulty axes:
  • lanes_count      — fewer lanes = harder to manoeuvre
  • vehicles_density — higher = more congestion
  • vehicles_count   — more NPCs on the road
  • NPC behaviour    — IDMVehicle (polite) → AggressiveVehicle (reckless)

Levels 1-5 :  easy   — wide roads, sparse traffic, polite drivers
Levels 6-10:  medium — moderate congestion, aggressive drivers
Levels 11-15: hard   — narrow roads, dense traffic, aggressive drivers
"""

from typing import Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Shared observation / action
# ─────────────────────────────────────────────────────────────────────────────
_OBSERVATION_CFG = {
    "type": "Kinematics",
    "vehicles_count": 7,       # observe more nearby vehicles
    "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
    "absolute": False,
    "normalize": True,
    "order": "sorted",         # sort by distance for consistency
}

_ACTION_CFG = {
    "type": "DiscreteMetaAction",
}

# NPC vehicle types
_IDM       = "highway_env.vehicle.behavior.IDMVehicle"
_AGGRESSIVE = "highway_env.vehicle.behavior.AggressiveVehicle"


def _cfg(lanes, density, vehicles, npc_type) -> Dict:
    """Build one highway-v0 config."""
    return {
        "env_id": "highway-v0",
        "observation": _OBSERVATION_CFG,
        "action": _ACTION_CFG,
        "lanes_count": lanes,
        "vehicles_density": density,
        "vehicles_count": vehicles,
        "collision_reward": -5,          # strong crash penalty
        "right_lane_reward": 0,         # default
        "high_speed_reward": 0.4,         # default
        "lane_change_reward": -0.05,          
        "reward_speed_range": [20, 30],   # default
        "normalize_reward": True,         # CRITICAL: stabilizes learning across levels
        "duration": 40,                   # shorter episodes = faster iteration
        "policy_frequency": 5,
        "other_vehicles_type": npc_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7 training configs  — FOCUSED curriculum (reduced from 15 to avoid forgetting)
# ─────────────────────────────────────────────────────────────────────────────
# Strategy: Vary ONE dimension at a time for better learning
#                        lanes  density  vehicles  npc_type
_TRAINING_CONFIGS = [
    # ── EASY: Learn basic driving ──────────────────────
    _cfg(3,    1.0,      15,    _IDM),  
    _cfg(2,    1.0,      15,    _IDM), 
    
    # ── Dimension 1: Traffic density (same layout) ────
    _cfg(3,    1.3,      25,    _IDM),          # 2  denser traffic, polite
    _cfg(4,    1.5,      35,    _IDM),          # 3  dense traffic, polite
    
    # ── Dimension 2: Add aggressive drivers ────────────
    _cfg(2,    1.3,      20,    _AGGRESSIVE),   # 4  moderate + aggressive
    _cfg(4,    1.4,      30,    _AGGRESSIVE),   # 5  dense + aggressive
    
    # ── Dimension 3: Narrow lanes (hardest) ────────────
    _cfg(3,    1.5,      25,    _AGGRESSIVE),
    _cfg(4,    1.6,      35,    _AGGRESSIVE),   # 6  narrower, busy, aggressive
    _cfg(2,    1.2,      25,    _AGGRESSIVE),   # 7  narrow, dense, aggressive
]


def generate_env_configs() -> List[Dict]:
    """Return the 15 hand-crafted training configs."""
    return list(_TRAINING_CONFIGS)


# ─────────────────────────────────────────────────────────────────────────────
# Test configs (unseen combos for generalisation evaluation)
# ─────────────────────────────────────────────────────────────────────────────
def generate_test_configs() -> List[Dict]:
    """
    Highway test configs with unseen parameter combinations.
    simulation_frequency=30 for smoother rendering.
    """
    test = []
    for lanes, density, veh, npc in [
        (4, 0.6,  12, _IDM),          # easy OOD
        (3, 1.3,  22, _IDM),          # medium, polite (not in training)
        (3, 1.3,  22, _AGGRESSIVE),   # medium, aggressive OOD
        (2, 1.8,  28, _AGGRESSIVE),   # hard OOD
        (5, 2.5,  40, _AGGRESSIVE),   # extreme OOD (5 lanes!)
        (2, 0.5,  10, _IDM),          # narrow but sparse (not in training)
    ]:
        c = _cfg(lanes, density, veh, npc)
        c["simulation_frequency"] = 30
        test.append(c)
    return test


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def config_without_env_id(cfg: Dict) -> Dict:
    """Return a copy of cfg with 'env_id' removed (for gym.make)."""
    return {k: v for k, v in cfg.items() if k != "env_id"}


def get_env_id(cfg: Dict) -> str:
    """Extract env_id, defaulting to highway-v0."""
    return cfg.get("env_id", "highway-v0")


def describe_config(cfg: Dict) -> str:
    """One-line human-readable summary."""
    lanes = cfg.get("lanes_count", "?")
    density = cfg.get("vehicles_density", "?")
    veh = cfg.get("vehicles_count", "?")
    npc = cfg.get("other_vehicles_type", "")
    npc_tag = "AGG" if "Aggressive" in npc else "IDM"
    return f"highway  L={lanes}  D={density}  V={veh}  NPC={npc_tag}"


# ─────────────────────────────────────────────────────────────────────────────
# Quick-print when run directly
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    configs = generate_env_configs()
    print(f"Training configs: {len(configs)}\n")
    for i, cfg in enumerate(configs):
        print(f"  [{i+1:2d}] {describe_config(cfg)}")

    print(f"\n{'─' * 60}")
    test = generate_test_configs()
    print(f"\nTest configs: {len(test)}\n")
    for i, cfg in enumerate(test):
        print(f"  [{i+1:2d}] {describe_config(cfg)}")
