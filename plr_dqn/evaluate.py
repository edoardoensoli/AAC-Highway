#!/usr/bin/env python3
"""
evaluate.py — Evaluate a trained SB3 DQN + PLR checkpoint on held-out seeds.

Loads models saved in the standard SB3 .zip format via
``stable_baselines3.DQN.load()``.

Usage (run from project root)
-----------------------------
    python plr_dqn/evaluate.py --checkpoint plr/runs/<run>/model_final
    python plr_dqn/evaluate.py --checkpoint plr/runs/<run>/model_final --render
    python plr_dqn/evaluate.py --checkpoint plr/runs/<run>/model_final --seeds 1000 1005 1010
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import highway_env  # noqa: F401

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import DQN as SB3_DQN

from plr_dqn.highway_levels import (
    HighwayVecEnv,
    OBS_SHAPE,
    NUM_ACTIONS,
    EVAL_SEEDS,
    make_flat_env,
)
from plr_dqn.plr_configs import seed_to_factors, seed_to_config, config_for_gym


def load_model(checkpoint_path: str, device: str = "cpu") -> SB3_DQN:
    """
    Load an SB3 DQN model from a checkpoint.

    Parameters
    ----------
    checkpoint_path : str
        Path to the SB3 .zip checkpoint (extension is added automatically
        by SB3 if not present).
    device : str
        Device to load the model onto.

    Returns
    -------
    SB3_DQN
        The loaded SB3 DQN model ready for inference.
    """
    dummy_env = make_flat_env(seed=0)
    model = SB3_DQN.load(checkpoint_path, env=dummy_env, device=device)
    dummy_env.close()
    print(f"Loaded SB3 DQN checkpoint: {checkpoint_path}")
    return model


def evaluate_seeds(
    model: SB3_DQN,
    seeds: list[int],
    num_episodes: int = 2,
    device: str = "cpu",
    render: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Evaluate the model on a list of seeds.

    Returns a dict with per-seed stats and aggregate.
    """
    all_returns = []
    all_lengths = []
    all_crashes = []
    action_counts = np.zeros(NUM_ACTIONS)
    results = {}

    for seed in seeds:
        factors = seed_to_factors(seed)
        
        if render:
            # Create environment with simulation_frequency=30 for smooth rendering
            cfg = config_for_gym(seed_to_config(seed))
            cfg["simulation_frequency"] = 30  # 30 Hz for smooth visualization
            env_gym = gym.make("highway-fast-v0", config=cfg, render_mode="human")
            # Wrap with FlattenObservation for SB3 compatibility
            env_gym = gym.wrappers.FlattenObservation(env_gym)
        else:
            env = HighwayVecEnv(seed=seed, device=device, render=False)
        
        seed_returns = []
        seed_lengths = []
        seed_crashes = []

        for ep in range(num_episodes):
            if render:
                obs, _ = env_gym.reset()
                obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
            else:
                obs = env.reset()
            ep_return = 0.0
            ep_len = 0
            crashed = False

            while True:
                obs_np = obs.cpu().numpy()
                action_np, _ = model.predict(obs_np, deterministic=True)
                action_int = int(np.asarray(action_np).flatten()[0])
                action_counts[action_int] += 1
                
                if render:
                    obs_np_next, reward, terminated, truncated, info = env_gym.step(action_int)
                    obs = torch.from_numpy(obs_np_next).float().to(device).unsqueeze(0)
                    done = terminated or truncated
                    reward_val = reward
                    info["done"] = done
                else:
                    obs, reward, mask, bad_mask, info = env.step(action_int)
                    reward_val = reward.item()
                    done = info["done"]
                
                ep_return += reward_val
                ep_len += 1
                if info.get("crashed", False):
                    crashed = True
                if done:
                    break

            seed_returns.append(ep_return)
            seed_lengths.append(ep_len)
            seed_crashes.append(int(crashed))

        if render:
            env_gym.close()
        else:
            env.close()

        mean_r = np.mean(seed_returns)
        std_r = np.std(seed_returns)
        crash_rate = np.mean(seed_crashes)

        results[seed] = {
            "factors": factors,
            "mean_return": float(mean_r),
            "std_return": float(std_r),
            "mean_length": float(np.mean(seed_lengths)),
            "crash_rate": float(crash_rate),
            "returns": [float(r) for r in seed_returns],
        }

        if verbose:
            print(
                f"  Seed {seed:5d} | "
                f"L={factors['lanes_count']} V={factors['vehicles_count']:2d} "
                f"D={factors['vehicles_density']:.2f} P={factors['politeness']:.3f} | "
                f"R̄={mean_r:7.2f} ± {std_r:5.2f} | "
                f"crash={crash_rate:.0%}"
            )

        all_returns.extend(seed_returns)
        all_lengths.extend(seed_lengths)
        all_crashes.extend(seed_crashes)

    total_acts = max(action_counts.sum(), 1)
    pct = action_counts / total_acts * 100
    actions = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]
    act_str = "  ".join(f"{a}={p:.0f}%" for a, p in zip(actions, pct))

    results["aggregate"] = {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "mean_length": float(np.mean(all_lengths)),
        "crash_rate": float(np.mean(all_crashes)),
        "num_seeds": len(seeds),
        "num_episodes": len(all_returns),
        "action_dist": act_str,
    }

    return results


def evaluate_generalisation(
    model: SB3_DQN,
    num_seeds: int = 100,
    num_episodes: int = 3,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    Evaluate on randomly generated unseen seeds to test generalisation.
    """
    seeds = list(range(5000, 5000 + num_seeds))
    if verbose:
        print(f"\nGeneralisation test: {num_seeds} unseen seeds")
        print("-" * 72)

    results = evaluate_seeds(
        model, seeds, num_episodes, device,
        verbose=False
    )

    if verbose:
        agg = results["aggregate"]
        print(f"  Aggregate: R̄={agg['mean_return']:.2f} ± {agg['std_return']:.2f} "
              f"| crash={agg['crash_rate']:.0%}")
        print(f"  Actions: {agg['action_dist']}")

        # Breakdown by lane count
        for lanes in [2, 3, 4, 5]:
            lane_returns = []
            lane_crashes = []
            for seed in seeds:
                if results[seed]["factors"]["lanes_count"] == lanes:
                    lane_returns.extend(results[seed]["returns"])
                    lane_crashes.append(results[seed]["crash_rate"])
            if lane_returns:
                print(f"  {lanes}-lane: R̄={np.mean(lane_returns):.2f} "
                      f"crash={np.mean(lane_crashes):.0%} "
                      f"({len(lane_crashes)} seeds)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SB3 DQN + PLR checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SB3 .zip checkpoint")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds to evaluate (default: EVAL_SEEDS)")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--generalisation", action="store_true",
                        help="Run generalisation test on 100 unseen seeds")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to JSON")

    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    seeds = args.seeds or EVAL_SEEDS

    print(f"\nEvaluating on {len(seeds)} seeds × {args.num_episodes} episodes")
    print("=" * 72)

    results = evaluate_seeds(
        model, seeds, args.num_episodes, args.device,
        render=args.render, verbose=True,
    )

    agg = results["aggregate"]
    print("-" * 72)
    print(f"Aggregate: R̄={agg['mean_return']:.2f} ± {agg['std_return']:.2f} "
          f"| crash={agg['crash_rate']:.0%}")
    print(f"Actions: {agg['action_dist']}")

    if args.generalisation:
        gen_results = evaluate_generalisation(
            model, num_seeds=100, num_episodes=3, device=args.device
        )
        results["generalisation"] = gen_results

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
