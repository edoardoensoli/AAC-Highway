"""
DQN + PLR (Prioritized Level Replay) — Fixed Implementation
=============================================================

Changes from original:
1. Uses modular PLR components from plr.py
2. Fixes replay buffer corruption: clears buffer on env switch
3. Fixes observation/reward config (via updated plr_configs.py)
4. Fixed hyperparameters for short-horizon highway episodes
5. Proper AggressiveVehicle registration
6. Score computed from evaluation rollout with value predictions
7. Optimised for MacBook Air M1

NOTE: PPO is the recommended algorithm for PLR (see ppo_plr.py).
      DQN + env switching is inherently unstable because the replay buffer
      mixes transitions from different MDPs. This version mitigates it by
      clearing the buffer on env switch, but PPO avoids the problem entirely.
"""

import gymnasium as gym
import highway_env
import torch
import numpy as np
import json
import time

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from highway_env.vehicle.behavior import IDMVehicle

from plr import LevelSampler, LevelRollout, RollingStats
from plr_configs import (
    generate_env_configs,
    generate_test_configs,
    config_without_env_id,
    get_env_id,
    describe_config,
)


# ──────────────────────────────────────────────────────────────────────────────
# Device detection
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────
NUM_ENVS = 4  # parallel envs for throughput


def _make_env(config: dict, rank: int = 0):
    env_id = get_env_id(config)
    env_config = config_without_env_id(config)

    def _init():
        return gym.make(env_id, config=env_config)

    return _init


def make_vec_env(config: dict, n_envs: int = NUM_ENVS):
    if n_envs == 1:
        return DummyVecEnv([_make_env(config, 0)])
    return SubprocVecEnv([_make_env(config, i) for i in range(n_envs)])


def make_single_env(config: dict, render: bool = False):
    env_id = get_env_id(config)
    env_config = config_without_env_id(config)
    render_mode = "human" if render else None
    return gym.make(env_id, config=env_config, render_mode=render_mode)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Evaluation for PLR scoring
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_level_dqn(
    model: DQN,
    config: dict,
    level_idx: int,
    n_episodes: int = 5,
    gamma: float = 0.99,
) -> LevelRollout:
    """
    Evaluate a DQN agent on one level.
    Collects Q-values as 'value predictions' for PLR scoring.
    """
    device = model.device
    rollout = LevelRollout(level_idx=level_idx)
    env = make_single_env(config)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            obs_t = torch.as_tensor(
                obs[np.newaxis].astype(np.float32), device=device
            )
            q_values = model.policy.q_net(obs_t)
            max_q = float(q_values.max(dim=1)[0].cpu())
            q_logits = q_values.cpu().numpy().squeeze()

            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, done, truncated, _ = env.step(action)

            rollout.add_step(
                obs=obs.copy(),
                action=int(action),
                reward=float(reward),
                value=max_q,
                logits=q_logits,
            )
            obs = next_obs

        crashed = done and not truncated
        rollout.end_episode(crashed=crashed)

    env.close()
    return rollout


# ──────────────────────────────────────────────────────────────────────────────
# DQN + PLR Training
# ──────────────────────────────────────────────────────────────────────────────
def train_dqn_plr(
    total_timesteps: int = 500_000,
    steps_per_level: int = 10_000,
    eval_episodes: int = 5,
    # PLR
    plr_strategy: str = "value_l1",
    plr_rho: float = 0.5,
    plr_nu: float = 0.5,
    plr_score_transform: str = "rank",
    plr_temperature: float = 0.1,
    plr_staleness_coef: float = 0.1,
    plr_alpha: float = 1.0,
    # DQN hyperparameters (tuned for multi-level learning)
    learning_rate: float = 3e-4,   # Lower LR for stability
    buffer_size: int = 50_000,     # Larger buffer to retain diverse experience
    learning_starts: int = 2_000,  # More warmup before training
    batch_size: int = 128,         # Larger batches for generalization
    gamma: float = 0.95,           # Longer horizon to see collision consequences
    train_freq: int = 1,
    gradient_steps: int = 1,
    target_update_interval: int = 500,  # More stable target network
    exploration_fraction: float = 0.4,  # Longer exploration for diverse scenarios
    exploration_final_eps: float = 0.05,
    # General
    save_dir: str = "highway_dqn_plr",
    verbose: int = 1,
):
    device = get_device()

    print(f"\n{'=' * 72}")
    print("  DQN + PLR  —  Fixed Implementation")
    print(f"{'=' * 72}")
    print(f"  Device .............. {device}")
    print(f"  Total timesteps ..... {total_timesteps:,}")
    print(f"  Steps per level ..... {steps_per_level:,}")
    print(f"  PLR strategy ........ {plr_strategy}")
    print(f"  Gamma ............... {gamma}")
    print(f"  Buffer size ......... {buffer_size:,}")
    print(f"  Target update ....... every {target_update_interval} steps")
    print(f"  NOTE: Buffer is CLEARED on env switch to avoid corruption")
    print(f"{'=' * 72}\n")

    # ── configs & PLR ──
    env_configs = generate_env_configs()
    n_configs = len(env_configs)
    print(f"Generated {n_configs} environment configurations\n")

    sampler = LevelSampler(
        num_levels=n_configs,
        strategy=plr_strategy,
        rho=plr_rho,
        nu=plr_nu,
        score_transform=plr_score_transform,
        temperature=plr_temperature,
        staleness_coef=plr_staleness_coef,
        alpha=plr_alpha,
    )

    # ── initial env & model ──
    level_idx = sampler.sample()
    config = env_configs[level_idx]
    vec_env = make_vec_env(config, NUM_ENVS)

    model = DQN(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[512, 512]),  # Larger network for multi-level capacity
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=0,
        tensorboard_log=None,
        device=device,
    )

    # ── bookkeeping ──
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {
        "configs": [], "rewards": [], "crash_rates": [],
        "ep_lengths": [], "timesteps": [], "fps": [], "plr_scores": [],
    }
    stats = RollingStats(window=20)
    timesteps_done = 0
    iteration = 0

    pbar = tqdm(total=total_timesteps, desc="DQN+PLR", unit="step",
                dynamic_ncols=True)
    wall_start = time.perf_counter()

    while timesteps_done < total_timesteps:
        iteration += 1
        steps_this = min(steps_per_level, total_timesteps - timesteps_done)

        model.learn(
            total_timesteps=steps_this,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        timesteps_done += steps_this
        pbar.update(steps_this)

        # ── PLR scoring ──
        rollout = evaluate_level_dqn(
            model, config, level_idx,
            n_episodes=eval_episodes,
            gamma=gamma,
        )
        plr_score = rollout.compute_score(plr_strategy, gamma=gamma)
        sampler.update_score(level_idx, plr_score)
        sampler.update_staleness(level_idx)

        stats.update(
            mean_reward=rollout.mean_reward,
            crash_count=rollout.crash_count,
            n_episodes=eval_episodes,
            mean_ep_length=rollout.mean_ep_length,
        )

        # ── logging ──
        elapsed = time.perf_counter() - wall_start
        fps = timesteps_done / max(elapsed, 1e-6)

        if verbose:
            plr_stats = sampler.get_stats()
            crash_str = (
                f"{rollout.crash_count}/{eval_episodes} crashed"
                if rollout.crash_count > 0 else "no crashes"
            )
            tqdm.write(
                f"\n  ┌─ Iter {iteration:>3d}  │  "
                f"{timesteps_done:,}/{total_timesteps:,} steps  │  "
                f"{elapsed:.0f}s"
            )
            tqdm.write(
                f"  │  Level: {describe_config(config)}  "
                f"(idx={level_idx}, {plr_stats['plr/seen']}/{n_configs} seen)"
            )
            tqdm.write(
                f"  │  R={rollout.mean_reward:>7.2f}  "
                f"Avg({stats.window})={stats.avg_reward:.2f}  "
                f"Best={stats.best_reward:.2f}  "
                f"Trend={stats.reward_trend}"
            )
            tqdm.write(
                f"  │  Surv={stats.survival_rate:.1f}%  {crash_str}  "
                f"PLR={plr_score:.4f}"
            )
            tqdm.write(f"  └─ FPS: {fps:.0f}")

        pbar.set_postfix_str(
            f"R={stats.avg_reward:.1f}  "
            f"Surv={stats.survival_rate:.0f}%  "
            f"{fps:.0f} FPS"
        )

        history["configs"].append(int(level_idx))
        history["rewards"].append(float(rollout.mean_reward))
        history["crash_rates"].append(float(stats.crash_rate))
        history["ep_lengths"].append(float(rollout.mean_ep_length))
        history["timesteps"].append(int(timesteps_done))
        history["fps"].append(float(fps))
        history["plr_scores"].append(float(plr_score))

        # ── next level: clear buffer to avoid cross-env pollution ──
        level_idx = sampler.sample()
        config = env_configs[level_idx]
        vec_env.close()
        vec_env = make_vec_env(config, NUM_ENVS)
        model.set_env(vec_env)

        # CRITICAL: reset replay buffer to avoid training on transitions
        # from a different environment config
        model.replay_buffer.reset()

    pbar.close()
    vec_env.close()

    wall_total = time.perf_counter() - wall_start
    avg_fps = total_timesteps / max(wall_total, 1e-6)

    # ── save ──
    model.save(save_path / "model")
    sampler.save(str(save_path / "plr_state.pkl"))

    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "algorithm": "DQN",
        "total_timesteps": total_timesteps,
        "steps_per_level": steps_per_level,
        "num_envs": NUM_ENVS,
        "device": device,
        "wall_time_s": round(wall_total, 1),
        "avg_fps": round(avg_fps, 1),
        "plr": {
            "strategy": plr_strategy,
            "rho": plr_rho,
            "nu": plr_nu,
            "score_transform": plr_score_transform,
        },
        "dqn": {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "gamma": gamma,
            "target_update_interval": target_update_interval,
        },
        "num_configs": n_configs,
    }
    with open(save_path / "training_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    plr_final = sampler.get_stats()
    print(f"\n{'=' * 72}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Duration ............ {wall_total/60:.1f} min")
    print(f"  Average FPS ......... {avg_fps:.0f}")
    print(f"  Avg reward (all) .... {stats.avg_reward_all:.2f}")
    print(f"  Avg reward (last {stats.window})  {stats.avg_reward:.2f}")
    print(f"  Best reward ......... {stats.best_reward:.2f}")
    print(f"  Survival rate ....... {stats.survival_rate:.1f}%")
    print(f"  Levels seen ......... {plr_final['plr/seen']}/{n_configs}")
    print(f"  Saved to ............ {save_path}")
    print(f"{'=' * 72}\n")

    return model, sampler, history


# ──────────────────────────────────────────────────────────────────────────────
# Test & Watch — reuse from ppo_plr
# ──────────────────────────────────────────────────────────────────────────────
from ppo_plr import evaluate_on_test_configs, watch_agent  # noqa: E402


def main():
    TRAIN = True
    SKIP_EVALUATION = False

    TOTAL_TIMESTEPS  = 2_000_000  # 2M steps for multi-level mastery
    STEPS_PER_LEVEL  = 60_000     # DQN needs extensive training per level
    EVAL_EPISODES    = 3
    SAVE_DIR = "highway_dqn_plr/v7"

    if TRAIN:
        model, sampler, history = train_dqn_plr(
            total_timesteps=TOTAL_TIMESTEPS,
            steps_per_level=STEPS_PER_LEVEL,
            eval_episodes=EVAL_EPISODES,
            save_dir=SAVE_DIR,
            verbose=1,
        )
    else:
        device = get_device()
        model = DQN.load(f"{SAVE_DIR}/model", device=device)

    if not SKIP_EVALUATION:
        test_cfgs = generate_test_configs()
        results = evaluate_on_test_configs(model, test_cfgs, n_episodes=10)
        with open(Path(SAVE_DIR) / "test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    watch_cfgs = generate_test_configs()
    watch_agent(model, watch_cfgs, n_episodes=3)


if __name__ == "__main__":
    main()
