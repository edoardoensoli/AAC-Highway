"""
QR-DQN + PLR⊥ (Robust Prioritized Level Replay) — Highway-env
================================================================

Off-policy curriculum learning using Quantile-Regression DQN from
``sb3-contrib``.  QR-DQN is recommended over standard DQN for PLR because:

1. **Distributional value estimates** — the quantile representation captures
   return uncertainty, which makes the agent more robust to the diverse
   MDPs produced by the curriculum.

2. **Better sample efficiency** — off-policy replay can reuse transitions
   from earlier levels, reducing the data-wastage inherent in PLR⊥ where
   exploration rollouts are discarded without gradient updates.

3. **Discrete action support** — unlike SAC (continuous), QR-DQN works
   natively with highway-env's ``DiscreteMetaAction`` space.

Architecture: OccupancyGridCNN feature extractor (spatial CNN for the
              2-D occupancy grid observation).

Scoring:      MaxMC (default) — ground-truth regret proxy.

Dependencies: ``pip install sb3-contrib==1.8.0``

Reference
---------
Dabney et al., "Distributional Reinforcement Learning with Quantile
Regression", AAAI 2018.
Jiang et al., "Prioritized Level Replay", arXiv:2010.03934.
"""

from __future__ import annotations

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

from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from plr import LevelSampler, LevelScorer, LevelRollout, RollingStats
from plr_configs import (
    HighwayLevelFactory,
    HighwayLevelWrapper,
    seed_to_config,
    seed_to_factors,
    config_without_env_id,
    TRAIN_SEEDS,
    TEST_SEEDS,
)
from feature_extractors import OccupancyGridCNN


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ──────────────────────────────────────────────────────────────────────────────
NUM_ENVS = 4


def _make_wrapped_env(seed: int, rank: int = 0):
    """Return a callable that creates one wrapped env from a seed."""
    cfg = seed_to_config(seed)
    env_config = config_without_env_id(cfg)

    def _init():
        env = gym.make("highway-v0", config=env_config)
        return HighwayLevelWrapper(env, seed=seed)

    return _init


def make_vec_env(seed: int, n_envs: int = NUM_ENVS):
    if n_envs == 1:
        return DummyVecEnv([_make_wrapped_env(seed, 0)])
    return SubprocVecEnv([_make_wrapped_env(seed, i) for i in range(n_envs)])


def make_single_env(seed: int, render: bool = False):
    factory = HighwayLevelFactory()
    return factory.make_env(seed, render=render)


# ──────────────────────────────────────────────────────────────────────────────
# PLR Scoring Rollout
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_level_qrdqn(
    model: QRDQN,
    seed: int,
    level_idx: int,
    n_episodes: int = 5,
    gamma: float = 0.99,
    strategy: str = "max_mc",
) -> LevelRollout:
    """
    Collect a scoring rollout on level ``seed`` using QR-DQN.

    For MaxMC scoring, only episode returns are needed.  For value-based
    strategies the mean quantile Q-value is used as the value estimate.
    """
    device = model.device
    rollout = LevelRollout(level_idx=level_idx)
    env = make_single_env(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            # Get Q-value estimate from quantile network
            obs_t = torch.as_tensor(
                obs[np.newaxis].astype(np.float32), device=device
            )
            try:
                # QR-DQN: quantile_net returns mean Q-values
                q_values = model.policy.quantile_net(obs_t)
                if q_values.dim() == 3:
                    # (1, n_actions, n_quantiles) → mean across quantiles
                    q_values = q_values.mean(dim=-1)
                max_q = float(q_values.max(dim=1)[0].cpu())
                q_logits = q_values.cpu().numpy().squeeze()
            except Exception:
                max_q = 0.0
                q_logits = np.zeros(env.action_space.n, dtype=np.float32)

            action, _ = model.predict(obs, deterministic=False)
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
# QR-DQN + PLR⊥ Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_qrdqn_plr_robust(
    total_timesteps: int = 5_000_000,
    steps_per_level: int = 10_000,
    eval_episodes: int = 5,
    # PLR⊥ parameters
    plr_strategy: str = "max_mc",
    plr_rho: float = 0.6,
    plr_nu: float = 0.7,              # replay rate ≈ 0.3
    plr_score_transform: str = "rank",
    plr_temperature: float = 0.1,
    plr_staleness_coef: float = 0.1,
    plr_alpha: float = 1.0,
    robust: bool = True,
    # QR-DQN hyperparameters
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 5_000,
    batch_size: int = 128,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1_000,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    n_quantiles: int = 50,
    # General
    save_dir: str = "highway_qrdqn_plr",
    verbose: int = 1,
):
    """
    Train QR-DQN with PLR⊥ (Robust Prioritized Level Replay).

    Off-policy curriculum: the replay buffer retains transitions across
    level switches, providing additional sample reuse beyond what PPO
    can achieve.  Combined with MaxMC scoring, this produces a robust
    agent that generalises to unseen highway configurations.
    """
    device = get_device()
    factory = HighwayLevelFactory()
    n_train = factory.num_train_levels
    mode_label = "PLR⊥ (Robust)" if robust else "PLR (Standard)"

    # ── banner ──
    print(f"\n{'=' * 72}")
    print(f"  QR-DQN + {mode_label}")
    print(f"{'=' * 72}")
    print(f"  Device .............. {device}")
    print(f"  Parallel envs ....... {NUM_ENVS}")
    print(f"  Total timesteps ..... {total_timesteps:,}")
    print(f"  Steps per level ..... {steps_per_level:,}")
    print(f"  PLR strategy ........ {plr_strategy}")
    print(f"  PLR rho ............. {plr_rho}")
    print(f"  PLR nu .............. {plr_nu}")
    print(f"  Robust (PLR⊥) ....... {robust}")
    print(f"  QR-DQN quantiles .... {n_quantiles}")
    print(f"  Buffer size ......... {buffer_size:,}")
    print(f"  Training levels ..... {n_train}")
    print(f"  Test levels ......... {factory.num_test_levels}")
    print(f"{'=' * 72}\n")

    # ── PLR sampler ──
    sampler = LevelSampler(
        num_levels=n_train,
        strategy=plr_strategy,
        rho=plr_rho,
        nu=plr_nu,
        score_transform=plr_score_transform,
        temperature=plr_temperature,
        staleness_coef=plr_staleness_coef,
        alpha=plr_alpha,
    )

    # ── initial env & model ──
    level_idx, is_replay = sampler.sample_with_decision()
    seed = factory.train_seeds[level_idx]
    vec_env = make_vec_env(seed, NUM_ENVS)

    model = QRDQN(
        "CnnPolicy",
        vec_env,
        policy_kwargs=dict(
            features_extractor_class=OccupancyGridCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[256, 256],
            n_quantiles=n_quantiles,
        ),
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
        "configs": [], "seeds": [], "rewards": [], "crash_rates": [],
        "ep_lengths": [], "timesteps": [], "fps": [], "plr_scores": [],
        "is_replay": [],
    }
    stats = RollingStats(window=20)
    timesteps_done = 0
    explore_count = 0
    replay_count = 0
    iteration = 0
    last_checkpoint = 0
    CHECKPOINT_INTERVAL = 100_000

    pbar = tqdm(total=total_timesteps, desc=f"QRDQN+{mode_label}",
                unit="step", dynamic_ncols=True)
    wall_start = time.perf_counter()

    # ── main loop ──
    while timesteps_done < total_timesteps:
        iteration += 1
        steps_this = min(steps_per_level, total_timesteps - timesteps_done)

        # ── PLR⊥ decision ──
        if is_replay or not robust:
            # REPLAY → train with gradient updates
            ts_before = model.num_timesteps
            model.learn(
                total_timesteps=steps_this,
                reset_num_timesteps=False,
                progress_bar=False,
            )
            actual_steps = model.num_timesteps - ts_before
            timesteps_done += actual_steps
            pbar.update(actual_steps)
            replay_count += 1

            # ── checkpoint ──
            if timesteps_done - last_checkpoint >= CHECKPOINT_INTERVAL:
                model.save(save_path / "checkpoint")
                last_checkpoint = timesteps_done
                if verbose:
                    tqdm.write(f"  Checkpoint saved at {timesteps_done:,} steps")
        else:
            # EXPLORE → score only, NO gradient update
            explore_count += 1

        # ── score the level ──
        rollout = score_level_qrdqn(
            model, seed, level_idx,
            n_episodes=eval_episodes,
            gamma=gamma,
            strategy=plr_strategy,
        )

        # ── MaxMC: regret = max_ever − current mean ──
        if plr_strategy == "max_mc":
            sampler.update_with_maxmc(
                level_idx, rollout.max_episode_return, rollout.mean_reward
            )
            plr_score = sampler.scores[level_idx]
        else:
            plr_score = rollout.compute_score(plr_strategy, gamma=gamma)
            sampler.update_score(level_idx, plr_score)
        sampler.update_staleness(level_idx)

        # ── stats ──
        stats.update(
            mean_reward=rollout.mean_reward,
            crash_count=rollout.crash_count,
            n_episodes=eval_episodes,
            mean_ep_length=rollout.mean_ep_length,
        )

        elapsed = time.perf_counter() - wall_start
        fps = timesteps_done / max(elapsed, 1e-6)

        if verbose:
            plr_stats = sampler.get_stats()
            crash_str = (
                f"{rollout.crash_count}/{eval_episodes} crashed"
                if rollout.crash_count > 0 else "no crashes"
            )
            mode_tag = (
                "REPLAY → train"
                if (is_replay or not robust) else "EXPLORE → score only"
            )
            tqdm.write(
                f"\n  ┌─ Iter {iteration:>3d}  │  "
                f"{timesteps_done:,}/{total_timesteps:,} steps  │  "
                f"{elapsed:.0f}s"
            )
            tqdm.write(
                f"  │  [{mode_tag}]  Level idx={level_idx}  "
                f"{factory.describe_level(level_idx)}  "
                f"({plr_stats['plr/seen']}/{n_train} seen)"
            )
            tqdm.write(
                f"  │  R={rollout.mean_reward:>7.2f}  "
                f"Avg({stats.window})={stats.avg_reward:>7.2f}  "
                f"Best={stats.best_reward:.2f}  "
                f"Trend={stats.reward_trend}"
            )
            tqdm.write(
                f"  │  Surv={stats.survival_rate:>5.1f}%  {crash_str}  "
                f"PLR={plr_score:.4f}"
            )
            tqdm.write(
                f"  └─ FPS: {fps:.0f}  "
                f"Replay/Explore: {replay_count}/{explore_count}"
            )

        pbar.set_postfix_str(
            f"R={stats.avg_reward:.1f} Surv={stats.survival_rate:.0f}% "
            f"{fps:.0f}FPS R/E={replay_count}/{explore_count}"
        )

        # ── history ──
        history["configs"].append(int(level_idx))
        history["seeds"].append(int(seed))
        history["rewards"].append(float(rollout.mean_reward))
        history["crash_rates"].append(float(stats.crash_rate))
        history["ep_lengths"].append(float(rollout.mean_ep_length))
        history["timesteps"].append(int(timesteps_done))
        history["fps"].append(float(fps))
        history["plr_scores"].append(float(plr_score))
        history["is_replay"].append(bool(is_replay or not robust))

        # ── sample next level ──
        level_idx, is_replay = sampler.sample_with_decision()
        seed = factory.train_seeds[level_idx]
        vec_env.close()
        vec_env = make_vec_env(seed, NUM_ENVS)
        model.set_env(vec_env)

        # NOTE: We do NOT clear the replay buffer.  Off-policy methods
        # tolerate distribution shift via the replay buffer — this is
        # exactly the advantage over PPO for PLR curriculum learning.

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
        "algorithm": "QR-DQN",
        "plr_variant": "PLR⊥ (Robust)" if robust else "PLR (Standard)",
        "total_timesteps": total_timesteps,
        "steps_per_level": steps_per_level,
        "num_envs": NUM_ENVS,
        "device": device,
        "wall_time_s": round(wall_total, 1),
        "avg_fps": round(avg_fps, 1),
        "replay_iterations": replay_count,
        "explore_iterations": explore_count,
        "plr": {
            "strategy": plr_strategy,
            "rho": plr_rho,
            "nu": plr_nu,
            "robust": robust,
            "score_transform": plr_score_transform,
        },
        "qrdqn": {
            "learning_rate": learning_rate,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "gamma": gamma,
            "n_quantiles": n_quantiles,
            "target_update_interval": target_update_interval,
        },
        "num_train_levels": n_train,
        "num_test_levels": factory.num_test_levels,
        "train_seeds": factory.train_seeds,
        "test_seeds": factory.test_seeds,
    }
    with open(save_path / "training_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── summary ──
    plr_final = sampler.get_stats()
    print(f"\n{'=' * 72}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Duration ............ {wall_total/60:.1f} min ({wall_total:.0f}s)")
    print(f"  Average FPS ......... {avg_fps:.0f}")
    print(f"  Replay iters ........ {replay_count}")
    print(f"  Explore iters ....... {explore_count}")
    print(f"{'=' * 72}")
    print(f"  AGENT PERFORMANCE (training levels)")
    print(f"  {'─' * 40}")
    print(f"  Avg reward (all) .... {stats.avg_reward_all:.2f}")
    print(f"  Avg reward (last {stats.window})  {stats.avg_reward:.2f}")
    print(f"  Best reward ......... {stats.best_reward:.2f}")
    print(f"  Survival rate ....... {stats.survival_rate:.1f}%")
    print(f"{'=' * 72}")
    print(f"  PLR CURRICULUM")
    print(f"  {'─' * 40}")
    print(f"  Strategy ............ {plr_strategy}")
    print(f"  Levels seen ......... {plr_final['plr/seen']}/{n_train}")
    print(f"  Mean PLR score ...... {plr_final['plr/mean_score']:.4f}")
    print(f"  Max PLR score ....... {plr_final['plr/max_score']:.4f}")
    print(f"{'=' * 72}")
    print(f"  Saved to ............ {save_path}")
    print(f"{'=' * 72}\n")

    return model, sampler, history


# ──────────────────────────────────────────────────────────────────────────────
# Held-out evaluation  (reuse from ppo_plr)
# ──────────────────────────────────────────────────────────────────────────────
from ppo_plr import evaluate_on_test_configs, watch_agent  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ====== FLAGS ===========================================================
    TRAIN           = True
    SKIP_EVALUATION = False

    TOTAL_TIMESTEPS  = 5_000_000
    STEPS_PER_LEVEL  = 10_000
    EVAL_EPISODES    = 5

    PLR_STRATEGY         = "max_mc"
    PLR_RHO              = 0.6
    PLR_NU               = 0.7      # replay rate ≈ 0.3
    PLR_SCORE_TRANSFORM  = "rank"
    PLR_TEMPERATURE      = 0.1
    PLR_STALENESS_COEF   = 0.1
    PLR_ALPHA            = 1.0
    ROBUST               = True

    SAVE_DIR = "highway_qrdqn_plr/v1"
    # ========================================================================

    if TRAIN:
        model, sampler, history = train_qrdqn_plr_robust(
            total_timesteps=TOTAL_TIMESTEPS,
            steps_per_level=STEPS_PER_LEVEL,
            eval_episodes=EVAL_EPISODES,
            plr_strategy=PLR_STRATEGY,
            plr_rho=PLR_RHO,
            plr_nu=PLR_NU,
            plr_score_transform=PLR_SCORE_TRANSFORM,
            plr_temperature=PLR_TEMPERATURE,
            plr_staleness_coef=PLR_STALENESS_COEF,
            plr_alpha=PLR_ALPHA,
            robust=ROBUST,
            save_dir=SAVE_DIR,
            verbose=1,
        )
    else:
        device = get_device()
        model = QRDQN.load(f"{SAVE_DIR}/checkpoint", device=device)

    # ── Λ_test evaluation (zero-shot generalisation) ──
    if not SKIP_EVALUATION:
        test_results = evaluate_on_test_configs(model, n_episodes=10)
        save_path = Path(SAVE_DIR)
        with open(save_path / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)

    # ── watch on test levels ──
    watch_seeds = TEST_SEEDS[:5]
    watch_agent(model, watch_seeds, n_episodes=2)


if __name__ == "__main__":
    main()
