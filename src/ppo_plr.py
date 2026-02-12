"""
PPO + PLR⊥ (Robust Prioritized Level Replay) — Highway-env
============================================================

PLR⊥ (Jiang et al., 2021 §4.2) restricts gradient updates to **replayed**
levels.  When the sampler decides to *explore* (discover new levels), the
agent only collects a scoring rollout — no weight updates.  This eliminates
noise from random discovery levels and keeps gradient signal focused on the
learning frontier.

Scoring: ``estimated_regret`` (default) or ``one_step_td_error``.
Level space: seed-based parametric mapping to (lanes_count, vehicles_count,
             vehicles_density, POLITENESS).

Architecture
------------
• plr.py                — PLR⊥ sampling + scoring (algorithm-agnostic)
• plr_configs.py        — seed→factor mapping, HighwayLevelWrapper,
                           HighwayLevelFactory, disjoint Λ_train / Λ_test
• ppo_plr.py (THIS FILE) — PPO training loop + PLR⊥ integration +
                            held-out Λ_test zero-shot evaluation

Optimised for MacBook Air M1 (MPS backend, 2–4 parallel envs).
"""

import gymnasium as gym
import highway_env
import torch
import numpy as np
import json
import time
import pickle

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from highway_env.vehicle.behavior import IDMVehicle

from plr import LevelSampler, LevelScorer, LevelRollout, RollingStats
from plr_configs import (
    HighwayLevelFactory,
    HighwayLevelWrapper,
    seed_to_config,
    seed_to_factors,
    describe_config,
    config_without_env_id,
    get_env_id,
    generate_env_configs,
    generate_test_configs,
    TRAIN_SEEDS,
    TEST_SEEDS,
)


# ──────────────────────────────────────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# Environment helpers  (wraps into HighwayLevelWrapper for POLITENESS)
# ──────────────────────────────────────────────────────────────────────────────
NUM_ENVS = 4


class TqdmUpdateCallback(BaseCallback):
    """Updates the shared tqdm bar on every training step for real-time feedback."""

    def __init__(self, pbar: tqdm):
        super().__init__()
        self._pbar = pbar

    def _on_step(self) -> bool:
        self._pbar.update(self.training_env.num_envs)
        return True


def _make_wrapped_env(seed: int, rank: int = 0):
    """Return a callable that creates one wrapped env from a seed."""
    cfg = seed_to_config(seed)
    env_config = config_without_env_id(cfg)

    def _init():
        env = gym.make("highway-v0", config=env_config)
        return HighwayLevelWrapper(env, seed=seed)

    return _init


def make_vec_env(seed: int, n_envs: int = NUM_ENVS):
    """Create a SubprocVecEnv of wrapped environments from a single seed."""
    if n_envs == 1:
        return DummyVecEnv([_make_wrapped_env(seed, 0)])
    return SubprocVecEnv([_make_wrapped_env(seed, i) for i in range(n_envs)])


def make_single_env(seed: int, render: bool = False):
    """Create a single wrapped env for evaluation."""
    factory = HighwayLevelFactory()
    return factory.make_env(seed, render=render)


# ──────────────────────────────────────────────────────────────────────────────
# PLR scoring rollout  (no gradient updates — data only)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_level(
    model: PPO,
    seed: int,
    level_idx: int,
    n_episodes: int = 5,
    gamma: float = 0.99,
    strategy: str = "estimated_regret",
) -> LevelRollout:
    """
    Collect a rollout on level ``seed`` for PLR scoring.

    No weight updates — purely forward passes.  Returns a
    :class:`LevelRollout` whose ``compute_score()`` method produces
    the PLR priority.
    """
    device = model.device
    rollout = LevelRollout(level_idx=level_idx)
    env = make_single_env(seed)

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False

        while not (done or truncated):
            obs_t = torch.as_tensor(
                obs[np.newaxis].astype(np.float32), device=device
            )
            value = model.policy.predict_values(obs_t)
            val = float(value.cpu().squeeze())

            dist = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy().squeeze()

            action, _ = model.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, _ = env.step(action)

            rollout.add_step(
                obs=obs.copy(),
                action=int(action),
                reward=float(reward),
                value=val,
                logits=logits,
            )
            obs = next_obs

        crashed = done and not truncated
        rollout.end_episode(crashed=crashed)

    env.close()
    return rollout


# ──────────────────────────────────────────────────────────────────────────────
# PPO + PLR⊥ Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_ppo_plr_robust(
    total_timesteps: int = 2_000_000,
    steps_per_level: int = 4_096,
    eval_episodes: int = 10,
    # PLR⊥ parameters
    plr_strategy: str = "estimated_regret",
    plr_rho: float = 0.6,
    plr_nu: float = 0.5,
    plr_score_transform: str = "rank",
    plr_temperature: float = 0.1,
    plr_staleness_coef: float = 0.1,
    plr_alpha: float = 1.0,
    robust: bool = True,           # PLR⊥: skip gradient on explore levels
    # PPO hyperparameters
    learning_rate: float = 2e-4,
    n_steps: int = 1024,
    batch_size: int = 256,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    # General
    save_dir: str = "highway_ppo_plr",
    verbose: int = 1,
):
    """
    Train PPO with PLR⊥ (Robust Prioritized Level Replay).

    Core loop per iteration:
    1. ``sampler.sample_with_decision()``  →  (level_idx, is_replay)
    2. **If** ``is_replay`` → train PPO for ``steps_per_level`` steps.
       **Else** → run scoring rollout only (no gradient).
    3. Score the level and update the sampler.
    """
    device = get_device()
    factory = HighwayLevelFactory()
    n_train = factory.num_train_levels
    mode_label = "PLR⊥ (Robust)" if robust else "PLR (Standard)"

    # ── banner ──
    print(f"\n{'=' * 72}")
    print(f"  PPO + {mode_label}")
    print(f"{'=' * 72}")
    print(f"  Device .............. {device}")
    print(f"  Parallel envs ....... {NUM_ENVS}")
    print(f"  Total timesteps ..... {total_timesteps:,}")
    print(f"  Steps per level ..... {steps_per_level:,}")
    print(f"  Eval episodes ....... {eval_episodes}")
    print(f"  PLR strategy ........ {plr_strategy}")
    print(f"  PLR rho ............. {plr_rho}")
    print(f"  PLR nu .............. {plr_nu}")
    print(f"  Robust (PLR⊥) ....... {robust}")
    print(f"  Training levels ..... {n_train}  (seeds {TRAIN_SEEDS[0]}–{TRAIN_SEEDS[-1]})")
    print(f"  Test levels ......... {factory.num_test_levels}  (seeds {TEST_SEEDS[0]}–{TEST_SEEDS[-1]})")
    print(f"  Learning rate ....... {learning_rate}")
    print(f"  PPO n_steps ......... {n_steps}")
    print(f"  PPO batch_size ...... {batch_size}")
    print(f"  Gamma ............... {gamma}")
    print(f"  Entropy coef ........ {ent_coef}")
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

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
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

    pbar = tqdm(total=total_timesteps, desc=f"PPO+{mode_label}",
                unit="step", dynamic_ncols=True)
    wall_start = time.perf_counter()

    # ── main loop ──
    while timesteps_done < total_timesteps:
        iteration += 1
        steps_this = min(steps_per_level, total_timesteps - timesteps_done)

        # ── PLR⊥ decision ──
        if is_replay or not robust:
            # REPLAY level → train with gradient updates
            ts_before = model.num_timesteps
            model.learn(
                total_timesteps=steps_this,
                reset_num_timesteps=False,
                progress_bar=False,
                callback=TqdmUpdateCallback(pbar),
            )
            actual_steps = model.num_timesteps - ts_before
            timesteps_done += actual_steps
            replay_count += 1

            # ── checkpoint every 100k steps (overwrite) ──
            if timesteps_done - last_checkpoint >= CHECKPOINT_INTERVAL:
                model.save(save_path / "checkpoint")
                last_checkpoint = timesteps_done
                if verbose:
                    tqdm.write(f"  💾 Checkpoint saved at {timesteps_done:,} steps")
        else:
            # EXPLORE level → score only, NO gradient update
            explore_count += 1
            # (timesteps are NOT added to budget since no training happened)

        # ── score the level ──
        rollout = score_level(
            model, seed, level_idx,
            n_episodes=eval_episodes,
            gamma=gamma,
            strategy=plr_strategy,
        )
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
            mode_tag = "REPLAY → train" if (is_replay or not robust) else "EXPLORE → score only"
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
        "algorithm": "PPO",
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
            "temperature": plr_temperature,
            "staleness_coef": plr_staleness_coef,
            "alpha": plr_alpha,
        },
        "ppo": {
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
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
    print(f"  Reward trend ........ {stats.reward_trend}")
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
# Held-out Λ_test evaluation  (zero-shot generalisation)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_on_test_configs(model, n_episodes: int = 10):
    """
    Evaluate on the **held-out** Λ_test seeds that were never seen during
    PLR sampling.  Measures true zero-shot generalisation.
    """
    factory = HighwayLevelFactory()

    print(f"\n{'=' * 72}")
    print("  EVALUATING ON Λ_test  (held-out, zero-shot generalisation)")
    print(f"{'=' * 72}")
    print(f"  Episodes per level .. {n_episodes}")
    print(f"  Test levels ......... {factory.num_test_levels}")
    print(f"  Test seed range ..... {TEST_SEEDS[0]}–{TEST_SEEDS[-1]}")
    print(f"{'=' * 72}\n")

    results = []
    all_rewards: List[float] = []
    all_crashes = 0
    all_lengths: List[int] = []
    total_episodes = 0

    for i in range(factory.num_test_levels):
        seed = factory.test_seeds[i]
        env = factory.make_test_env(i)
        rewards: List[float] = []
        ep_lengths: List[int] = []
        crashes = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False
            ep_r = 0.0
            steps = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                ep_r += r
                steps += 1
            rewards.append(ep_r)
            ep_lengths.append(steps)
            if done and not truncated:
                crashes += 1
        env.close()

        avg = float(np.mean(rewards))
        std = float(np.std(rewards))
        med = float(np.median(rewards))
        best_r = float(np.max(rewards))
        worst_r = float(np.min(rewards))
        cr = 100.0 * crashes / n_episodes
        surv = 100.0 - cr
        avg_len = float(np.mean(ep_lengths))

        results.append({
            "seed": seed,
            "config_desc": factory.describe_test_level(i),
            "factors": seed_to_factors(seed),
            "avg_reward": avg, "std_reward": std,
            "median_reward": med, "best_reward": best_r,
            "worst_reward": worst_r,
            "crash_rate": cr, "survival_rate": surv,
            "avg_ep_length": avg_len, "crashes": crashes,
        })

        all_rewards.extend(rewards)
        all_crashes += crashes
        all_lengths.extend(ep_lengths)
        total_episodes += n_episodes

        crash_str = (
            f"{crashes}/{n_episodes} crashed"
            if crashes > 0 else "no crashes"
        )
        print(
            f"  ┌─ Test {i+1}/{factory.num_test_levels}  │  "
            f"{factory.describe_test_level(i)}"
        )
        print(
            f"  │  R={avg:>7.2f} ± {std:.2f}  "
            f"Med={med:.2f}  Best={best_r:.2f}  Worst={worst_r:.2f}"
        )
        print(
            f"  └─ Surv={surv:>5.1f}%  {crash_str}  "
            f"Avg length={avg_len:.0f} steps\n"
        )

    overall_avg = float(np.mean(all_rewards))
    overall_std = float(np.std(all_rewards))
    overall_cr = 100.0 * all_crashes / max(total_episodes, 1)
    overall_surv = 100.0 - overall_cr
    overall_len = float(np.mean(all_lengths))

    print(f"{'=' * 72}")
    print("  Λ_test SUMMARY  (zero-shot generalisation)")
    print(f"  {'─' * 40}")
    print(f"  Total episodes ...... {total_episodes}")
    print(f"  Avg reward .......... {overall_avg:.2f} ± {overall_std:.2f}")
    print(f"  Survival rate ....... {overall_surv:.1f}%  "
          f"({all_crashes}/{total_episodes} crashed)")
    print(f"  Avg episode length .. {overall_len:.0f} steps")
    print(f"{'=' * 72}\n")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate on Λ_train  (in-distribution check)
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_on_train_configs(model, n_episodes: int = 5):
    """Evaluate on training seeds for in-distribution reference."""
    factory = HighwayLevelFactory()

    print(f"\n{'=' * 72}")
    print("  EVALUATING ON Λ_train  (in-distribution reference)")
    print(f"{'=' * 72}\n")

    results = []
    for i in range(factory.num_train_levels):
        seed = factory.train_seeds[i]
        env = factory.make_train_env(i)
        rewards, crashes = [], 0
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False
            ep_r = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                ep_r += r
            rewards.append(ep_r)
            if done and not truncated:
                crashes += 1
        env.close()

        avg = float(np.mean(rewards))
        surv = 100.0 * (1 - crashes / n_episodes)
        results.append({
            "seed": seed,
            "desc": factory.describe_level(i),
            "avg_reward": avg,
            "survival_rate": surv,
        })
        print(f"  [{i:2d}] {factory.describe_level(i)}  "
              f"R={avg:.2f}  Surv={surv:.0f}%")

    avg_all = float(np.mean([r["avg_reward"] for r in results]))
    surv_all = float(np.mean([r["survival_rate"] for r in results]))
    print(f"\n  Overall: R={avg_all:.2f}  Surv={surv_all:.1f}%\n")
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Watch agent
# ──────────────────────────────────────────────────────────────────────────────
def watch_agent(model, seeds: List[int], n_episodes: int = 3):
    """Render the agent on a list of level seeds."""
    factory = HighwayLevelFactory()

    print(f"\n{'=' * 60}")
    print("  WATCHING AGENT  (Ctrl-C to stop)")
    print(f"  {len(seeds)} levels  ×  {n_episodes} episodes each")
    print(f"{'=' * 60}\n")

    try:
        for si, seed in enumerate(seeds):
            factors = seed_to_factors(seed)
            print(f"  ── Level seed={seed}  "
                  f"L={factors['lanes_count']}  V={factors['vehicles_count']}  "
                  f"D={factors['vehicles_density']:.1f}  P={factors['politeness']:.2f}")
            env = factory.make_env(seed, render=True)
            try:
                for ep in range(n_episodes):
                    obs, _ = env.reset()
                    env.render()
                    done = truncated = False
                    ep_r, steps = 0.0, 0
                    while not (done or truncated):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, r, done, truncated, _ = env.step(action)
                        env.render()
                        ep_r += r
                        steps += 1
                    tag = "CRASH" if (done and not truncated) else "SURVIVED"
                    print(f"    Ep {ep+1}: {tag}  R={ep_r:.2f}  Steps={steps}")
            finally:
                env.close()
            print()
    except KeyboardInterrupt:
        print("\n  Stopped by user")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ====== FLAGS ===========================================================
    TRAIN           = True
    SKIP_EVALUATION = False

    # Training budget
    TOTAL_TIMESTEPS  = 2_000_000
    STEPS_PER_LEVEL  = 4_096
    EVAL_EPISODES    = 10

    # PLR⊥ — estimated_regret + rank (recommended)
    PLR_STRATEGY         = "estimated_regret"
    PLR_RHO              = 0.6
    PLR_NU               = 0.5
    PLR_SCORE_TRANSFORM  = "rank"
    PLR_TEMPERATURE      = 0.1
    PLR_STALENESS_COEF   = 0.1
    PLR_ALPHA            = 1.0
    ROBUST               = True  # PLR⊥: no gradient on explore levels

    SAVE_DIR = "highway_ppo_plr/v7"
    # ========================================================================

    if TRAIN:
        model, sampler, history = train_ppo_plr_robust(
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
        # Remember to use checkpoint or model based on what you want to test
        device = get_device()
        model = PPO.load(f"{SAVE_DIR}/checkpoint", device=device)

    # ── Λ_test evaluation (zero-shot generalisation) ──
    if not SKIP_EVALUATION:
        test_results = evaluate_on_test_configs(model, n_episodes=10)
        save_path = Path(SAVE_DIR)
        with open(save_path / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2, default=str)

        train_results = evaluate_on_train_configs(model, n_episodes=5)
        with open(save_path / "train_eval_results.json", "w") as f:
            json.dump(train_results, f, indent=2, default=str)
    else:
        print("Skipping evaluation, jumping to visualisation...")

    # ── watch on a sample of test levels ──
    watch_seeds = TEST_SEEDS[:5]
    watch_agent(model, watch_seeds, n_episodes=2)


if __name__ == "__main__":
    main()
