"""
PPO + PLR (Prioritized Level Replay) — Highway-env
====================================================

PPO is the natural fit for PLR because:
1. On-policy: no replay-buffer poisoning when switching environments.
2. GAE / value-loss from the training rollout directly measures learning
   potential (the PLR score signal).
3. Policy gradient methods handle stochastic environments better.

Architecture
------------
• plr.py            — algorithm-agnostic PLR components
• plr_configs.py    — environment configurations (15 levels)
• aggressive_vehicle.py — custom NPC for harder traffic
• ppo_plr.py        — THIS FILE: PPO training loop + PLR integration

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
from highway_env.vehicle.behavior import IDMVehicle

from plr import LevelSampler, LevelScorer, LevelRollout, RollingStats
from plr_configs import (
    generate_env_configs,
    generate_test_configs,
    config_without_env_id,
    get_env_id,
    describe_config,
)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Device detection
# ──────────────────────────────────────────────────────────────────────────────
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Environment helpers
# ──────────────────────────────────────────────────────────────────────────────
NUM_ENVS = 4  # M1 MacBook Air — keep low to avoid memory pressure


def _make_env(config: dict, rank: int = 0):
    """Return a callable that creates one env instance."""
    env_id = get_env_id(config)
    env_config = config_without_env_id(config)

    def _init():
        env = gym.make(env_id, config=env_config)
        return env

    return _init


def make_vec_env(config: dict, n_envs: int = NUM_ENVS):
    """Create a SubprocVecEnv with `n_envs` copies."""
    if n_envs == 1:
        return DummyVecEnv([_make_env(config, 0)])
    return SubprocVecEnv([_make_env(config, i) for i in range(n_envs)])


def make_single_env(config: dict, render: bool = False):
    env_id = get_env_id(config)
    env_config = config_without_env_id(config)
    render_mode = "human" if render else None
    return gym.make(env_id, config=env_config, render_mode=render_mode)


# ──────────────────────────────────────────────────────────────────────────────
# 4. PLR Evaluation  (collect rollout data + score for a level)
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_level(
    model: PPO,
    config: dict,
    level_idx: int,
    n_episodes: int = 5,
    gamma: float = 0.99,
    strategy: str = "value_l1",
) -> LevelRollout:
    """
    Run the agent on one level config for n_episodes.
    Collects observations, rewards, value predictions, and action logits
    for PLR scoring.
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
            # Get value prediction
            value = model.policy.predict_values(obs_t)
            val = float(value.cpu().squeeze())

            # Get action distribution for logits
            dist = model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy().squeeze()

            # Step environment
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
# 5. PPO + PLR Training Loop
# ──────────────────────────────────────────────────────────────────────────────
def train_ppo_plr(
    total_timesteps: int = 500_000,
    steps_per_level: int = 4_096,
    eval_episodes: int = 5,
    # PLR parameters (paper defaults for value_l1 + rank)
    plr_strategy: str = "value_l1",
    plr_rho: float = 0.5,
    plr_nu: float = 0.5,
    plr_score_transform: str = "rank",
    plr_temperature: float = 0.1,
    plr_staleness_coef: float = 0.1,
    plr_alpha: float = 1.0,
    # PPO hyperparameters (tuned for multi-level learning)
    learning_rate: float = 2e-4,    # Lower LR for stability across envs
    n_steps: int = 1024,            # More steps per update for stable gradients
    batch_size: int = 128,          # Larger batches for better generalization
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
    Train PPO with Prioritized Level Replay.

    The training alternates:
    1. Sample a level from PLR
    2. Train PPO on that level for `steps_per_level` steps
    3. Evaluate the level to compute a PLR score
    4. Update the sampler and repeat
    """
    device = get_device()

    # ── header ──
    print(f"\n{'=' * 72}")
    print("  PPO + PLR  —  Prioritized Level Replay")
    print(f"{'=' * 72}")
    print(f"  Device .............. {device}")
    print(f"  Parallel envs ....... {NUM_ENVS}")
    print(f"  Total timesteps ..... {total_timesteps:,}")
    print(f"  Steps per level ..... {steps_per_level:,}")
    print(f"  Eval episodes ....... {eval_episodes}")
    print(f"  PLR strategy ........ {plr_strategy}")
    print(f"  PLR rho ............. {plr_rho}")
    print(f"  PLR nu .............. {plr_nu}")
    print(f"  PLR score transform . {plr_score_transform}")
    print(f"  PLR temperature ..... {plr_temperature}")
    print(f"  PLR staleness coef .. {plr_staleness_coef}")
    print(f"  Learning rate ....... {learning_rate}")
    print(f"  PPO n_steps ......... {n_steps}")
    print(f"  PPO batch_size ...... {batch_size}")
    print(f"  PPO n_epochs ........ {n_epochs}")
    print(f"  Gamma ............... {gamma}")
    print(f"  GAE lambda .......... {gae_lambda}")
    print(f"  Entropy coef ........ {ent_coef}")
    print(f"{'=' * 72}\n")

    # ── configs & PLR sampler ──
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

    model = PPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),  # Larger network for multi-level capacity
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
        "configs": [],
        "rewards": [],
        "crash_rates": [],
        "ep_lengths": [],
        "timesteps": [],
        "fps": [],
        "plr_scores": [],
    }
    stats = RollingStats(window=20)
    timesteps_done = 0
    iteration = 0

    pbar = tqdm(total=total_timesteps, desc="PPO+PLR", unit="step",
                dynamic_ncols=True)
    wall_start = time.perf_counter()

    # ── main loop ──
    while timesteps_done < total_timesteps:
        iteration += 1
        steps_this = min(steps_per_level, total_timesteps - timesteps_done)

        # Train on the current level
        model.learn(
            total_timesteps=steps_this,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        timesteps_done += steps_this
        pbar.update(steps_this)

        # ── PLR scoring via evaluation ──
        rollout = evaluate_level(
            model, config, level_idx,
            n_episodes=eval_episodes,
            gamma=gamma,
            strategy=plr_strategy,
        )

        # Compute PLR score from rollout
        plr_score = rollout.compute_score(plr_strategy, gamma=gamma)
        sampler.update_score(level_idx, plr_score)
        sampler.update_staleness(level_idx)

        # Update stats
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
            cfg = env_configs[level_idx]
            plr_stats = sampler.get_stats()
            crash_str = (
                f"{rollout.crash_count}/{eval_episodes} crashed"
                if rollout.crash_count > 0
                else "no crashes"
            )
            tqdm.write(
                f"\n  ┌─ Iter {iteration:>3d}  │  "
                f"{timesteps_done:,}/{total_timesteps:,} steps  │  "
                f"{elapsed:.0f}s elapsed"
            )
            tqdm.write(
                f"  │  Level: {describe_config(cfg)}  "
                f"(idx={level_idx}, "
                f"{plr_stats['plr/seen']}/{n_configs} seen)"
            )
            tqdm.write(
                f"  │  Reward: {rollout.mean_reward:>7.2f}  │  "
                f"Avg({stats.window}): {stats.avg_reward:>7.2f}  │  "
                f"Best: {stats.best_reward:.2f}  │  "
                f"Trend: {stats.reward_trend}"
            )
            tqdm.write(
                f"  │  Survival: {stats.survival_rate:>5.1f}%  │  "
                f"{crash_str}  │  "
                f"Avg length: {rollout.mean_ep_length:.0f} steps"
            )
            tqdm.write(
                f"  │  PLR score: {plr_score:.4f}  │  "
                f"Mean PLR: {plr_stats['plr/mean_score']:.4f}"
            )
            tqdm.write(f"  └─ FPS: {fps:.0f}")

        pbar.set_postfix_str(
            f"R={stats.avg_reward:.1f}  "
            f"Surv={stats.survival_rate:.0f}%  "
            f"{fps:.0f} FPS"
        )

        # ── history ──
        history["configs"].append(int(level_idx))
        history["rewards"].append(float(rollout.mean_reward))
        history["crash_rates"].append(float(stats.crash_rate))
        history["ep_lengths"].append(float(rollout.mean_ep_length))
        history["timesteps"].append(int(timesteps_done))
        history["fps"].append(float(fps))
        history["plr_scores"].append(float(plr_score))

        # ── sample next level, swap env ──
        level_idx = sampler.sample()
        config = env_configs[level_idx]
        vec_env.close()
        vec_env = make_vec_env(config, NUM_ENVS)
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
        "num_configs": n_configs,
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
    print(f"{'=' * 72}")
    print(f"  AGENT PERFORMANCE")
    print(f"  {'─' * 40}")
    print(f"  Avg reward (all) .... {stats.avg_reward_all:.2f}")
    print(f"  Avg reward (last {stats.window})  {stats.avg_reward:.2f}")
    print(f"  Best reward ......... {stats.best_reward:.2f}")
    print(f"  Worst reward ........ {stats.worst_reward:.2f}")
    print(f"  Survival rate ....... {stats.survival_rate:.1f}%")
    print(f"  Avg episode length .. {stats.avg_ep_length:.0f} steps")
    print(f"  Reward trend ........ {stats.reward_trend}")
    print(f"{'=' * 72}")
    print(f"  CURRICULUM (PLR)")
    print(f"  {'─' * 40}")
    print(f"  Strategy ............ {plr_strategy}")
    print(f"  Levels seen ......... {plr_final['plr/seen']}/{n_configs}")
    print(f"  Mean PLR score ...... {plr_final['plr/mean_score']:.4f}")
    print(f"  Max PLR score ....... {plr_final['plr/max_score']:.4f}")
    print(f"{'=' * 72}")
    print(f"  Saved to ............ {save_path}")
    print(f"{'=' * 72}\n")

    return model, sampler, history


# ──────────────────────────────────────────────────────────────────────────────
# 6. Test evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_on_test_configs(model, test_configs, n_episodes=10):
    print(f"\n{'=' * 72}")
    print("  EVALUATING ON TEST CONFIGURATIONS")
    print(f"{'=' * 72}")
    print(f"  Episodes per config . {n_episodes}")
    print(f"  Test configs ........ {len(test_configs)}")
    print(f"{'=' * 72}\n")

    results = []
    all_rewards: List[float] = []
    all_crashes = 0
    all_lengths: List[int] = []
    total_episodes = 0

    for i, config in enumerate(test_configs):
        env = make_single_env(config)
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
            "config_desc": describe_config(config),
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
            f"  ┌─ Config {i+1}/{len(test_configs)}  │  "
            f"{describe_config(config)}"
        )
        print(
            f"  │  Reward: {avg:>7.2f} ± {std:.2f}  │  "
            f"Median: {med:.2f}  │  "
            f"Best: {best_r:.2f}  │  Worst: {worst_r:.2f}"
        )
        print(
            f"  └─ Survival: {surv:>5.1f}%  │  "
            f"{crash_str}  │  "
            f"Avg length: {avg_len:.0f} steps\n"
        )

    overall_avg = float(np.mean(all_rewards))
    overall_std = float(np.std(all_rewards))
    overall_cr = 100.0 * all_crashes / max(total_episodes, 1)
    overall_surv = 100.0 - overall_cr
    overall_len = float(np.mean(all_lengths))

    print(f"{'=' * 72}")
    print("  EVALUATION SUMMARY")
    print(f"  {'─' * 40}")
    print(f"  Total episodes ...... {total_episodes}")
    print(f"  Avg reward .......... {overall_avg:.2f} ± {overall_std:.2f}")
    print(f"  Survival rate ....... {overall_surv:.1f}%  "
          f"({all_crashes}/{total_episodes} crashed)")
    print(f"  Avg episode length .. {overall_len:.0f} steps")
    print(f"{'=' * 72}\n")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 7. Watch agent
# ──────────────────────────────────────────────────────────────────────────────
def watch_agent(model, configs, n_episodes=3):
    if isinstance(configs, dict):
        configs = [configs]

    print(f"\n{'=' * 60}")
    print("  WATCHING AGENT  (Ctrl-C to stop)")
    print(f"  {len(configs)} config(s)  ×  {n_episodes} episodes each")
    print(f"{'=' * 60}\n")

    try:
        for ci, config in enumerate(configs):
            desc = describe_config(config)
            print(f"  ── Config {ci+1}/{len(configs)}: {desc} ──")
            env = make_single_env(config, render=True)
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
# 8. Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # ====== FLAGS ===========================================================
    TRAIN           = True
    SKIP_EVALUATION = False

    # Training budget — 2M steps needed for multi-level mastery (~2-3 hours)
    TOTAL_TIMESTEPS  = 2_000_000
    STEPS_PER_LEVEL  = 40_000     # Each level gets meaningful training time
    EVAL_EPISODES    = 3          # Faster evaluation

    # PLR — paper-recommended for value_l1 + rank
    PLR_STRATEGY         = "value_l1"
    PLR_RHO              = 0.5
    PLR_NU               = 0.5
    PLR_SCORE_TRANSFORM  = "rank"
    PLR_TEMPERATURE      = 0.1
    PLR_STALENESS_COEF   = 0.1
    PLR_ALPHA            = 1.0    # no EMA, use latest score

    SAVE_DIR = "highway_ppo_plr/v3"
    # ========================================================================

    if TRAIN:
        model, sampler, history = train_ppo_plr(
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
            save_dir=SAVE_DIR,
            verbose=1,
        )
    else:
        device = get_device()
        model = PPO.load(f"{SAVE_DIR}/model", device=device)

    # ── evaluation ──
    if not SKIP_EVALUATION:
        test_cfgs = generate_test_configs()
        results = evaluate_on_test_configs(model, test_cfgs, n_episodes=10)
        save_path = Path(SAVE_DIR)
        with open(save_path / "test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print("Skipping evaluation, jumping to visualization...")

    # ── watch ──
    watch_cfgs = generate_test_configs()
    watch_agent(model, watch_cfgs, n_episodes=3)


if __name__ == "__main__":
    main()
