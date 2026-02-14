#!/usr/bin/env python3
"""
train.py — SB3 DQN + PLR training loop for highway-env.

Uses Stable-Baselines3 DQN as the core learning algorithm with
Prioritised Level Replay (PLR) for automatic curriculum learning.
Checkpoints are saved in the standard SB3 .zip format for full
compatibility with ``stable_baselines3.DQN.load()``.

Key change from the legacy custom DQN: the replay buffer, Q-networks,
gradient updates, and checkpoint format are all managed by SB3.  The PLR
curriculum machinery (level_sampler, RolloutStorage) remains unchanged.

Usage
-----
    python -m plr_dqn.train                 # defaults (SB3 DQN + PLR)
    python -m plr_dqn.train --total_steps 5_000_000 --lr 3e-4
    python -m plr_dqn.train --no_plr        # vanilla DQN baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ── local imports ────────────────────────────────────────────────────────────
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from stable_baselines3 import DQN as SB3_DQN
from stable_baselines3.common.utils import polyak_update
import stable_baselines3.common.logger as sb3_logger

from plr_dqn.level_sampler import LevelSampler
from plr_dqn.storage import RolloutStorage
from plr_dqn.highway_levels import (
    HighwayVecEnv, SubprocVecEnv, OBS_SHAPE, NUM_ACTIONS, EVAL_SEEDS,
    make_flat_env,
)
from plr_dqn.plr_configs import seed_to_factors, describe_level


# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    # environment
    num_steps=1024,         # rollout length per level
    num_processes=4,        # parallel envs
    gamma=0.99,
    gae_lambda=0.95,        # for PLR return computation only
    # DQN (SB3)
    lr=5e-4,
    batch_size=128,
    gradient_steps=256,     # DQN updates per rollout
    replay_buffer_size=100_000,
    learning_starts=1024,   # env steps before first DQN update
    target_update_interval=10,  # hard-update target every N PLR updates
    max_grad_norm=10.0,
    epsilon_start=1.0,
    epsilon_end=0.1,        # Higher minimum exploration
    epsilon_fraction=0.5,   # Decay over 50% of training (1M steps)
    net_arch=[256, 256],    # hidden layer sizes for SB3 MlpPolicy
    # PLR
    seed_buffer_size=200,
    replay_prob=0.95,
    rho=0.5,
    staleness_coef=0.3,
    staleness_transform="power",
    staleness_temperature=1.0,
    strategy="one_step_td_error",
    score_transform="rank",
    temperature=0.1,
    replay_schedule="fixed",
    seed_buffer_priority="replay_support",
    # training
    total_steps=2_000_000,
    eval_interval=50,       # evaluate every N updates
    log_interval=5,         # print every N updates
    save_interval=100,      # checkpoint every N updates
    eval_episodes=3,
    output_dir="plr/runs",
    run_name=None,
    device="cpu",
    seed=1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: SB3_DQN,
    eval_seeds: list[int],
    num_episodes: int,
    device: str,
) -> dict:
    """Evaluate the greedy policy on held-out test seeds."""
    results = {}
    all_returns = []
    action_counts = np.zeros(NUM_ACTIONS)

    for seed in eval_seeds:
        env = HighwayVecEnv(seed=seed, device=device)
        seed_returns = []
        seed_lengths = []

        for _ep in range(num_episodes):
            obs = env.reset()
            ep_return = 0.0
            ep_len = 0

            while True:
                obs_np = obs.cpu().numpy()
                action_np, _ = model.predict(obs_np, deterministic=True)
                action_int = int(np.asarray(action_np).flatten()[0])
                action_counts[action_int] += 1
                obs, reward, mask, bad_mask, info = env.step(action_int)
                ep_return += reward.item()
                ep_len += 1
                if info["done"]:
                    break

            seed_returns.append(ep_return)
            seed_lengths.append(ep_len)

        env.close()
        results[seed] = {
            "mean_return": float(np.mean(seed_returns)),
            "std_return": float(np.std(seed_returns)),
            "mean_length": float(np.mean(seed_lengths)),
        }
        all_returns.extend(seed_returns)

    total_acts = max(action_counts.sum(), 1)
    pct = action_counts / total_acts * 100
    actions = ["L_LEFT", "IDLE", "L_RIGHT", "FASTER", "SLOWER"]
    act_str = "  ".join(f"{a}={p:.0f}%" for a, p in zip(actions, pct))

    results["aggregate"] = {
        "mean_return": float(np.mean(all_returns)),
        "std_return": float(np.std(all_returns)),
        "num_seeds": len(eval_seeds),
        "num_episodes": len(all_returns),
        "action_dist": act_str,
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    # ── output dir ───────────────────────────────────────────────────────
    run_name = args.run_name or f"dqn_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = (Path(args.output_dir) / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── SB3 DQN model ───────────────────────────────────────────────────
    dummy_env = make_flat_env(seed=0)

    if args.resume:
        print(f"\n📂 Resuming from checkpoint: {args.resume}")
        model = SB3_DQN.load(
            args.resume,
            env=dummy_env,
            device=args.device,
        )
        resume_steps = args.resume_steps or 0
        print(f"   Continuing from step {resume_steps:,}")
    else:
        model = SB3_DQN(
        "MlpPolicy",
        dummy_env,
        learning_rate=args.lr,
        buffer_size=args.replay_buffer_size,
        learning_starts=0,  # we manage this in our loop
        batch_size=args.batch_size,
        gamma=args.gamma,
        target_update_interval=1_000_000,  # we manage target updates ourselves
        exploration_initial_eps=args.epsilon_start,
        exploration_final_eps=args.epsilon_end,
        exploration_fraction=args.epsilon_fraction,
        max_grad_norm=args.max_grad_norm,
        gradient_steps=args.gradient_steps,
        tau=1.0,  # hard update
        train_freq=(args.num_steps, "step"),
        policy_kwargs=dict(net_arch=args.net_arch),
        verbose=0,
        device=args.device,
        seed=args.seed,
        )
        resume_steps = 0
    
    dummy_env.close()

    # Set up SB3 logger (quiet — our loop handles console printing)
    model.set_logger(sb3_logger.configure(str(run_dir / "sb3_logs"), ["json"]))

    num_params = sum(
        p.numel() for p in model.q_net.parameters() if p.requires_grad
    )
    print(f"SB3 DQN (MlpPolicy {args.net_arch}): {num_params:,} params")

    # ── rollout storage (used ONLY for PLR level scoring) ────────────────
    import gymnasium as gym
    action_space = gym.spaces.Discrete(NUM_ACTIONS)
    rollouts = RolloutStorage(
        args.num_steps, args.num_processes, OBS_SHAPE,
        action_space, device=str(device),
    ).to(str(device))

    # ── PLR sampler ──────────────────────────────────────────────────────
    use_plr = not args.no_plr
    level_sampler = None

    if use_plr:
        level_sampler = LevelSampler(
            obs_space=None,
            action_space=action_space,
            num_actors=args.num_processes,
            strategy=args.strategy,
            replay_schedule=args.replay_schedule,
            score_transform=args.score_transform,
            temperature=args.temperature,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature,
            replay_prob=args.replay_prob,
            rho=args.rho,
            seed_buffer_size=args.seed_buffer_size,
            seed_buffer_priority=args.seed_buffer_priority,
        )
        print(f"PLR: strategy={args.strategy}, buffer={args.seed_buffer_size}, "
              f"replay_prob={args.replay_prob}, rho={args.rho}")
    else:
        print("PLR disabled — vanilla DQN with random levels.")

    # ── env ──────────────────────────────────────────────────────────────
    N = args.num_processes
    use_subproc = N > 1

    if use_subproc:
        env = SubprocVecEnv(
            num_envs=N, initial_seeds=list(range(N)), device=str(device)
        )
        print(f"Using {N} parallel environments (SubprocVecEnv)")
    else:
        env = HighwayVecEnv(seed=0, device=str(device))
        print("Using 1 environment (HighwayVecEnv)")

    # ── metrics ──────────────────────────────────────────────────────────
    episode_returns = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    current_ep_returns = np.zeros(N)
    current_ep_lengths = np.zeros(N, dtype=int)
    num_episodes = 0
    replay_decisions = deque(maxlen=100)
    history = {"updates": [], "eval_results": []}

    # ── main loop ────────────────────────────────────────────────────────
    num_updates = args.total_steps // (args.num_steps * N)
    total_env_steps = resume_steps
    start_update = (resume_steps // (args.num_steps * N)) + 1
    replay_started = False  # Track when replay begins (always starts fresh)
    replay_start_step = 0   # Step when replay started (always resets)
    start_time = time.time()
    
    # Notify user if resuming (epsilon will restart from beginning)
    if args.resume and resume_steps > 0:
        print(f"\n⏩ Skipping to update {start_update} (step {resume_steps:,})")
        print(f"    Epsilon will restart from {args.epsilon_start}")

    remaining_updates = num_updates - start_update + 1
    print(f"\nTraining for {remaining_updates} updates "
          f"(target: {args.total_steps:,} steps, {args.num_steps} steps/update, "
          f"SB3 DQN with ε-greedy exploration)")
    print("=" * 72)

    for update in range(start_update, num_updates + 1):
        # ── (1) PLR: decide replay or explore ────────────────────────────
        is_replay = False
        if use_plr and level_sampler is not None:
            is_replay = level_sampler.sample_replay_decision()
            
            # Track when replay first starts
            if is_replay and not replay_started:
                replay_started = True
                replay_start_step = total_env_steps
                print(f"\n🔄 Replay started at step {total_env_steps}, epsilon decay begins now\n")

        # ── Update SB3 exploration rate (only decay after replay starts) ─
        if not replay_started:
            # Keep epsilon at start value until replay begins
            epsilon = args.epsilon_start
        else:
            # Decay epsilon based on steps since replay started
            steps_since_replay = total_env_steps - replay_start_step
            decay_steps = int(args.total_steps * args.epsilon_fraction)
            if steps_since_replay >= decay_steps:
                epsilon = args.epsilon_end
            else:
                frac = steps_since_replay / max(decay_steps, 1)
                epsilon = args.epsilon_start + frac * (args.epsilon_end - args.epsilon_start)
        
        model.num_timesteps = total_env_steps
        model.exploration_rate = epsilon

        if is_replay:
            level_seed = level_sampler.sample_replay_level()
            level_seeds = [level_seed] * N
        elif use_plr and level_sampler is not None:
            level_seeds = [level_sampler.sample_unseen_level() for _ in range(N)]
            level_sampler.observe_external_unseen_sample(level_seeds)
        else:
            level_seeds = [int(np.random.randint(0, 100_000)) for _ in range(N)]

        replay_decisions.append(int(is_replay))

        # ── (2) set level ────────────────────────────────────────────────
        current_level_seeds = level_seeds
        if use_subproc:
            env.set_levels(level_seeds)
            obs = env.reset()
        else:
            env.set_level(level_seeds[0])
            obs = env.reset()

        if is_replay:
            for s in sorted(set(level_seeds)):
                f = seed_to_factors(s)
                print(f"    REPLAY seed={s} → L={f['lanes_count']} "
                      f"V={f['vehicles_count']} D={f['vehicles_density']:.2f} "
                      f"P={f['politeness']:.3f}")

        # Fill PLR rollout buffer slot 0
        rollouts.obs[0].copy_(obs)
        current_ep_returns[:] = 0.0
        current_ep_lengths[:] = 0

        # ── (3) collect rollout (ε-greedy via SB3) ───────────────────────
        for step in range(args.num_steps):
            step_obs = obs
            if step_obs.dim() == 1:
                step_obs = step_obs.unsqueeze(0)

            # Action selection via SB3 (uses model.exploration_rate)
            obs_np = step_obs.cpu().numpy()
            action_np, _ = model.predict(obs_np, deterministic=False)
            action_np = np.asarray(action_np).flatten()  # ensure (N,)
            action = torch.tensor(
                action_np, device=device, dtype=torch.long
            ).reshape(N, 1)

            # Q-values for PLR scoring (treat as logits)
            with torch.no_grad():
                q_values = model.q_net(step_obs)                     # (N, A)
                value = q_values.max(dim=-1, keepdim=True)[0]        # (N, 1)
                action_log_prob = torch.log_softmax(
                    q_values, dim=-1
                ).gather(1, action)

            # Step env
            if use_subproc:
                next_obs, reward, mask, bad_mask, infos = env.step(action)
            else:
                next_obs, reward, mask, bad_mask, info = env.step(action)
                infos = [info]

            # ── Store in SB3 replay buffer ───────────────────────────
            done_flags = (1.0 - mask).cpu().numpy()
            next_obs_np = next_obs.cpu().numpy()
            reward_np = reward.cpu().numpy()

            for i in range(N):
                model.replay_buffer.add(
                    obs_np[i],                                      # obs
                    next_obs_np[i],                                 # next_obs
                    np.array([action_np[i]]),                       # action
                    np.array([reward_np[i].flatten()[0]]),           # reward
                    np.array([done_flags[i].flatten()[0] > 0.5]),   # done
                    [infos[i] if isinstance(infos, list)
                     else infos],
                )

            # ── Store in PLR rollout buffer ──────────────────────────
            seed_tensor = torch.tensor(
                [[s] for s in current_level_seeds],
                dtype=torch.int, device=str(device),
            )
            rollouts.insert(
                obs=next_obs,
                actions=action,
                action_log_probs=action_log_prob,
                action_log_dist=q_values,       # raw Q-values as "logits"
                value_preds=value,
                rewards=reward,
                masks=mask,
                bad_masks=bad_mask,
                level_seeds=seed_tensor,
            )

            # ── Episode tracking ─────────────────────────────────────
            for i, inf in enumerate(infos):
                current_ep_returns[i] += reward[i].item()
                current_ep_lengths[i] += 1
                if inf.get("done", False):
                    episode_returns.append(current_ep_returns[i])
                    episode_lengths.append(current_ep_lengths[i])
                    num_episodes += 1
                    current_ep_returns[i] = 0.0
                    current_ep_lengths[i] = 0

            obs = next_obs

        total_env_steps += args.num_steps * N

        # ── (4) compute returns for PLR scoring ──────────────────────────
        with torch.no_grad():
            last_obs = rollouts.obs[-1]
            if last_obs.dim() == 1:
                last_obs = last_obs.unsqueeze(0)
            next_q = model.q_net(last_obs)
            next_value = next_q.max(dim=-1, keepdim=True)[0]

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # ── (5) PLR level scoring ────────────────────────────────────────
        if use_plr and level_sampler is not None:
            level_sampler.update_with_rollouts(rollouts)

        # ── (6) SB3 DQN gradient updates ────────────────────────────────
        dqn_loss = 0.0
        n_grad = 0

        if model.replay_buffer.size() >= args.learning_starts:
            model.num_timesteps = total_env_steps
            model._current_progress_remaining = max(
                1.0 - total_env_steps / args.total_steps, 0.0
            )
            model.train(
                gradient_steps=args.gradient_steps,
                batch_size=args.batch_size,
            )
            n_grad = args.gradient_steps
            dqn_loss = model.logger.name_to_value.get("train/loss", 0.0)

        # ── (7) target network update ────────────────────────────────────
        if update % args.target_update_interval == 0:
            polyak_update(
                model.q_net.parameters(),
                model.q_net_target.parameters(),
                1.0,  # tau=1.0 → hard update
            )

        # ── (8) PLR after-update ─────────────────────────────────────────
        if use_plr and level_sampler is not None:
            level_sampler.after_update()

        rollouts.after_update()

        # ── logging ──────────────────────────────────────────────────────
        if update % args.log_interval == 0 and len(episode_returns) > 0:
            elapsed = time.time() - start_time
            fps = total_env_steps / elapsed
            mean_return = np.mean(episode_returns)
            mean_length = np.mean(episode_lengths)
            replay_ratio = (
                np.mean(replay_decisions) if replay_decisions else 0.0
            )

            plr_info = ""
            if use_plr and level_sampler is not None:
                plr_info = (
                    f"  PLR: buf={level_sampler.working_seed_buffer_size}"
                    f"/{args.seed_buffer_size} replay={replay_ratio:.2f}"
                )

            learning = "learning" if n_grad > 0 else "filling"
            print(
                f"Upd {update:5d} | "
                f"Steps {total_env_steps:>9,} | "
                f"FPS {fps:5.0f} | "
                f"ε={epsilon:.3f} | "
                f"R̄={mean_return:7.2f} | "
                f"L̄={mean_length:5.1f} | "
                f"loss={dqn_loss:.4f} | "
                f"{learning}"
                f"{plr_info}"
            )

            history["updates"].append({
                "update": update,
                "total_steps": total_env_steps,
                "mean_return": float(mean_return),
                "mean_length": float(mean_length),
                "dqn_loss": dqn_loss,
                "epsilon": epsilon,
                "replay_ratio": float(replay_ratio),
                "fps": fps,
                "is_replay": is_replay,
                "level_seeds": level_seeds,
                "replay_buffer_size": int(model.replay_buffer.size()),
            })

        # ── evaluation ───────────────────────────────────────────────────
        if update % args.eval_interval == 0:
            eval_results = evaluate(
                model, EVAL_SEEDS,
                num_episodes=args.eval_episodes,
                device=str(device),
            )
            agg = eval_results["aggregate"]
            print(
                f"  ▸ EVAL ({len(EVAL_SEEDS)} seeds × {args.eval_episodes} eps): "
                f"R̄={agg['mean_return']:.2f} ± {agg['std_return']:.2f}  "
                f"[{agg['action_dist']}]"
            )
            history["eval_results"].append({
                "update": update,
                "total_steps": total_env_steps,
                **{str(k): v for k, v in eval_results.items()},
            })

        # ── checkpointing (SB3 .zip format) ──────────────────────────────
        if update % args.save_interval == 0:
            ckpt_path = run_dir / f"checkpoint_{update}"
            model.save(str(ckpt_path))
            print(f"  💾 SB3 checkpoint saved: {ckpt_path}.zip")

            with open(run_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2, default=str)

    # ── final save ───────────────────────────────────────────────────────
    env.close()

    final_path = run_dir / "model_final"
    model.save(str(final_path))
    print(f"  💾 Final SB3 model saved: {final_path}.zip")

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)

    # ── final evaluation ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("Final evaluation")
    eval_results = evaluate(
        model, EVAL_SEEDS,
        num_episodes=5,
        device=str(device),
    )
    agg = eval_results["aggregate"]
    print(f"  Mean return: {agg['mean_return']:.2f} ± {agg['std_return']:.2f}")
    print(f"  Action dist: {agg['action_dist']}")
    print(f"  {agg['num_seeds']} seeds, {agg['num_episodes']} episodes")

    for seed in EVAL_SEEDS:
        r = eval_results[seed]
        print(f"  Seed {seed}: R̄={r['mean_return']:.2f} ± {r['std_return']:.2f}")

    with open(run_dir / "final_eval.json", "w") as f:
        json.dump(eval_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed/3600:.1f}h ({total_env_steps:,} steps)")
    print(f"Artifacts: {run_dir}")
    return run_dir


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="SB3 DQN + PLR for highway-env",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Environment
    parser.add_argument("--num_steps", type=int, default=DEFAULTS["num_steps"])
    parser.add_argument("--num_processes", type=int, default=DEFAULTS["num_processes"])
    parser.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    parser.add_argument("--gae_lambda", type=float, default=DEFAULTS["gae_lambda"])

    # DQN (SB3)
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--gradient_steps", type=int, default=DEFAULTS["gradient_steps"])
    parser.add_argument("--replay_buffer_size", type=int,
                        default=DEFAULTS["replay_buffer_size"])
    parser.add_argument("--learning_starts", type=int,
                        default=DEFAULTS["learning_starts"])
    parser.add_argument("--target_update_interval", type=int,
                        default=DEFAULTS["target_update_interval"])
    parser.add_argument("--max_grad_norm", type=float,
                        default=DEFAULTS["max_grad_norm"])
    parser.add_argument("--epsilon_start", type=float,
                        default=DEFAULTS["epsilon_start"])
    parser.add_argument("--epsilon_end", type=float,
                        default=DEFAULTS["epsilon_end"])
    parser.add_argument("--epsilon_fraction", type=float,
                        default=DEFAULTS["epsilon_fraction"])
    parser.add_argument("--net_arch", type=int, nargs="+",
                        default=DEFAULTS["net_arch"],
                        help="Hidden layer sizes for SB3 MlpPolicy")

    # PLR
    parser.add_argument("--no_plr", action="store_true", help="Disable PLR")
    parser.add_argument("--seed_buffer_size", type=int,
                        default=DEFAULTS["seed_buffer_size"])
    parser.add_argument("--replay_prob", type=float,
                        default=DEFAULTS["replay_prob"])
    parser.add_argument("--rho", type=float, default=DEFAULTS["rho"])
    parser.add_argument("--staleness_coef", type=float,
                        default=DEFAULTS["staleness_coef"])
    parser.add_argument("--staleness_transform", type=str,
                        default=DEFAULTS["staleness_transform"])
    parser.add_argument("--staleness_temperature", type=float,
                        default=DEFAULTS["staleness_temperature"])
    parser.add_argument("--strategy", type=str,
                        default=DEFAULTS["strategy"],
                        choices=["value_l1", "gae", "positive_value_loss",
                                 "signed_value_loss", "one_step_td_error",
                                 "policy_entropy", "least_confidence",
                                 "min_margin", "random", "off"])
    parser.add_argument("--score_transform", type=str,
                        default=DEFAULTS["score_transform"])
    parser.add_argument("--temperature", type=float,
                        default=DEFAULTS["temperature"])
    parser.add_argument("--replay_schedule", type=str,
                        default=DEFAULTS["replay_schedule"])
    parser.add_argument("--seed_buffer_priority", type=str,
                        default=DEFAULTS["seed_buffer_priority"])

    # Training
    parser.add_argument("--total_steps", type=int,
                        default=DEFAULTS["total_steps"])
    parser.add_argument("--eval_interval", type=int,
                        default=DEFAULTS["eval_interval"])
    parser.add_argument("--log_interval", type=int,
                        default=DEFAULTS["log_interval"])
    parser.add_argument("--save_interval", type=int,
                        default=DEFAULTS["save_interval"])
    parser.add_argument("--eval_episodes", type=int,
                        default=DEFAULTS["eval_episodes"])
    parser.add_argument("--output_dir", type=str,
                        default=DEFAULTS["output_dir"])
    parser.add_argument("--run_name", type=str, default=DEFAULTS["run_name"])
    parser.add_argument("--device", type=str, default=DEFAULTS["device"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (path to .zip file)")
    parser.add_argument("--resume_steps", type=int, default=None,
                        help="Steps completed in checkpoint (for progress tracking)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
