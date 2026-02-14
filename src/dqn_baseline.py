"""
DQN Baseline for Highway-Env
=============================
Trains a DQN model on highway-fast-v0 with parameters aligned
to ACCEL FIXED_PARAMS for a fair comparison.

The saved model (best_model.zip) can be used as pre-trained for ACCEL:
  python src/dqn_accel.py --pretrained highway_dqn/best_model.zip
"""

import gymnasium
import highway_env
import torch
import numpy as np
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json


TRAIN = True
TOTAL_TIMESTEPS = 1_000_000
MODEL_NAME = f'dqn_baseline_{1}M' 
SAVE_DIR = Path("highway_dqn")
NUM_ENVS = 4                # Parallel envs (SubprocVecEnv)
RENDER_DELAY_MS = 200       # Delay between frames during rendering (ms)


print("model name:", MODEL_NAME)
# =============================================================================
#  ENV CONFIG — Aligned to ACCEL FIXED_PARAMS
# =============================================================================
# These parameters MUST be identical to ACCELGenerator.FIXED_PARAMS
# to ensure baseline and ACCEL are comparable.

ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 12,
    "vehicles_density": 0.8,
    "duration": 60,                    # 60 seconds per episode
    "policy_frequency": 2,             # 2 decisions/sec
    "collision_reward": -10.0,          # Strong crash penalty
    "high_speed_reward": 0.3,          # Speed incentive: up to +0.3/step at 30 m/s
    "right_lane_reward": 0.0,          # No right-lane bonus
    "lane_change_reward": 0,           # Neutral: lane changes not penalized
    "reward_speed_range": [20, 30],
    "normalize_reward": False,         # Raw rewards: crash = -10.0
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

# ACCEL-compatible observation — used for training AND evaluation.
# The model trained with this observation expects shape (7,5),
# so the eval env MUST use it too.
ACCEL_OBSERVATION = {
    "type": "Kinematics",
    "vehicles_count": 7,           # Ego + 6 others
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,             # Positions relative to ego
    "normalize": True,             # Normalized to [-1, 1]
    "see_behind": True,            # Rear-view: sees vehicles behind
    "order": "sorted",             # Sorted by distance
}

# Reward with normalize_reward=False (aligned to ACCEL FIXED_PARAMS):
#   Per step: collision_reward * crashed + high_speed_reward * speed_frac
#   - Normal driving 25 m/s:     +0.15/step  (speed_frac ~ 0.5)
#   - Perfect driving 30 m/s:    +0.3/step   (speed_frac = 1.0)
#   - Collision:                 -10.0 + episode TERMINATES
#
# Con gamma=0.9, episodio 120 step (60s × 2 Hz):
#   - Return max discounted ~ 3.0  (sum 0.3 * 0.9^t)
#   - Crash = -10.0 instant -> dominates the signal
#   - Penalty/return ratio ~ 333% -> risk of unstable Q if exploration too high


# =============================================================================
#  CALLBACKS
# =============================================================================

class TqdmCallback(BaseCallback):
    """Progress bar."""
    def __init__(self):
        super().__init__()
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'], unit="step")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()


class BestModelCallback(BaseCallback):
    """
    Saves the model when mean reward (over a window) improves.
    best_model.zip always contains the best model seen so far.
    """

    def __init__(self, save_path: str, check_freq: int = 500, window: int = 30, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.check_freq = check_freq
        self.window = window
        self.best_mean_reward = -np.inf
        self.best_mean_length = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.saves_count = 0

    def _on_step(self) -> bool:
        # Collect episode stats from Monitor wrapper
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

        # Periodic check
        if len(self.episode_rewards) >= self.window and self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-self.window:])
            mean_length = np.mean(self.episode_lengths[-self.window:])

            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                self.best_mean_length = mean_length
                self.saves_count += 1

                # Save model
                self.model.save(str(self.save_path / "best_model"))

                # Save info
                info_data = {
                    "step": self.n_calls,
                    "mean_reward": float(mean_reward),
                    "mean_length": float(mean_length),
                    "improvement": float(improvement),
                    "saves_count": self.saves_count,
                    "total_episodes": len(self.episode_rewards),
                    "timestamp": datetime.now().isoformat(),
                }
                with open(self.save_path / "best_model_info.json", "w") as f:
                    json.dump(info_data, f, indent=2)

                if self.verbose:
                    print(f"\n  ★ BEST MODEL saved! Reward: {mean_reward:.2f} (+{improvement:.2f}) "
                          f"| Len: {mean_length:.0f}/120 | Step: {self.n_calls:,}")

        return True


# =============================================================================
#  ENV FACTORY — Required for SubprocVecEnv (each env in a process)
# =============================================================================

def make_env(rank: int, seed: int = 42, config: dict = None):
    """Create a single env wrapped with Monitor (required for episode stats)."""
    env_config = config or ENV_CONFIG
    def _init():
        env = gymnasium.make("highway-fast-v0", config=env_config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# =============================================================================
#  MAIN — __name__ guard required for SubprocVecEnv on Windows (spawn)
# =============================================================================

def main():
    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    # Training config: add ACCEL-compatible observation if TRAIN=True
    # This makes the pre-trained model compatible with dqn_accel.py
    if TRAIN:
        train_config = {**ENV_CONFIG, "observation": ACCEL_OBSERVATION}
        print("  Observation:   ACCEL-compatible (7 vehicles, see_behind=True)")
    else:
        train_config = ENV_CONFIG
        print("  Observation:   default (model already trained)")

    # Training env — SubprocVecEnv: each env in a separate process
    print(f"\nCreating {NUM_ENVS} parallel environments (SubprocVecEnv)...")
    env = SubprocVecEnv([make_env(i, config=train_config) for i in range(NUM_ENVS)])

    # Print config
    max_steps = ENV_CONFIG['duration'] * ENV_CONFIG['policy_frequency']
    print(f"\n{'='*60}")
    print(f"  DQN Baseline — highway-fast-v0 x {NUM_ENVS} envs")
    print(f"{'='*60}")
    print(f"  Timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs: {NUM_ENVS} (SubprocVecEnv)")
    print(f"  Duration:      {ENV_CONFIG['duration']}s -> {max_steps} max steps")
    print(f"  Policy freq:   {ENV_CONFIG['policy_frequency']} Hz")
    print(f"  Vehicles:      {ENV_CONFIG['vehicles_count']} (density={ENV_CONFIG['vehicles_density']})")
    print(f"  Reward:        raw (normalize=False)")
    print(f"  Collision:     -> reward={ENV_CONFIG['collision_reward']} + episode terminates")
    print(f"  High speed:    -> up to +{ENV_CONFIG['high_speed_reward']}/step")
    print(f"{'='*60}\n")

    # =================================================================
    #  DQN MODEL
    # =================================================================
    # Ottimizations vs previous version (single-env):
    #
    # buffer_size  50k -> 100k : more data from 4 parallel envs
    # batch_size   64  -> 128  : more stable gradient estimate
    # gradient_steps 1 -> 2    : more updates per step (sample-efficient)
    # exploration_fraction 0.5 -> 0.25 :
    #   BEFORE: epsilon min at 100k step -> at 18k epsilon ~ 0.83 (almost random)
    #   NOW:    epsilon min at  50k step -> at 18k epsilon ~ 0.64 (still exploratory,
    #           but agent already exploits what it learned)
    # learning_starts 500 -> 1000 : collects more diverse data before training

    model = DQN('MlpPolicy', env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=100_000,        # Larger buffer for multi-env
        learning_starts=1_000,      # ~250 step/env, collects diverse data
        batch_size=128,             # Larger batch -> stable gradient
        gamma=0.9,                  # Horizon ~10 steps (5s at 2 Hz)
        train_freq=4,               # 4 step/env -> 16 transitions per update
        gradient_steps=2,           # 2 gradient steps per update (sample-efficient)
        target_update_interval=250, # Stable target network
        exploration_fraction=0.25,  # Epsilon min at ~50k steps
        exploration_final_eps=0.05, # 5% residual exploration
        verbose=1,
        tensorboard_log=str(SAVE_DIR),
        device=device,
    )

    # =================================================================
    #  TRAINING
    # =================================================================

    if TRAIN:
        SAVE_DIR.mkdir(parents=True, exist_ok=True)

        callbacks = [
            TqdmCallback(),
            # Periodic checkpoint every 25k steps (safety net)
            CheckpointCallback(
                save_freq=25_000,
                save_path=str(SAVE_DIR),
                name_prefix="dqn_checkpoint",
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            # Save best model (collapse protection)
            BestModelCallback(
                save_path=str(SAVE_DIR),
                check_freq=500,
                window=30,
                verbose=1,
            ),
        ]

        model.learn(TOTAL_TIMESTEPS, callback=callbacks)
        model.save(str(SAVE_DIR / f"{MODEL_NAME}.zip"))
        print(f"\n+ Final model saved: {SAVE_DIR}/{MODEL_NAME}.zip")
        print(f"  (use {SAVE_DIR}/best_model.zip for the best model)")

    env.close()

    # =================================================================
    #  EVALUATION ON MULTIPLE SCENARIOS (Easy -> Expert) with rendering
    # =================================================================

    best_path = SAVE_DIR / f"{MODEL_NAME}.zip"
    final_path = SAVE_DIR / f"{MODEL_NAME}.zip"
    load_path = best_path if best_path.exists() else final_path

    print(f"\n{'='*60}")
    print(f"  EVALUATION: {load_path}")
    print(f"{'='*60}\n")

    model = DQN.load(str(load_path), device=device)

    # Import standard scenarios from metrics_tracker (same used for comparison)
    from metrics_tracker import EVAL_SCENARIOS

    N_EVAL_EPISODES = 5  # episodes per scenario (with render)
    all_results = {}

    for sc in EVAL_SCENARIOS:
        sc_name = sc['name']
        sc_config = sc['config']

        print(f"\n{'='*60}")
        print(f"  {sc_name}: {sc['description']}")
        print(f"  lanes={sc_config['lanes_count']}, vehicles={sc_config['vehicles_count']}, "
              f"density={sc_config['vehicles_density']}, duration={sc_config['duration']}")
        print(f"{'='*60}")

        eval_env = gymnasium.make("highway-fast-v0", config=sc_config, render_mode='human')

        returns, lengths, crashes = [], [], []
        for ep in range(N_EVAL_EPISODES):
            obs, _ = eval_env.reset(seed=42 + ep)
            done, ep_return, ep_len, crashed = False, 0, 0, False
            while not done:
                eval_env.render()
                time.sleep(RENDER_DELAY_MS / 1000.0)
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                crashed = info.get('crashed', False)
                ep_return += reward
                ep_len += 1
            returns.append(ep_return)
            lengths.append(ep_len)
            crashes.append(crashed)
            status = "CRASH" if crashed else "OK"
            print(f"  Ep {ep+1}/{N_EVAL_EPISODES}: [{status}] return={ep_return:.2f}, len={ep_len}")

        survival = (1 - np.mean(crashes)) * 100
        print(f"\n  --- {sc_name} ---")
        print(f"  Reward:   {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print(f"  Length:   {np.mean(lengths):.0f} steps")
        print(f"  Survival: {survival:.0f}%")

        all_results[sc_name] = {
            'avg_reward': float(np.mean(returns)),
            'std_reward': float(np.std(returns)),
            'avg_length': float(np.mean(lengths)),
            'survival_rate': float(survival),
            'crashes': int(sum(crashes)),
            'episodes': N_EVAL_EPISODES,
        }
        eval_env.close()

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'EVALUATION SUMMARY':^60}")
    print(f"{'='*60}")
    print(f"  {'Scenario':<12} {'Reward':>10} {'Survival':>10} {'Length':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {name:<12} {r['avg_reward']:>10.2f} {r['survival_rate']:>9.0f}% {r['avg_length']:>10.0f}")
    print(f"{'='*60}")

    # Save results JSON
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"dqn_baseline_eval_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump({
            "model": "DQN_baseline",
            "model_path": str(load_path),
            "timestamp": timestamp,
            "n_episodes_per_scenario": N_EVAL_EPISODES,
            "scenarios": all_results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
