import gymnasium
import highway_env
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from metrics_tracker import evaluate, HighwayMetrics
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import os

TRAIN = True
NUM_ENVS = 4

# ---------------------------------------------------------------------------
# Environment configuration (7x5 observation to match DQN models)
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100], "y": [-100, 100],
            "vx": [-20, 20], "vy": [-20, 20],
        },
        "absolute": False,
        "normalize": True,
        "see_behind": True,
        "order": "sorted",
    },
    "lanes_count": 3,
    "vehicles_count": 12,
    "vehicles_density": 0.8,
    "duration": 40,
    "policy_frequency": 2,
    "collision_reward": -10.0,
    "high_speed_reward": 0.3,
    "right_lane_reward": 0.0,
    "lane_change_reward": 0,
    "reward_speed_range": [20, 30],
    "normalize_reward": False,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

device = "cpu"

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None
        self._last = 0
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'], unit="step")
    
    def _on_step(self):
        self.pbar.update(self.num_timesteps - self._last)
        self._last = self.num_timesteps
        return True
    
    def _on_training_end(self):
        self.pbar.close()


# --- Milestone checkpoint callback ----------------------------------------
class MilestoneCheckpointCallback(BaseCallback):
    """Save the model when reaching specific timestep milestones."""

    def __init__(self, milestones: list[int], save_dir: str):
        super().__init__()
        self.milestones = sorted(milestones)
        self.save_dir = save_dir
        self._saved: set[int] = set()

    def _on_step(self) -> bool:
        for m in self.milestones:
            if m not in self._saved and self.num_timesteps >= m:
                if m >= 1_000_000:
                    name = f"ppo_baseline_{m // 1_000_000}M.zip"
                else:
                    name = f"ppo_baseline_{m // 1000}k.zip"
                path = os.path.join(self.save_dir, name)
                self.model.save(path)
                print(f"\nCheckpoint saved: {path}  (step {self.num_timesteps:,})")
                self._saved.add(m)
        return True


def make_env(rank: int, seed: int = 42, config: dict = None):
    """Create a single env wrapped with Monitor for episode stats."""
    env_config = config or ENV_CONFIG
    def _init():
        env = gymnasium.make("highway-fast-v0", config=env_config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


if __name__ == '__main__':
    TOTAL_TIMESTEPS = 1_000_000
    SAVE_DIR = "highway_ppo"
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PPO Baseline — highway-fast-v0 x {NUM_ENVS} envs")
    print(f"{'='*60}")
    print(f"  Timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs: {NUM_ENVS} (SubprocVecEnv)")
    print(f"  Duration:      {ENV_CONFIG['duration']}s")
    print(f"  Policy freq:   {ENV_CONFIG['policy_frequency']} Hz")
    print(f"  Vehicles:      {ENV_CONFIG['vehicles_count']} (density={ENV_CONFIG['vehicles_density']})")
    print(f"{'='*60}\n")

    if TRAIN:
        print(f"Creating {NUM_ENVS} parallel environments (SubprocVecEnv)...")
        env = SubprocVecEnv([make_env(i, config=ENV_CONFIG) for i in range(NUM_ENVS)])

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=2e-4,
            n_steps=512,
            batch_size=256,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,
            verbose=0,
            tensorboard_log=None,
            device=device,
        )

        milestones = [200_000, 500_000, 1_000_000]
        callbacks = [
            TqdmCallback(),
            MilestoneCheckpointCallback(milestones, SAVE_DIR),
        ]
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
        env.close()

    # ---------------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PPO MODEL EVALUATION")
    print("=" * 60 + "\n")

    eval_env = gymnasium.make("highway-fast-v0", config=ENV_CONFIG, render_mode="rgb_array")
    model = PPO.load(os.path.join(SAVE_DIR, "ppo_baseline_1M.zip"), device=device)

    metrics_to_use = {
        "collision_rate",
        "survival_rate",
        "avg_reward",
        "cars_overtaken",
        "total_cars_overtaken",
        "avg_speed",
        "max_speed",
        "distance_traveled",
        "lane_changes",
    }

    results = evaluate(
        model=model,
        env=eval_env,
        n_episodes=10,
        metrics=metrics_to_use,
        render=True,
        verbose=True,
        seed=42,
    )

    # Save results to JSON
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"ppo_metrics_{timestamp}.json"
    logs_dir.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "PPO",
                "timestamp": timestamp,
                "total_timesteps": TOTAL_TIMESTEPS,
                "n_episodes": 10,
                "config": ENV_CONFIG,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to: {out_path}")