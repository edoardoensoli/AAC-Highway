import gymnasium
import highway_env
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from metrics_tracker import evaluate, HighwayMetrics
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import os

TRAIN = True

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
ENV_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 7,           # ego + 6 nearest vehicles
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
        "absolute": False,
        "order": "sorted",
        "normalize": True,
        "clip": True,
        "see_behind": True,
        "observe_intentions": False,
    },
    "lanes_count": 3,
    "vehicles_count": 12,
    "vehicles_density": 0.8,
    "duration": 60,                    # 60 secondi per episodio
    "policy_frequency": 2,             # 2 decisioni/sec — reazione rapida per frenare
    "collision_reward": -10.0,         # Penalità FORTE per crash
    "high_speed_reward": 0.3,          # Incentivo velocità: fino a +0.3/step a 30 m/s
    "right_lane_reward": 0.0,          # Nessun bonus corsia destra
    "lane_change_reward": 0,           # Neutrale: cambi corsia non penalizzati
    "reward_speed_range": [20, 30],
    "normalize_reward": False,         # RAW rewards: crash = -10.0 (penalità vera)
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

env = gymnasium.make(
    "highway-fast-v0",
    config=ENV_CONFIG,
    render_mode="rgb_array",
)


device = "cpu"

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

# ---------------------------------------------------------------------------
# Training setup — 1M env steps, checkpoints at 100k / 200k / 500k / 1M
# ---------------------------------------------------------------------------
TOTAL_TIMESTEPS = 1_000_000
SAVE_DIR = "highway_ppo/v1"
os.makedirs(SAVE_DIR, exist_ok=True)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=2e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.8,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    verbose=1,
    tensorboard_log=None,
    device=device,
)


# --- Custom callback: save at exact milestone steps -----------------------
class MilestoneCheckpointCallback(BaseCallback):
    """Save the model at specific timestep milestones."""

    def __init__(self, milestones: list[int], save_dir: str):
        super().__init__()
        self.milestones = sorted(milestones)
        self.save_dir = save_dir
        self._saved: set[int] = set()

    def _on_step(self) -> bool:
        for m in self.milestones:
            if m not in self._saved and self.num_timesteps >= m:
                path = os.path.join(self.save_dir, f"model_{m // 1000}k")
                self.model.save(path)
                print(f"\n💾 Checkpoint saved: {path}  (step {self.num_timesteps:,})")
                self._saved.add(m)
        return True


if TRAIN:
    milestones = [100_000, 200_000, 500_000]
    callbacks = [
        TqdmCallback(),
        MilestoneCheckpointCallback(milestones, SAVE_DIR),
    ]
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    # Final model at 1M
    final_path = os.path.join(SAVE_DIR, "model_1000k")
    model.save(final_path)
    print(f"\n💾 Final model saved: {final_path}")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("VALUTAZIONE MODELLO PPO")
print("=" * 60 + "\n")

# Load best / final model
model = PPO.load(os.path.join(SAVE_DIR, "model_1000k"), device=device)

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
    env=env,
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
print(f"\nRisultati salvati in: {out_path}")