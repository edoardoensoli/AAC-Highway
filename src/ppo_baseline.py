import gymnasium
import highway_env
import torch
from highway_env.vehicle.behavior import IDMVehicle
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from metrics_tracker import evaluate, HighwayMetrics
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json

TRAIN = False

class DMVehicle(IDMVehicle):
    ACC_MAX = 8.0
    ACC_MIN = -3.5
    tau = 0.8
    delta = 3.0
    POLITENESS = 1
    LANE_CHANGE_MIN_ACC_GAIN = 0.1
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0

env = gymnasium.make(
  "highway-v0",
  config={
    "lanes_count": 4, 
    "vehicles_count": 25,  
    "vehicles_density": 1,
    "duration": 60,
    "simulation_frequency":30,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    },
  render_mode='rgb_array'
)

# Device detection
if torch.backends.mps.is_available():
    device = "mps"
    print("Using MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

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

model = PPO('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=2e-4,
              n_steps=2048,
              batch_size=64,
              gamma=0.8,
              gae_lambda=0.95,
              clip_range=0.2,
              ent_coef=0.05,
              verbose=1,
              tensorboard_log="highway_ppo/",
              device=device)

if TRAIN:
    model.learn(int(100000), callback=TqdmCallback())
    model.save("highway_ppo/model")

# Load and test saved model
model = PPO.load("highway_ppo/model", device=device)

# =============================================================================
# VALUTAZIONE CON METRICHE
# =============================================================================
print("\n" + "="*60)
print("VALUTAZIONE MODELLO PPO")
print("="*60 + "\n")

# Metriche disponibili:
# - collision_rate: % di episodi con crash
# - survival_rate: % di episodi senza crash
# - avg_reward: ricompensa media
# - avg_speed: velocità media (m/s)
# - max_speed: velocità massima (m/s)
# - cars_overtaken: media sorpassi per episodio
# - total_cars_overtaken: totale sorpassi
# - avg_episode_length: durata media episodi
# - lane_changes: cambi corsia medi
# - distance_traveled: distanza percorsa media (m)
# - min_ttc: Time To Collision minimo (sicurezza)
# - near_miss_rate: % episodi con quasi-incidenti

# Usa tutte le metriche (None) oppure specifica quelle desiderate
metrics_to_use = {
    'collision_rate', 
    'survival_rate',
    'avg_reward',
    'cars_overtaken',
    'total_cars_overtaken',
    'avg_speed',
    'max_speed',
    'distance_traveled',
    'lane_changes',
}

# Valutazione
results = evaluate(
    model=model,
    env=env,
    n_episodes=10,
    metrics=metrics_to_use,
    render=True,
    verbose=True,  # Mostra sorpassi in tempo reale
    seed=42
)

# Salva risultati in JSON
repo_root = Path(__file__).resolve().parents[1]
logs_dir = repo_root / "logs"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = logs_dir / f"ppo_metrics_{timestamp}.json"

# Crea tracker per salvare (la funzione evaluate restituisce solo il dict)
tracker = HighwayMetrics(metrics=metrics_to_use)
# Riesegui una rapida valutazione per avere i dati completi (oppure salva manualmente)
logs_dir.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "model": "PPO",
        "timestamp": timestamp,
        "n_episodes": 10,
        "config": {
            "lanes_count": env.unwrapped.config["lanes_count"],
            "vehicles_count": env.unwrapped.config["vehicles_count"],
            "vehicles_density": env.unwrapped.config["vehicles_density"],
            "duration": env.unwrapped.config["duration"],
            "simulation_frequency": env.unwrapped.config["simulation_frequency"],
            "other_vehicles_type": env.unwrapped.config["other_vehicles_type"],
        },
        "results": results,
    }, f, indent=2)
print(f"\nRisultati salvati in: {out_path}")

# Visualizzazione interattiva (opzionale - decommentare per usare)
"""
print("\nAvvio visualizzazione interattiva...")
while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
"""