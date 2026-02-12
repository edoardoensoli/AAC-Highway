"""
DQN Baseline per Highway-Env
=============================
Allena un modello DQN su highway-fast-v0 con parametri allineati
a ACCEL FIXED_PARAMS per un confronto equo.

Il modello salvato (best_model.zip) può essere usato come pre-trained per ACCEL:
  python src/dqn_accel.py --pretrained highway_dqn/best_model.zip
"""

import gymnasium
import highway_env
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json


TRAIN = True
TOTAL_TIMESTEPS = 200_000
SAVE_DIR = Path("highway_dqn")

# =============================================================================
#  ENV CONFIG — Allineata a ACCEL FIXED_PARAMS
# =============================================================================
# Questi parametri DEVONO essere identici a ACCELGenerator.FIXED_PARAMS
# per garantire che la baseline e ACCEL siano confrontabili.

ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 25,
    "vehicles_density": 1,
    "duration": 60,                    # 60 secondi per episodio
    "policy_frequency": 2,             # 2 decisioni/sec — reazione rapida per frenare
    "collision_reward": -5.0,          # Penalità FORTE per crash
    "high_speed_reward": 0.4,          # Incentivo velocità: fino a +0.4/step a 30 m/s
    "right_lane_reward": 0.1,          # Bonus corsia destra: +0.1/step
    "lane_change_reward": 0,           # Neutrale: cambi corsia non penalizzati
    "reward_speed_range": [20, 30],
    "normalize_reward": False,         # RAW rewards: crash = -5.0 (penalità vera)
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

# Reward con normalize_reward=False (allineata a ACCEL FIXED_PARAMS):
#   Per step: collision_reward * crashed + high_speed * frac + right_lane * frac
#   - Guida normale 25 m/s:      +0.25 + 0.0 = +0.25/step
#   - Guida perfetta 30 m/s dx:  +0.4  + 0.1 = +0.5/step
#   - Collisione:                -5.0 + episodio TERMINA
#
# Con gamma=0.9, episodio 120 step (60s × 2 Hz):
#   - Return max scontato ≈ 5.0
#   - Crash = -5.0 istantaneo + reward future perse ≈ -7.5 totale
#   - Ratio penalità/ritorno = 150% → segnale FORTE per evitare collisioni


# =============================================================================
#  CALLBACKS
# =============================================================================

class TqdmCallback(BaseCallback):
    """Barra di progresso."""
    def __init__(self):
        super().__init__()
        self.pbar = None

    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'], unit="step")

    def _on_step(self):
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()


class BestModelCallback(BaseCallback):
    """
    Salva il modello quando la reward media (su una finestra) migliora.
    
    Questo previene la perdita di progressi in caso di collapse:
    il file best_model.zip contiene sempre il miglior modello visto.
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
        # Raccogli statistiche episodi dal Monitor wrapper
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

        # Check periodico
        if len(self.episode_rewards) >= self.window and self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-self.window:])
            mean_length = np.mean(self.episode_lengths[-self.window:])

            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                self.best_mean_length = mean_length
                self.saves_count += 1

                # Salva modello
                self.model.save(str(self.save_path / "best_model"))

                # Salva info
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
                    print(f"\n  ★ BEST MODEL salvato! Reward: {mean_reward:.2f} (+{improvement:.2f}) "
                          f"| Len: {mean_length:.0f}/120 | Step: {self.n_calls:,}")

        return True


# =============================================================================
#  SETUP
# =============================================================================

# Device
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

# Env per training (senza render per velocità)
env = gymnasium.make("highway-fast-v0", config=ENV_CONFIG)

# Stampa configurazione
print(f"\n{'='*60}")
print(f"  DQN Baseline — highway-fast-v0")
print(f"{'='*60}")
print(f"  Timesteps:     {TOTAL_TIMESTEPS:,}")
print(f"  Duration:      {ENV_CONFIG['duration']}s → {ENV_CONFIG['duration'] * ENV_CONFIG['policy_frequency']} step max")
print(f"  Policy freq:   {ENV_CONFIG['policy_frequency']} Hz")
print(f"  Vehicles:      {ENV_CONFIG['vehicles_count']} (density={ENV_CONFIG['vehicles_density']})")
print(f"  Reward:        raw (normalize=False)")
print(f"  Collision:     → reward=-5.0 + episode terminates")
print(f"  High speed:    → up to +0.4 per step")
print(f"  Right lane:    → up to +0.1 per step")
print(f"{'='*60}\n")


# =============================================================================
#  DQN MODEL
# =============================================================================
# Parametri ottimizzati per highway-env con reward raw (normalize_reward=False):
#
# gamma=0.9 è il punto ideale:
#   - Orizzonte effettivo: ~10 step (5 secondi a 2 Hz)
#   - Q-values range: [-5, +5] (gestibile per la rete neurale)
#   - Crash = -5.0 istantaneo + reward future perse ≈ -7.5 totale
#   - Guida sicura 120 step: return scontato ≈ 5.0
#   - Ratio penalità/ritorno = 150% → segnale forte per evitare crash
#
# train_freq=4: con single-env, aggiorna più spesso per efficienza
# target_update_interval=250: target network stabile (50 era troppo aggressivo)

model = DQN('MlpPolicy', env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=5e-4,
    buffer_size=50_000,         # Proporzionato a 200k step (25% del training)
    learning_starts=500,        # Inizia presto ma con un po' di dati
    batch_size=64,
    gamma=0.9,                  # Q-values moderati, training stabile
    train_freq=4,               # 1 update ogni 4 step (efficiente per single-env)
    gradient_steps=1,
    target_update_interval=250, # Target network aggiornata ogni 250 update (stabile)
    exploration_fraction=0.5,   # Esplora per metà del training (100k step)
    exploration_final_eps=0.05, # 5% exploration residua
    verbose=1,
    tensorboard_log=str(SAVE_DIR),
    device=device,
)


# =============================================================================
#  TRAINING
# =============================================================================

if TRAIN:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        TqdmCallback(),
        # Checkpoint periodico ogni 25k step (safety net)
        CheckpointCallback(
            save_freq=25_000,
            save_path=str(SAVE_DIR),
            name_prefix="dqn_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        ),
        # Salva il miglior modello (protezione contro collapse)
        BestModelCallback(
            save_path=str(SAVE_DIR),
            check_freq=500,     # Controlla ogni 500 step
            window=30,          # Media su ultimi 30 episodi
            verbose=1,
        ),
    ]

    model.learn(TOTAL_TIMESTEPS, callback=callbacks)
    model.save(str(SAVE_DIR / "model_v2"))
    print(f"\n✓ Modello finale salvato: {SAVE_DIR}/model_v2.zip")
    print(f"  (usa {SAVE_DIR}/best_model.zip per il miglior modello)")


# =============================================================================
#  VALUTAZIONE
# =============================================================================

# Carica il MIGLIOR modello (non l'ultimo, che potrebbe aver collassato)
best_path = SAVE_DIR / "best_model.zip"
final_path = SAVE_DIR / "model_v2.zip"
load_path = best_path if best_path.exists() else final_path

print(f"\n{'='*60}")
print(f"  VALUTAZIONE: {load_path}")
print(f"{'='*60}\n")

model = DQN.load(str(load_path), device=device)

# Env di valutazione con render
eval_env = gymnasium.make("highway-fast-v0", config=ENV_CONFIG, render_mode='rgb_array')

try:
    from metrics_tracker import evaluate as mt_evaluate

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

    results = mt_evaluate(
        model=model,
        env=eval_env,
        n_episodes=10,
        metrics=metrics_to_use,
        render=True,
        verbose=True,
        seed=42,
    )

    # Salva risultati
    repo_root = Path(__file__).resolve().parents[1]
    logs_dir = repo_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = logs_dir / f"dqn_baseline_{timestamp}.json"

    with open(out_path, "w") as f:
        json.dump({
            "model": "DQN_baseline_v2",
            "model_path": str(load_path),
            "timestamp": timestamp,
            "n_episodes": 10,
            "env_config": ENV_CONFIG,
            "dqn_params": {
                "gamma": 0.9,
                "train_freq": 4,
                "target_update_interval": 250,
                "exploration_fraction": 0.5,
                "exploration_final_eps": 0.05,
                "buffer_size": 50_000,
                "net_arch": [256, 256],
            },
            "results": results,
        }, f, indent=2)
    print(f"\nRisultati salvati: {out_path}")

except Exception as e:
    print(f"\n[WARN] Valutazione con metrics_tracker fallita: {e}")
    print("Valutazione semplice...")

    returns, lengths = [], []
    for ep in range(10):
        obs, _ = eval_env.reset()
        done, ep_return, ep_len = False, 0, 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_len += 1
        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"  Ep {ep+1}: return={ep_return:.2f}, len={ep_len}")

    print(f"\n  Media: return={np.mean(returns):.2f} ± {np.std(returns):.2f}, "
          f"len={np.mean(lengths):.0f}/120")

eval_env.close()
env.close()
