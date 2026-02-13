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
import time
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json


TRAIN = False
TOTAL_TIMESTEPS = 500_000
MODEL_NAME = f'dqn_baseline_{TOTAL_TIMESTEPS//1000}k' 
SAVE_DIR = Path("highway_dqn")
NUM_ENVS = 4                # Env paralleli (SubprocVecEnv) — sfrutta multi-core
RENDER_DELAY_MS = 200       # Delay tra i frame durante rendering (ms): 100ms = ~10 FPS (lento & leggibile)


print("model name:", MODEL_NAME)
# =============================================================================
#  ENV CONFIG — Allineata a ACCEL FIXED_PARAMS
# =============================================================================
# Questi parametri DEVONO essere identici a ACCELGenerator.FIXED_PARAMS
# per garantire che la baseline e ACCEL siano confrontabili.

ENV_CONFIG = {
    "lanes_count": 3,
    "vehicles_count": 12,
    "vehicles_density": 0.8,
    "duration": 60,                    # 60 secondi per episodio
    "policy_frequency": 2,             # 2 decisioni/sec — reazione rapida per frenare
    "collision_reward": -10.0,          # Penalità FORTE per crash (-10.0, non -5!)
    "high_speed_reward": 0.3,          # Incentivo velocità: fino a +0.3/step a 30 m/s
    "right_lane_reward": 0.0,          # Nessun bonus corsia destra
    "lane_change_reward": 0,           # Neutrale: cambi corsia non penalizzati
    "reward_speed_range": [20, 30],
    "normalize_reward": False,         # RAW rewards: crash = -10.0 (penalità vera)
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}

# Observation ACCEL-compatibile — usata per training E valutazione.
# Il modello trainato con questa observation si aspetta shape (7,5),
# quindi anche l'env di valutazione DEVE usarla.
ACCEL_OBSERVATION = {
    "type": "Kinematics",
    "vehicles_count": 7,           # Ego + 6 altri (default: 5 = troppo pochi)
    "features": ["presence", "x", "y", "vx", "vy"],
    "features_range": {
        "x": [-100, 100],
        "y": [-100, 100],
        "vx": [-20, 20],
        "vy": [-20, 20],
    },
    "absolute": False,             # Posizioni relative all'ego
    "normalize": True,             # Normalizzato in [-1, 1]
    "see_behind": True,            # Specchietto: vede veicoli dietro
    "order": "sorted",             # Ordinati per distanza
}

# Reward con normalize_reward=False (allineata a ACCEL FIXED_PARAMS):
#   Per step: collision_reward * crashed + high_speed_reward * speed_frac
#   - Guida normale 25 m/s:      +0.15/step  (speed_frac ≈ 0.5)
#   - Guida perfetta 30 m/s:     +0.3/step   (speed_frac = 1.0)
#   - Collisione:                -10.0 + episodio TERMINA
#
# Con gamma=0.9, episodio 120 step (60s × 2 Hz):
#   - Return max scontato ≈ 3.0  (Σ 0.3 * 0.9^t)
#   - Crash = -10.0 istantaneo → domina il segnale
#   - Ratio penalità/ritorno ≈ 333% → rischio Q instabili se exploration troppo alta


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
        self.pbar.update(self.training_env.num_envs)
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
#  ENV FACTORY — Necessario per SubprocVecEnv (ogni env in un processo)
# =============================================================================

def make_env(rank: int, seed: int = 42, config: dict = None):
    """Crea un singolo env wrappato con Monitor (necessario per stats episodio)."""
    env_config = config or ENV_CONFIG
    def _init():
        env = gymnasium.make("highway-fast-v0", config=env_config)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# =============================================================================
#  MAIN — guard __name__ obbligatorio per SubprocVecEnv su Windows (spawn)
# =============================================================================

def main():
    # Device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    # Config di training: aggiunge observation ACCEL-compatibile se TRAIN=True
    # Questo rende il modello pre-trainato compatibile con dqn_accel.py
    if TRAIN:
        train_config = {**ENV_CONFIG, "observation": ACCEL_OBSERVATION}
        print("  Observation:   ACCEL-compatibile (7 veicoli, see_behind=True)")
    else:
        train_config = ENV_CONFIG
        print("  Observation:   default (modello già trainato)")

    # Env per training — SubprocVecEnv: ogni env in un processo separato
    print(f"\nCreating {NUM_ENVS} parallel environments (SubprocVecEnv)...")
    env = SubprocVecEnv([make_env(i, config=train_config) for i in range(NUM_ENVS)])

    # Stampa configurazione
    max_steps = ENV_CONFIG['duration'] * ENV_CONFIG['policy_frequency']
    print(f"\n{'='*60}")
    print(f"  DQN Baseline — highway-fast-v0 × {NUM_ENVS} envs")
    print(f"{'='*60}")
    print(f"  Timesteps:     {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs: {NUM_ENVS} (SubprocVecEnv)")
    print(f"  Duration:      {ENV_CONFIG['duration']}s → {max_steps} step max")
    print(f"  Policy freq:   {ENV_CONFIG['policy_frequency']} Hz")
    print(f"  Vehicles:      {ENV_CONFIG['vehicles_count']} (density={ENV_CONFIG['vehicles_density']})")
    print(f"  Reward:        raw (normalize=False)")
    print(f"  Collision:     → reward={ENV_CONFIG['collision_reward']} + episode terminates")
    print(f"  High speed:    → up to +{ENV_CONFIG['high_speed_reward']}/step")
    print(f"{'='*60}\n")

    # =================================================================
    #  DQN MODEL
    # =================================================================
    # Ottimizzazioni vs versione precedente (single-env):
    #
    # buffer_size  50k → 100k : più dati da 4 env paralleli
    # batch_size   64  → 128  : gradient estimate più stabile
    # gradient_steps 1 → 2    : più update per step (sample-efficient)
    # exploration_fraction 0.5 → 0.25 :
    #   PRIMA: epsilon min a 100k step → a 18k epsilon ≈ 0.83 (QUASI RANDOM!)
    #   ORA:   epsilon min a  50k step → a 18k epsilon ≈ 0.64 (ancora esplorativo,
    #          ma l'agente sfrutta già ciò che ha imparato)
    # learning_starts 500 → 1000 : raccoglie dati più diversi prima di trainare

    model = DQN('MlpPolicy', env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=100_000,        # Buffer più grande per multi-env
        learning_starts=1_000,      # ~250 step/env, raccoglie dati diversi
        batch_size=128,             # Batch più grande → gradiente stabile
        gamma=0.9,                  # Orizzonte ~10 step (5s a 2 Hz)
        train_freq=4,               # 4 step/env → 16 transizioni per update
        gradient_steps=2,           # 2 gradient step per update (sample-efficient)
        target_update_interval=250, # Target network stabile
        exploration_fraction=0.25,  # Epsilon min a ~50k step (era 100k!)
        exploration_final_eps=0.05, # 5% exploration residua
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
                check_freq=500,
                window=30,
                verbose=1,
            ),
        ]

        model.learn(TOTAL_TIMESTEPS, callback=callbacks)
        model.save(str(SAVE_DIR / f"{MODEL_NAME}.zip"))
        print(f"\n✓ Modello finale salvato: {SAVE_DIR}/{MODEL_NAME}.zip")
        print(f"  (usa {SAVE_DIR}/best_model.zip per il miglior modello)")

    env.close()

    # =================================================================
    #  VALUTAZIONE SU SCENARI MULTIPLI (Easy → Expert) con rendering
    # =================================================================

    best_path = SAVE_DIR / f"{MODEL_NAME}.zip"
    final_path = SAVE_DIR / f"{MODEL_NAME}.zip"
    load_path = best_path if best_path.exists() else final_path

    print(f"\n{'='*60}")
    print(f"  VALUTAZIONE: {load_path}")
    print(f"{'='*60}\n")

    model = DQN.load(str(load_path), device=device)

    # Importa scenari standard da metrics_tracker (stessi usati per il confronto)
    from metrics_tracker import EVAL_SCENARIOS

    N_EVAL_EPISODES = 5  # episodi per scenario (con render)
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
        print(f"  Durata:   {np.mean(lengths):.0f} steps")
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

    # Tabella riepilogativa
    print(f"\n{'='*60}")
    print(f"{'RIEPILOGO VALUTAZIONE':^60}")
    print(f"{'='*60}")
    print(f"  {'Scenario':<12} {'Reward':>10} {'Survival':>10} {'Durata':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for name, r in all_results.items():
        print(f"  {name:<12} {r['avg_reward']:>10.2f} {r['survival_rate']:>9.0f}% {r['avg_length']:>10.0f}")
    print(f"{'='*60}")

    # Salva risultati JSON
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
    print(f"\nRisultati salvati: {out_path}")


if __name__ == '__main__':
    main()
