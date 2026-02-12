"""
DQN + ACCEL Training for Highway-Env
======================================
Implements ACCEL (Evolving Curricula with Regret-Based Environment Design)
with DQN on the highway-env driving simulator.

Based on:
- ACCEL: Parker-Holder et al., 2022 (https://accelagent.github.io/)
- Robust PLR: Jiang et al., 2021 (https://arxiv.org/abs/2110.02439)
- DCD Framework: Facebook Research (https://github.com/facebookresearch/dcd)

Adapted for Stable Baselines3 DQN + highway-env.

Key design decisions:
- Uses highway-fast-v0 for 15x simulation speedup
- SubprocVecEnv for true CPU parallelization (each env in its own process)
- Lazy config updates via unwrapped.config.update (no env recreation)
- PLR scoring via TD-error proxy + return deviation
- Rank-based score transform with staleness weighting (as in DCD paper)
- Proportionate replay schedule (DCD default)
- ACCEL mutations = small random perturbations to env params
"""

import numpy as np
import gymnasium
import highway_env
import torch
import json
import time
import os
import platform
import copy
import pickle
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm


# =============================================================================
#  CONFIGURABLE HIGHWAY ENVIRONMENT WRAPPER
# =============================================================================

class ConfigurableHighwayEnv(gymnasium.Wrapper):
    """
    Wrapper per highway-env che supporta il cambio di configurazione lazy
    e aggiunge una reward di prossimità per incentivare la distanza di sicurezza.
    
    La nuova configurazione viene applicata al prossimo reset(),
    evitando di ricreare l'env durante un episodio attivo.
    Questo è il pattern corretto per PLR/ACCEL dove i livelli cambiano
    solo tra episodi.
    
    PROXIMITY REWARD:
    Aggiunge una penalità continua quando l'agente è troppo vicino ad altri veicoli.
    Questo insegna proattivamente a mantenere la distanza di sicurezza, invece di
    imparare solo dal crash terminale (-5.0). La penalità è quadratica:
    forte quando molto vicini, dolce quando a distanza moderata.
    
    Formula: penalty = -max_penalty * max(0, 1 - min_dist / safety_distance)²
    
    Con safety_distance=25m e max_penalty=0.5:
      25m+ → 0.0 (safe)
      15m  → -0.08/step (attenzione)
      10m  → -0.18/step (pericoloso, quasi cancella speed reward)
       5m  → -0.32/step (molto pericoloso, supera speed reward)
       0m  → -0.50/step (contatto)
    """

    def __init__(self, env_id: str = "highway-fast-v0", initial_config: Optional[Dict] = None,
                 safety_distance: float = 25.0, proximity_penalty: float = 0.0):
        self.env_id = env_id
        self._config = initial_config or {}
        self._next_config = None
        self.safety_distance = safety_distance
        self.proximity_penalty = proximity_penalty

        env = gymnasium.make(env_id, config=self._config)
        super().__init__(env)

    def set_next_config(self, config: Dict):
        """Schedula una nuova configurazione per il prossimo reset."""
        self._next_config = config.copy()

    def get_config(self) -> Dict:
        return self._config.copy()

    def reset(self, **kwargs):
        if self._next_config is not None:
            self._config = self._next_config
            self._next_config = None
            # Aggiorna config direttamente — highway-env legge self.config in reset()
            # Evita gymnasium.make() che è estremamente costoso (~100ms per chiamata)
            self.env.unwrapped.config.update(self._config)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Calcola penalità di prossimità solo se l'episodio è ancora attivo
        if not terminated:
            proximity_pen = self._compute_proximity_penalty()
            reward += proximity_pen
            if proximity_pen < 0:
                info['proximity_penalty'] = float(proximity_pen)

        return obs, reward, terminated, truncated, info

    def _compute_proximity_penalty(self) -> float:
        """
        Calcola una penalità basata sulla distanza minima dai veicoli vicini.
        
        Considera solo veicoli DAVANTI o ai lati (dx > -5m), non quelli già
        superati dietro. La penalità è quadratica per creare un gradiente
        più forte vicino alla collisione.
        
        Returns:
            Penalità negativa (0.0 se sicuro, fino a -proximity_penalty se a contatto)
        """
        try:
            ego = self.env.unwrapped.vehicle
            road = self.env.unwrapped.road
            if ego is None or road is None:
                return 0.0
        except Exception:
            return 0.0

        min_dist = float('inf')

        for v in road.vehicles:
            if v is ego:
                continue
            dx = v.position[0] - ego.position[0]
            dy = v.position[1] - ego.position[1]

            # Considera solo veicoli davanti o nelle vicinanze laterali
            # dx > -5: ignora veicoli già superati (ben dietro)
            # |dy| < 8: ignora veicoli su corsie lontane (> 2 corsie, ~4m ciascuna)
            if dx > -5.0 and abs(dy) < 8.0:
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    min_dist = dist

        if min_dist >= self.safety_distance:
            return 0.0

        # Penalità quadratica: cresce rapidamente avvicinandosi
        ratio = 1.0 - min_dist / self.safety_distance
        return -self.proximity_penalty * ratio * ratio


# =============================================================================
#  LEVEL SAMPLER (PLR Implementation - faithful to DCD)
# =============================================================================

@dataclass
class Level:
    """Un livello di training = configurazione highway-env + statistiche."""
    seed: int
    config: Dict
    score: float = 0.0
    max_score: float = float('-inf')
    visit_count: int = 0
    staleness: int = 0
    returns: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)
    parent_seed: Optional[int] = None
    num_mutations: int = 0


class LevelSampler:
    """
    Prioritized Level Replay (PLR) sampler.
    
    Implementazione fedele al paper DCD/PLR con:
    - Scoring via value_l1 (adattato come TD-error per DQN)
    - Rank-based score transform (default DCD)
    - Staleness-aware sampling (staleness_coef=0.3, power transform)
    - Proportionate replay schedule (default DCD)
    - Seed buffer con priorità replay_support
    
    Parametri DCD di riferimento:
    - replay_prob: 0.95 (alta per ACCEL/Robust PLR)
    - score_transform: rank, temperature: 0.1
    - staleness_coef: 0.3, staleness_transform: power
    - alpha: 1.0 (EWA smoothing)
    - rho: 0.1 (min fill ratio prima di fare replay)
    """

    def __init__(
        self,
        buffer_size: int = 4000,
        replay_prob: float = 0.95,
        score_transform: str = 'rank',
        temperature: float = 0.1,
        staleness_coef: float = 0.3,
        staleness_transform: str = 'power',
        staleness_temperature: float = 1.0,
        alpha: float = 1.0,
        rho: float = 0.1,
        seed: int = 42,
    ):
        self.buffer_size = buffer_size
        self.replay_prob = replay_prob
        self.score_transform = score_transform
        self.temperature = temperature
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.alpha = alpha
        self.rho = rho
        self.rng = np.random.RandomState(seed)

        self.levels: Dict[int, Level] = {}
        self.seed_counter = 0

        self.stats = {
            'replay_count': 0,
            'new_count': 0,
            'mutation_count': 0,
        }

    @property
    def proportion_filled(self) -> float:
        return min(len(self.levels) / max(self.buffer_size, 1), 1.0)

    @property
    def is_warm(self) -> bool:
        return self.proportion_filled >= self.rho

    def _next_seed(self) -> int:
        self.seed_counter += 1
        return self.seed_counter

    def add_level(self, config: Dict, parent_seed: Optional[int] = None) -> int:
        """Aggiunge un livello al buffer. Se pieno, sostituisce il livello con score più basso."""
        seed = self._next_seed()
        level = Level(
            seed=seed,
            config=config.copy(),
            parent_seed=parent_seed,
            num_mutations=(
                self.levels[parent_seed].num_mutations + 1
                if parent_seed and parent_seed in self.levels
                else 0
            )
        )

        if len(self.levels) >= self.buffer_size:
            # Rimuovi livello con score più basso (come DCD: seed_buffer_priority=replay_support)
            min_seed = min(self.levels, key=lambda s: self.levels[s].score)
            del self.levels[min_seed]

        self.levels[seed] = level
        return seed

    def sample_replay_decision(self) -> bool:
        """
        Proportionate replay schedule (DCD default).
        Replay con probabilità proporzionale al riempimento del buffer.
        """
        if not self.is_warm:
            return False
        proportion_seen = self.proportion_filled
        if proportion_seen >= self.rho and self.rng.rand() < min(proportion_seen, self.replay_prob):
            return True
        return False

    def sample_replay_level(self) -> int:
        """Campiona un livello dal buffer con prioritized sampling."""
        seeds = list(self.levels.keys())
        scores = np.array([self.levels[s].score for s in seeds])
        staleness = np.array([self.levels[s].staleness for s in seeds], dtype=np.float64)

        # Score weights (rank transform, come DCD)
        score_weights = self._transform_scores(
            self.score_transform, self.temperature, scores
        )

        # Staleness weights
        staleness_weights = np.zeros_like(staleness)
        if self.staleness_coef > 0:
            staleness_weights = self._transform_scores(
                self.staleness_transform, self.staleness_temperature, staleness
            )

        # Combinazione pesata (come DCD: (1-s)*score + s*staleness)
        weights = (1 - self.staleness_coef) * score_weights + self.staleness_coef * staleness_weights

        # Normalizza
        total = weights.sum()
        if total > 0:
            weights /= total
        else:
            weights = np.ones(len(seeds)) / len(seeds)

        # Campiona
        idx = self.rng.choice(len(seeds), p=weights)
        selected_seed = seeds[idx]

        # Aggiorna staleness (come DCD: incrementa tutti, azzera il selezionato)
        for s in self.levels:
            self.levels[s].staleness += 1
        self.levels[selected_seed].staleness = 0

        return selected_seed

    def update_score(self, seed: int, td_error: float, episode_return: float):
        """
        Aggiorna lo score del livello.
        
        Usa TD-error come proxy per value_l1 (|returns - value_preds|)
        che è la scoring function di default nel DCD paper.
        """
        if seed not in self.levels:
            return

        level = self.levels[seed]
        level.visit_count += 1
        level.returns.append(episode_return)
        level.td_errors.append(td_error)

        # Score = media recente TD error (learning potential)
        recent = level.td_errors[-10:]
        new_score = np.mean(recent) if recent else 0.0

        # EWA update (come DCD: alpha=1.0 di default, weighted update)
        level.score = (1 - self.alpha) * level.score + self.alpha * new_score
        level.max_score = max(level.max_score, new_score)

    def _transform_scores(self, transform: str, temperature: float, scores: np.ndarray) -> np.ndarray:
        """
        Trasforma gli score per il sampling (matching esatto dell'implementazione DCD).
        """
        if len(scores) == 0:
            return np.array([])

        if transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1.0 / ranks ** (1.0 / temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (scores.clip(0) + eps) ** (1.0 / temperature)
        elif transform == 'softmax':
            # Numerically stable softmax
            shifted = scores - scores.max()
            weights = np.exp(shifted / max(temperature, 1e-8))
        else:
            weights = np.ones_like(scores)

        return weights

    def get_stats(self) -> Dict:
        if not self.levels:
            return {**self.stats, 'buffer_size': 0, 'buffer_capacity': self.buffer_size,
                    'proportion_filled': 0, 'is_warm': False, 'avg_score': 0, 'max_score': 0}
        return {
            **self.stats,
            'buffer_size': len(self.levels),
            'buffer_capacity': self.buffer_size,
            'proportion_filled': self.proportion_filled,
            'is_warm': self.is_warm,
            'avg_score': float(np.mean([l.score for l in self.levels.values()])),
            'max_score': float(max(l.score for l in self.levels.values())),
            'avg_visits': float(np.mean([l.visit_count for l in self.levels.values()])),
            'avg_staleness': float(np.mean([l.staleness for l in self.levels.values()])),
        }

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'levels': self.levels,
                'seed_counter': self.seed_counter,
                'stats': self.stats,
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.levels = data['levels']
        self.seed_counter = data['seed_counter']
        self.stats = data['stats']


# =============================================================================
#  ACCEL LEVEL GENERATOR
# =============================================================================

class ACCELGenerator:
    """
    Genera e muta configurazioni highway-env per il curriculum ACCEL.
    
    Il curriculum varia solo i parametri dell'ambiente (traffico, corsie, durata)
    mantenendo fissi i parametri di reward per un segnale di apprendimento consistente.
    
    ACCEL mutations = perturbazioni random piccole (1-3 parametri) come nel paper.
    """

    # Spazio dei parametri variabili per il curriculum
    # Range progettati per coprire Stage 0 (2 corsie, 8 veicoli) → Stage 6 (2 corsie, 50 veicoli)
    PARAM_SPACE = {
        'vehicles_count':  {'type': 'int',   'range': (5, 50),    'step': 5},
        'vehicles_density': {'type': 'float', 'range': (0.5, 2.0), 'step': 0.2},
        'lanes_count':     {'type': 'int',   'range': (2, 4),     'step': 1},
        'initial_spacing': {'type': 'float', 'range': (1.0, 3.0), 'step': 0.5},
        'duration':        {'type': 'int',   'range': (30, 80),   'step': 10},
    }

    # Parametri fissi (reward e simulazione - mai mutati per consistenza)
    #
    # REWARD DESIGN SEMPLIFICATO (normalize_reward=False):
    #   SOLO 2 obiettivi: vai veloce (supera le auto) + non crashare.
    #
    #   Per-step reward = collision_reward * crashed
    #                   + high_speed_reward * speed_fraction
    #
    #   Guida normale ~25 m/s: +0.2/step
    #   Guida perfetta 30 m/s: +0.4/step
    #   Collisione:            -10.0 + episodio TERMINA
    #
    #   Con gamma=0.95, episodio 60 step (Stage 0: 30s×2Hz):
    #     Return max scontato ≈ 0.4 * Σ(0.95^t, t=0..59) ≈ 7.6
    #     Return normale    ≈ 0.2 * 18.9 ≈ 3.8
    #     Crash = -10.0 istantaneo + perdita reward future
    #     Crash a step 5:  ~-10.0 + earned(~1.0) = -9.0 vs survive(3.8) → gap 12.8
    #     Crash a step 30: ~-10.0 * 0.95^30 + earned(2.9) = -0.2 vs survive(3.8)
    #
    #   RIMOSSO: right_lane_reward (rumore — spinge verso corsia lenta)
    #   RIMOSSO: proximity_penalty (confondeva il segnale — su 2 corsie
    #            l'agente è SEMPRE entro 25m da qualcuno)
    #
    FIXED_PARAMS = {
        'policy_frequency': 2,
        'collision_reward': -10.0,        # Penalità MOLTO FORTE per crash: il segnale più chiaro
        'high_speed_reward': 0.4,         # Incentivo velocità: vai veloce = supera le auto
        'right_lane_reward': 0.0,         # DISABILITATO: era rumore, spingeva verso corsia lenta
        'lane_change_reward': 0.0,        # Neutrale: cambi corsia liberi
        'reward_speed_range': [20, 30],
        'normalize_reward': False,        # RAW rewards: crash = -10.0 (penalità vera, non mappata)
        # OSSERVAZIONE: l'agente deve VEDERE abbastanza per prendere decisioni sicure.
        # Default highway-env: solo 5 veicoli, solo davanti, no specchietto.
        # Migliorato: 7 veicoli (vede quasi tutti), specchietto retrovisore,
        # posizioni relative (molto più facili da imparare per la rete neurale).
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 7,           # Ego + 6 altri (default: 5 = ego + 4, troppo pochi)
            'features': ['presence', 'x', 'y', 'vx', 'vy'],
            'features_range': {
                'x': [-100, 100],
                'y': [-100, 100],
                'vx': [-20, 20],
                'vy': [-20, 20],
            },
            'absolute': False,             # Posizioni RELATIVE all'ego (default, più facile da imparare)
            'normalize': True,             # Normalizzato in [-1, 1]
            'see_behind': False,            # SPECCHIETTO: vede chi arriva da dietro per cambi corsia sicuri
            'order': 'sorted',             # Ordinati per distanza
        },
        # other_vehicles_type è ORA per-stage: IDM (prevedibile) → Aggressive (caotico)
        # NON va qui perché get_stage_config() fa config.update(FIXED_PARAMS)
        # e sovrascriverebbe il tipo veicolo specifico dello stage.
    }

    # ---- Curriculum di difficoltà progressiva ----
    # Ogni stage rappresenta un punto di ancoraggio nella scala di difficoltà.
    # L'agente deve padroneggiare uno stage prima di passare al successivo.
    # ACCEL muta livelli ATTORNO allo stage corrente per esplorare la frontiera.
    #
    # DESIGN KEY v4: Stage 0 su 2 corsie per forzare interazione, poi subito 3
    # corsie. Aggiungere più auto su 2 corsie è un vicolo cieco: la strada è
    # satura e l'agente non ha dove andare. 3 corsie = nuova abilità reale
    # (navigazione multi-corsia) e più spazio per manovrare con traffico denso.
    #
    # REGOLA FERREA: cambiare UNA SOLA variabile alla volta tra stage.
    # (v3 violava questa regola in Stage 3: density+duration insieme → stallo)
    #
    # Progressione:
    #   Stage 0: 2 corsie, poche auto → impara la meccanica base (schivare)
    #   Stage 1: 3 corsie, stesse auto → impara navigazione multi-corsia
    #   Stage 2: 3 corsie, più auto   → gestisce traffico su 3 corsie
    #   Stage 3: 3 corsie, densità+   → traffico più fitto
    #   Stage 4: 3 corsie, durata+    → sopravvive a lungo
    #   Stage 5: 3 corsie, aggressive → gestisce imprevedibilità
    #   Stage 6: 4 corsie, denso+aggressivo → scenario complesso
    DIFFICULTY_STAGES = [
        {   # Stage 0: 2 corsie, 8 auto — IMPARA A SCHIVARE
            # 2 corsie + 8 macchine = incontri frequenti, DEVE cambiare corsia.
            # Durata corta (30s = max 60 step) = facile sopravvivere.
            'vehicles_count': 8, 'vehicles_density': 0.8,
            'lanes_count': 2, 'initial_spacing': 2.0, 'duration': 30,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        {   # Stage 1: 3 CORSIE — nuova abilità chiave: navigazione multi-corsia
            # Stesse auto (8), stessa durata (30s), ma 3 corsie.
            # L'agente deve imparare a sfruttare la corsia extra per sorpassare.
            # UNA variabile cambiata: lanes_count 2 → 3.
            'vehicles_count': 8, 'vehicles_density': 0.8,
            'lanes_count': 3, 'initial_spacing': 2.0, 'duration': 30,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        {   # Stage 2: 3 corsie, PIÙ auto — traffico moderato su 3 corsie
            # Ora che sa navigare su 3 corsie, aumenta il traffico.
            # UNA variabile: vehicles_count 8 → 15.
            'vehicles_count': 15, 'vehicles_density': 0.8,
            'lanes_count': 3, 'initial_spacing': 2.0, 'duration': 30,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        {   # Stage 3: 3 corsie, DENSITÀ AUMENTATA — traffico più fitto
            # Stesse auto e durata, ma più vicine tra loro.
            # UNA variabile: vehicles_density 0.8 → 1.0.
            'vehicles_count': 15, 'vehicles_density': 1.0,
            'lanes_count': 3, 'initial_spacing': 2.0, 'duration': 30,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        {   # Stage 4: 3 corsie, DURATA AUMENTATA — sopravvivere a lungo
            # Ora che gestisce traffico denso, deve farlo PIÙ A LUNGO.
            # UNA variabile: duration 30 → 50.
            'vehicles_count': 15, 'vehicles_density': 1.0,
            'lanes_count': 3, 'initial_spacing': 2.0, 'duration': 50,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        {   # Stage 5: 3 corsie, traffico AGGRESSIVO — imprevedibilità
            # Stesse condizioni ma macchine aggressive: cambi corsia improvvisi.
            # UNA variabile: IDMVehicle → AggressiveVehicle (+ più auto).
            'vehicles_count': 20, 'vehicles_density': 1.0,
            'lanes_count': 3, 'initial_spacing': 1.5, 'duration': 50,
            'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
        },
        {   # Stage 6: 4 corsie, denso + aggressivo + lungo — scenario finale
            # Scenario complesso: tante auto, aggressività, lunga durata.
            'vehicles_count': 30, 'vehicles_density': 1.5,
            'lanes_count': 4, 'initial_spacing': 1.5, 'duration': 60,
            'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
        },
    ]

    def __init__(self, base_config: Optional[Dict] = None, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        # Default: Stage 0 (facile) — la progressione parte dal basso
        self.base_config = base_config or self.DIFFICULTY_STAGES[0].copy()

    @property
    def num_stages(self) -> int:
        return len(self.DIFFICULTY_STAGES)

    def get_stage_config(self, stage_idx: int) -> Dict:
        """Ritorna la config completa per lo stage di difficoltà specificato."""
        idx = min(stage_idx, len(self.DIFFICULTY_STAGES) - 1)
        config = self.DIFFICULTY_STAGES[idx].copy()
        config.update(self.FIXED_PARAMS)
        return config

    def base_level(self) -> Dict:
        """Config base del livello (Stage 0 = facile)."""
        config = self.base_config.copy()
        config.update(self.FIXED_PARAMS)
        return config

    def random_level(self) -> Dict:
        """Genera un livello completamente random (Domain Randomization)."""
        config = {}
        config.update(self.FIXED_PARAMS)

        for param, spec in self.PARAM_SPACE.items():
            lo, hi = spec['range']
            if spec['type'] == 'int':
                config[param] = int(self.rng.randint(lo, hi + 1))
            else:
                config[param] = float(self.rng.uniform(lo, hi))

        return config

    def mutate_level(self, config: Dict, num_edits: int = 3,
                     stage_bounds: Optional[Dict] = None) -> Dict:
        """
        Muta un livello con perturbazioni ACCEL-style.
        
        Come nel paper DCD/ACCEL:
        - Seleziona num_edits parametri random
        - Applica perturbazione ±step in una direzione random
        - Clip ai range dello stage (se forniti) o al range globale
        
        Args:
            stage_bounds: Dict param → (lo, hi) per limitare le mutazioni
                          allo "spazio" dello stage corrente. Previene
                          chain-drift verso difficoltà Expert su Stage 0.
        """
        mutated = config.copy()
        params = list(self.PARAM_SPACE.keys())
        edit_params = self.rng.choice(params, size=min(num_edits, len(params)), replace=False)

        for param in edit_params:
            spec = self.PARAM_SPACE[param]
            direction = self.rng.choice([-1, 1])
            current = mutated.get(param, self.base_config.get(param, spec['range'][0]))
            new_val = current + direction * spec['step']

            # Clip ai bounds dello stage (se forniti) o al range globale
            if stage_bounds and param in stage_bounds:
                lo, hi = stage_bounds[param]
            else:
                lo, hi = spec['range']
            new_val = max(lo, min(hi, new_val))

            if spec['type'] == 'int':
                mutated[param] = int(round(new_val))
            else:
                mutated[param] = round(float(new_val), 2)

        # Assicura parametri fissi
        mutated.update(self.FIXED_PARAMS)
        return mutated

    def get_stage_param_bounds(self, stage_idx: int) -> Dict[str, Tuple]:
        """
        Ritorna i range validi per le mutazioni allo stage dato.
        
        Bounds STRETTI: le mutazioni possono variare solo di +/- 1 step
        attorno ai parametri dello stage CORRENTE. Non raggiungono mai
        la difficoltà dello stage successivo.
        
        Questo è fondamentale perché: se i bounds arrivano allo stage N+1,
        PLR preferirà i livelli più difficili (alto regret) e l'agente
        finisce per allenarsi su livelli che non può ancora affrontare.
        """
        idx = min(stage_idx, len(self.DIFFICULTY_STAGES) - 1)
        current = self.DIFFICULTY_STAGES[idx]

        bounds = {}
        for param, spec in self.PARAM_SPACE.items():
            curr_val = current.get(param, spec['range'][0])

            # Range: solo +/- 1 step attorno al valore corrente
            lo = max(spec['range'][0], curr_val - spec['step'])
            hi = min(spec['range'][1], curr_val + spec['step'])
            bounds[param] = (lo, hi)

        return bounds


# =============================================================================
#  ACCEL CALLBACK (SB3 Integration)
# =============================================================================

class ACCELCallback(BaseCallback):
    """
    Callback SB3 che implementa il loop ACCEL con progressione basata su mastery.
    
    Algoritmo ACCEL (Parker-Holder et al. 2022, DCD Framework):
    1. Il modello inizia sullo Stage 0 (facile: poche macchine, poco traffico)
    2. A OGNI fine episodio, per ciascun env si decide il prossimo livello:
       a) PLR Replay: campiona livello ad alto regret dal buffer (prob ~0.95)
       b) ACCEL Edit: muta un livello ad alto score (prob ~0.5 del rimanente)
       c) Stage Config: usa la config dello stage corrente
    3. Il PLR buffer traccia lo score (TD-error) di ogni livello visitato
    4. Quando l'agente padroneggia lo stage (alta survival rate), avanza al prossimo
    5. Ogni stage è più difficile: più macchine, più densità, meno corsie
    
    Differenze rispetto alla versione precedente:
    - Decisione per-env (ogni env può avere un livello diverso)
    - PLR replay effettivamente utilizzato (sample_replay_decision)
    - Difficoltà progressiva con stage espliciti (Easy → Expert)
    - Mastery basata su survival (lunghezza episodio), non su mediana mobile
    - Max episodi per stage per prevenire overfitting
    """

    def __init__(
        self,
        vec_env,
        sampler: LevelSampler,
        generator: ACCELGenerator,
        num_envs: int,
        use_accel: bool = True,
        level_editor_prob: float = 0.5,
        num_edits: int = 3,
        warmup_episodes: int = 500,
        mastery_threshold: float = 0.85,
        mastery_window: int = 50,
        mastery_min_episodes: int = 50,
        max_episodes_per_stage: int = 500,
        log_interval: int = 25,
        save_dir: Optional[str] = None,
        save_interval: int = 50000,
        use_fast_env: bool = True,
        verbose: int = 1,
        start_stage: int = 0,
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.sampler = sampler
        self.generator = generator
        self.num_envs = num_envs
        self.use_accel = use_accel
        self.level_editor_prob = level_editor_prob
        self.num_edits = num_edits
        self.warmup_episodes = warmup_episodes
        self.mastery_threshold = mastery_threshold
        self.mastery_window = mastery_window
        self.mastery_min_episodes = mastery_min_episodes
        self.max_episodes_per_stage = max_episodes_per_stage
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.use_fast_env = use_fast_env

        # RNG per decisioni ACCEL
        self.rng = np.random.RandomState(42)

        # Stato per-env: quale livello sta eseguendo ogni env
        self.env_level_seeds = [None] * num_envs

        # Tracking globale
        self.total_episodes = 0
        self.all_returns = deque(maxlen=5000)
        self.all_lengths = deque(maxlen=5000)
        self.warmup_done = False

        # === STAGE TRACKING (Curriculum Progressivo) ===
        # current_stage è impostato più sotto in base a start_stage
        self.stage_episodes: List[Dict] = []           # episodi dello stage [{return, length}]
        self.stages_mastered: List[Dict] = []          # storico stages padroneggiati
        self.current_stage_seed: Optional[int] = None  # seed del livello base dello stage
        self.best_return_ever: float = float('-inf')
        self.last_checkpoint_step: int = 0

        # ---- Eval episodes: mastery basata SOLO su episodi sulla config pura dello stage ----
        # Problema risolto: prima TUTTI gli episodi (inclusi PLR replay su livelli
        # impossibili mutati) venivano contati per la mastery → survival sempre basso.
        # Ora: 20% degli episodi sono sulla config pura dello stage, e SOLO quelli
        # vengono usati per la mastery. Gli altri episodi servono per il TRAINING.
        self.eval_episodes: List[Dict] = []  # Solo episodi su pure stage config
        self.env_is_eval = [False] * num_envs  # Flag per-env
        self.eval_fraction = 0.3  # 30% episodi su config pura per mastery

        # ---- COMPETENCE GATE ----
        # L'agente deve dimostrare competenza BASE prima che PLR/ACCEL si attivino.
        # Senza questa protezione, PLR crea una death spiral: l'agente fallisce
        # su livelli mutati → alto regret → più replay su quei livelli → ancora fallisce.
        # Con il gate: sotto il 50% survival, 100% episodi su config pura dello stage.
        self.competence_survival = 0.0  # Survival corrente (aggiornata ogni N episodi)
        self.competence_gate_threshold = 0.50  # Sotto questo, NO PLR/ACCEL
        self.competence_full_threshold = 0.75  # Sopra questo, PLR/ACCEL a pieno regime
        self.competence_check_interval = 25  # Controlla ogni N episodi

        # ---- STALL DETECTION ----
        # Se il survival non migliora per N eval episodes, entra in focus-mode:
        # più training sulla config pura, meno distrazioni da PLR.
        self.stall_patience = 200  # eval episodes senza miglioramento
        self.stall_best_survival = 0.0  # miglior survival visto sullo stage
        self.stall_eval_at_best = 0  # n_eval quando abbiamo visto il best
        self.in_focus_mode = False  # se True, eval_fraction sale al 60%

        # Timing
        self.start_time = None
        self.pbar = None

        # Inizia dallo stage specificato (default: 0)
        self.current_stage = min(start_stage, generator.num_stages - 1)
        stage_config = generator.get_stage_config(self.current_stage)
        self.current_stage_seed = sampler.add_level(stage_config)
        for i in range(num_envs):
            self.env_level_seeds[i] = self.current_stage_seed

        # Se start_stage > 0, salta il warmup (l'agente sa già guidare)
        if self.current_stage > 0:
            self.warmup_done = True
            self.warmup_episodes = 0  # nessun warmup necessario

        # Pre-seed PLR buffer con varianti dello stage iniziale (con bounds!)
        stage_bounds = generator.get_stage_param_bounds(self.current_stage)
        for i in range(min(20, sampler.buffer_size)):
            variant = generator.mutate_level(stage_config, num_edits=2, stage_bounds=stage_bounds)
            sampler.add_level(variant, parent_seed=self.current_stage_seed)

    def _on_training_start(self):
        self.start_time = time.time()
        total = self.locals.get('total_timesteps', 0)
        self.pbar = tqdm(total=total, desc="DQN+ACCEL", unit="step")

    def _on_step(self) -> bool:
        if self.pbar:
            self.pbar.update(self.num_envs)

        # Estrai TD-error dal logger DQN
        current_loss = 0.0
        if hasattr(self.model, 'logger') and self.model.logger:
            if 'train/loss' in self.model.logger.name_to_value:
                current_loss = self.model.logger.name_to_value['train/loss']

        # Controlla episodi completati
        infos = self.locals.get('infos', [])
        for env_idx, info in enumerate(infos):
            if 'episode' in info:
                ep_return = info['episode']['r']
                ep_length = info['episode']['l']

                self.total_episodes += 1
                self.all_returns.append(ep_return)
                self.all_lengths.append(ep_length)
                self.stage_episodes.append({'return': ep_return, 'length': ep_length})

                # Se questo episodio era su config pura dello stage → conta per mastery
                # Durante il warmup, TUTTI gli episodi sono su config pura → contano tutti
                if self.env_is_eval[env_idx] or not self.warmup_done:
                    self.eval_episodes.append({'return': ep_return, 'length': ep_length})
                    self.env_is_eval[env_idx] = False

                # Aggiorna best return
                if ep_return > self.best_return_ever:
                    self.best_return_ever = ep_return

                # Calcola score per PLR (Positive Value Loss regret proxy)
                # PVL (Parker-Holder et al. 2022): score = max(0, V(s) - return)
                # Livelli dove l'agente va peggio del previsto → regret alto → più replay
                level_seed = self.env_level_seeds[env_idx]
                if level_seed is not None:
                    mean_return = np.mean(list(self.all_returns)[-100:]) if len(self.all_returns) > 1 else 0
                    pvl_score = max(0.0, mean_return - ep_return)  # Solo deficit (non surplus)
                    # Bonus per varianza alta: livelli inconsistenti = alto learning potential
                    level = self.sampler.levels.get(level_seed)
                    if level and len(level.returns) >= 3:
                        pvl_score += np.std(level.returns[-5:]) * 0.3
                    td_proxy = pvl_score + current_loss * 0.1  # TD-error come contributo secondario
                    self.sampler.update_score(level_seed, td_proxy, ep_return)

                # Dopo il warmup, curriculum ACCEL attivo
                if self.total_episodes >= self.warmup_episodes:
                    if not self.warmup_done:
                        self.warmup_done = True
                        self._update_competence()  # Calcola competence con dati del warmup
                        if self.verbose:
                            print(f"\n{'='*55}")
                            print(f" WARMUP COMPLETO ({self.warmup_episodes} ep)")
                            print(f" Curriculum ACCEL attivo — Stage 0/{self.generator.num_stages-1}")
                            print(f" Soglia mastery: {self.mastery_threshold*100:.0f}% survival")
                            print(f" Competence post-warmup: {self.competence_survival*100:.0f}%")
                            print(f"{'='*55}")

                    # Controlla mastery dello stage corrente
                    if self._check_mastery():
                        self._advance_stage()

                    # ACCEL: decidi il prossimo livello per QUESTO env (vero algoritmo DCD)
                    next_seed, next_config = self._decide_next_level(env_idx)
                    self.env_level_seeds[env_idx] = next_seed
                    try:
                        self.vec_env.env_method('set_next_config', next_config, indices=[env_idx])
                    except Exception as e:
                        if self.verbose > 1:
                            print(f"[WARN] set_next_config env {env_idx}: {e}")

                # Log periodico
                if self.total_episodes % self.log_interval == 0:
                    self._log_progress()

        # Salvataggio periodico
        if self.save_dir and self.n_calls > 0 and self.n_calls % (self.save_interval // self.num_envs) == 0:
            self._save_checkpoint()

        return True

    def _decide_next_level(self, env_idx: int) -> Tuple[int, Dict]:
        """
        Decisione ACCEL per-env con COMPETENCE GATE + FOCUS MODE.
        
        Fase 1 (survival < 50%): SOLO config pura dello stage.
          L'agente deve prima imparare a guidare senza crash.
          PLR/ACCEL sono completamente disabilitati.
        
        Fase 2 (survival 50-75%): PLR graduale.
          PLR si attiva con probabilità proporzionale alla competenza.
          Mutazioni limitate e conservative.
        
        Fase 3 (survival > 75%): PLR/ACCEL a pieno regime.
          Come il paper originale: replay 95%, mutazioni ACCEL.
        
        Focus mode (stallo): eval_fraction sale al 60%, PLR dimezzato.
          L'agente si concentra sulla config pura per sbloccarsi.
        """
        # ---- Aggiorna competence ogni N episodi ----
        if (len(self.eval_episodes) > 0 and 
            len(self.eval_episodes) % self.competence_check_interval == 0):
            self._update_competence()

        # ---- Eval fraction dinamico ----
        current_eval_fraction = 0.60 if self.in_focus_mode else self.eval_fraction

        # ---- 0. EVAL: config pura dello stage per mastery tracking ----
        # Sempre attivo: serve per misurare la vera capacità dell'agente
        if self.rng.random() < current_eval_fraction:
            config = self.generator.get_stage_config(self.current_stage)
            self.env_is_eval[env_idx] = True
            return self.current_stage_seed, config

        # ---- COMPETENCE GATE: sotto soglia, SOLO stage config ----
        if self.competence_survival < self.competence_gate_threshold:
            # L'agente non sa ancora guidare. Niente PLR, niente mutazioni.
            config = self.generator.get_stage_config(self.current_stage)
            return self.current_stage_seed, config

        # ---- PLR/ACCEL scaling basato su competenza ----
        # Scala lineare: 50% survival → PLR 0%, 75% survival → PLR 95%
        competence_ratio = min(1.0, max(0.0,
            (self.competence_survival - self.competence_gate_threshold) /
            (self.competence_full_threshold - self.competence_gate_threshold)
        ))
        # In focus mode, dimezza PLR/ACCEL
        if self.in_focus_mode:
            competence_ratio *= 0.5
        effective_replay_prob = self.sampler.replay_prob * competence_ratio
        effective_editor_prob = self.level_editor_prob * competence_ratio

        stage_bounds = self.generator.get_stage_param_bounds(self.current_stage)

        # 1. PLR Replay — campiona livello ad alto regret dal buffer
        if self.sampler.is_warm and self.rng.random() < effective_replay_prob:
            seed = self.sampler.sample_replay_level()
            config = self.sampler.levels[seed].config
            # Filtra: se il livello è fuori dai bounds dello stage, scartalo
            if self._is_within_stage_bounds(config, stage_bounds):
                self.sampler.stats['replay_count'] += 1
                return seed, config
            # Livello troppo difficile → fallback a stage config

        # 2. ACCEL Edit — muta un livello CON bounds dello stage
        if self.use_accel and self.rng.random() < effective_editor_prob:
            if self.sampler.is_warm:
                parent_seed = self.sampler.sample_replay_level()
                parent_config = self.sampler.levels[parent_seed].config
            else:
                parent_config = self.generator.get_stage_config(self.current_stage)
                parent_seed = None
            new_config = self.generator.mutate_level(
                parent_config, self.num_edits, stage_bounds=stage_bounds
            )
            new_seed = self.sampler.add_level(new_config, parent_seed=parent_seed)
            self.sampler.stats['mutation_count'] += 1
            return new_seed, new_config

        # 3. Stage config corrente (baseline dello stage)
        config = self.generator.get_stage_config(self.current_stage)
        return self.current_stage_seed, config

    def _update_competence(self):
        """Aggiorna il livello di competenza basato sugli eval episodes recenti.
        
        Metrica CONTINUA: misura QUANTO l'agente sopravvive, non solo SE sopravvive.
        Un episodio di 50/60 step vale 0.83 (prima valeva 0 perché < 51 step soglia).
        Questo permette al competence gate di aprirsi proporzionalmente ai progressi.
        """
        if len(self.eval_episodes) < 10:
            self.competence_survival = 0.0
            return

        recent = self.eval_episodes[-self.mastery_window:]
        stage_config = self.generator.get_stage_config(self.current_stage)
        expected_duration = stage_config.get('duration', 60)
        policy_freq = stage_config.get('policy_frequency', 2)
        expected_steps = expected_duration * policy_freq

        # Continuo: proporzione media di episodio completata (0.0 - 1.0)
        survival_scores = [min(1.0, ep['length'] / expected_steps) for ep in recent]
        self.competence_survival = float(np.mean(survival_scores))

    def _is_within_stage_bounds(self, config: Dict, bounds: Dict) -> bool:
        """Verifica che una config sia entro i bounds dello stage corrente."""
        for param, (lo, hi) in bounds.items():
            val = config.get(param)
            if val is not None and (val < lo or val > hi):
                return False
        return True

    def _check_mastery(self) -> bool:
        """
        Verifica se lo stage corrente è padroneggiato.
        
        Filosofia: NESSUN soft-advance. L'agente DEVE raggiungere la soglia.
        Se è in stallo, entra in focus-mode (più training su config pura)
        ma non avanza mai senza vera competenza.
        """
        n_eval = len(self.eval_episodes)
        n_total = len(self.stage_episodes)

        if n_eval < self.mastery_min_episodes:
            return False

        recent = self.eval_episodes[-self.mastery_window:]

        # Metrica CONTINUA: media della proporzione di episodio completata
        # Invece di binario (sopravvissuto sì/no con soglia 85% durata),
        # misura QUANTO dell'episodio è stato completato.
        # Es: 50/60 step = 0.83, 30/60 = 0.50, 60/60 = 1.00
        # Mastery a 0.85 significa: in media l'agente completa l'85% dell'episodio.
        stage_config = self.generator.get_stage_config(self.current_stage)
        expected_duration = stage_config.get('duration', 60)
        policy_freq = stage_config.get('policy_frequency', 2)
        expected_steps = expected_duration * policy_freq

        survival_scores = [min(1.0, ep['length'] / expected_steps) for ep in recent]
        survival_rate = float(np.mean(survival_scores))

        # ---- Stall detection: se survival non migliora, entra in focus mode ----
        if survival_rate > self.stall_best_survival + 0.02:  # miglioramento > 2%
            self.stall_best_survival = survival_rate
            self.stall_eval_at_best = n_eval
            if self.in_focus_mode:
                self.in_focus_mode = False
                if self.verbose:
                    print(f"\n  [FOCUS OFF] Survival migliorato a {survival_rate*100:.0f}% "
                          f"→ esco dal focus mode")

        stall_duration = n_eval - self.stall_eval_at_best
        if stall_duration >= self.stall_patience and not self.in_focus_mode:
            self.in_focus_mode = True
            if self.verbose:
                print(f"\n  [FOCUS ON] Stallo da {stall_duration} eval ep "
                      f"(survival={survival_rate*100:.0f}%, best={self.stall_best_survival*100:.0f}%) "
                      f"→ focus mode: 60% eval, PLR ridotto")

        # ---- Mastery piena: soglia raggiunta ----
        if survival_rate >= self.mastery_threshold:
            return True

        # ---- Nessun soft-advance: l'agente DEVE padroneggiare lo stage ----
        if self.verbose and n_total >= 500 and n_total % 200 == 0:
            print(f"\n  [STAGE {self.current_stage}] {n_total} ep totali, {n_eval} eval "
                  f"— survival {survival_rate*100:.0f}% / target {self.mastery_threshold*100:.0f}% "
                  f"{'[FOCUS]' if self.in_focus_mode else ''}")

        return False

    @property
    def mastery_info(self) -> Dict:
        """Informazioni sullo stato di mastery (basata su eval episodes)."""
        if not self.eval_episodes:
            return {'survival_rate': 0, 'avg_return': 0, 'eval_episodes': 0,
                    'total_episodes': len(self.stage_episodes)}
        recent = self.eval_episodes[-self.mastery_window:]
        stage_config = self.generator.get_stage_config(self.current_stage)
        expected_duration = stage_config.get('duration', 60)
        policy_freq = stage_config.get('policy_frequency', 2)
        expected_steps = expected_duration * policy_freq
        # Metrica continua: proporzione media di episodio completata
        survival_scores = [min(1.0, ep['length'] / expected_steps) for ep in recent]
        return {
            'survival_rate': float(np.mean(survival_scores)) * 100,
            'avg_return': np.mean([ep['return'] for ep in recent]),
            'eval_episodes': len(self.eval_episodes),
            'total_episodes': len(self.stage_episodes),
        }

    def _advance_stage(self):
        """
        Stage padroneggiato: avanza allo stage di difficoltà successivo.
        
        Ogni stage ha una config di ancoraggio (DIFFICULTY_STAGES).
        La nuova config viene aggiunta al buffer PLR e tutti gli env
        vengono aggiornati al nuovo stage come punto di partenza.
        """
        old_stage = self.current_stage
        recent_eps = self.stage_episodes[-self.mastery_window:]
        avg_return = np.mean([ep['return'] for ep in recent_eps]) if recent_eps else 0

        # Salva lo stage padroneggiato nello storico
        self.stages_mastered.append({
            'stage': old_stage,
            'avg_return': float(avg_return),
            'episodes_on_stage': len(self.stage_episodes),
            'total_episodes': self.total_episodes,
            'config': self.generator.get_stage_config(old_stage),
        })

        # Avanza (o resta sull'ultimo se già al massimo)
        self.current_stage = min(old_stage + 1, self.generator.num_stages - 1)

        if self.verbose:
            new_cfg = self.generator.get_stage_config(self.current_stage)
            print(f"\n{'*'*55}")
            print(f" ★ STAGE {old_stage} PADRONEGGIATO!")
            print(f"   Reward media: {avg_return:.2f}")
            print(f"   Episodi sullo stage: {len(self.stage_episodes)}")
            if self.current_stage > old_stage:
                print(f"   → Avanzo allo Stage {self.current_stage}")
                print(f"   Config: vehicles={new_cfg.get('vehicles_count')}, "
                      f"density={new_cfg.get('vehicles_density', 1.0):.1f}, "
                      f"lanes={new_cfg.get('lanes_count')}, "
                      f"duration={new_cfg.get('duration')}")
            else:
                print(f"   STAGE FINALE — continuo su Stage {self.current_stage}")
            print(f"{'*'*55}")

        # Resetta tracking per il nuovo stage
        self.stage_episodes = []
        self.eval_episodes = []
        self.competence_survival = 0.0  # Resetta competence per il nuovo stage
        self.stall_best_survival = 0.0  # Resetta stall detection
        self.stall_eval_at_best = 0
        self.in_focus_mode = False

        # === CHECKPOINT: salva modello al completamento dello stage ===
        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)
            step = self.n_calls * self.num_envs
            tag = f"stage{old_stage}_done_step{step}"
            self.model.save(str(path / f"{tag}.zip"))
            self.sampler.save(str(path / f"plr_{tag}.pkl"))
            self.model.save(str(path / "checkpoint_latest.zip"))
            self.sampler.save(str(path / "plr_latest.pkl"))
            resume_state = {
                'step': step,
                'total_episodes': self.total_episodes,
                'current_stage': self.current_stage,
                'stages_mastered': self.stages_mastered,
                'best_return': float(self.best_return_ever),
                'tag': tag,
                'timestamp': datetime.now().isoformat(),
            }
            with open(path / "checkpoint_info.json", 'w') as f:
                json.dump(resume_state, f, indent=2)
            with open(path / f"{tag}_info.json", 'w') as f:
                json.dump(resume_state, f, indent=2)
            if self.verbose:
                print(f"  [CHECKPOINT] Stage {old_stage} salvato: {tag}.zip")

        # Aggiungi config del nuovo stage al buffer PLR + varianti con bounds
        new_config = self.generator.get_stage_config(self.current_stage)
        self.current_stage_seed = self.sampler.add_level(new_config)
        stage_bounds = self.generator.get_stage_param_bounds(self.current_stage)
        for i in range(10):
            variant = self.generator.mutate_level(new_config, num_edits=2, stage_bounds=stage_bounds)
            self.sampler.add_level(variant, parent_seed=self.current_stage_seed)

        # Aggiorna tutti gli env al nuovo stage come punto di partenza
        for i in range(self.num_envs):
            self.env_level_seeds[i] = self.current_stage_seed
            try:
                self.vec_env.env_method('set_next_config', new_config, indices=[i])
            except Exception as e:
                if self.verbose > 1:
                    print(f"[WARN] set_next_config env {i}: {e}")

    def _log_progress(self):
        if not self.verbose:
            return

        elapsed = time.time() - self.start_time if self.start_time else 0
        sps = self.n_calls * self.num_envs / elapsed if elapsed > 0 else 0

        print(f"\n{'='*65}")
        print(f" Step {self.n_calls * self.num_envs:>8,} | Ep: {self.total_episodes:>5} | "
              f"Tempo: {elapsed/60:.1f}min | SPS: {sps:.0f}")
        print(f"{'-'*65}")

        if len(self.all_returns) > 0:
            recent = list(self.all_returns)[-50:]
            best = max(self.all_returns)
            print(f" Reward (last 50):  {np.mean(recent):>7.2f} ± {np.std(recent):.2f}")
            print(f" Reward (best):     {best:>7.2f}")
            if len(self.all_lengths) > 0:
                print(f" Lunghezza media:   {np.mean(list(self.all_lengths)[-50:]):>7.0f} steps")

        mode = "WARMUP (stage 0 fisso)" if not self.warmup_done else "ACCEL CURRICULUM"
        if self.warmup_done:
            if self.competence_survival < self.competence_gate_threshold:
                mode += f" [GATE: survival={self.competence_survival*100:.0f}%<{self.competence_gate_threshold*100:.0f}%]"
            elif self.competence_survival < self.competence_full_threshold:
                cr = (self.competence_survival - self.competence_gate_threshold) / (
                    self.competence_full_threshold - self.competence_gate_threshold)
                mode += f" [PLR {cr*100:.0f}%]"
            if self.in_focus_mode:
                mode += " [FOCUS]"
        print(f" Modalità:          {mode}")

        # Stage progress (mastery basata SOLO su eval episodes)
        mi = self.mastery_info
        stage_max = self.generator.num_stages - 1
        print(f" Stage attuale:     {self.current_stage}/{stage_max} "
              f"(eval={mi['eval_episodes']}/{mi['total_episodes']} ep, "
              f"avg={mi['avg_return']:.1f}, "
              f"survival={mi['survival_rate']:.0f}%, "
              f"target={self.mastery_threshold*100:.0f}%)")
        print(f" Stages completati: {len(self.stages_mastered)}")

        # Config dello stage
        stage_cfg = self.generator.get_stage_config(self.current_stage)
        print(f" Config stage:      vehicles={stage_cfg.get('vehicles_count')}, "
              f"density={stage_cfg.get('vehicles_density', 1.0):.1f}, "
              f"lanes={stage_cfg.get('lanes_count')}, "
              f"duration={stage_cfg.get('duration', 60)}")

        # PLR buffer stats
        plr = self.sampler.get_stats()
        print(f" PLR Buffer:        {plr['buffer_size']}/{plr['buffer_capacity']} "
              f"(replay={self.sampler.stats.get('replay_count', 0)}, "
              f"mutations={self.sampler.stats.get('mutation_count', 0)})")

        print(f"{'='*65}")

    def _save_checkpoint(self):
        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)

            step = self.n_calls * self.num_envs

            # Salva con nome unico (non sovrascrive mai)
            self.model.save(str(path / f"checkpoint_step{step}.zip"))
            self.sampler.save(str(path / f"plr_step{step}.pkl"))

            # Salva anche come 'latest' per ripristino rapido
            self.model.save(str(path / "checkpoint_latest.zip"))
            self.sampler.save(str(path / "plr_latest.pkl"))

            # Stats completi per ripristino
            stats = {
                'step': step,
                'total_episodes': self.total_episodes,
                'current_stage': self.current_stage,
                'stages_mastered': self.stages_mastered,
                'best_return': float(self.best_return_ever),
                'avg_return_50': float(np.mean(list(self.all_returns)[-50:])) if self.all_returns else 0,
                'timestamp': datetime.now().isoformat(),
            }
            with open(path / "checkpoint_info.json", 'w') as f:
                json.dump(stats, f, indent=2)

            if self.verbose:
                print(f"\n  [CHECKPOINT] Step {step:,} salvato in {path}")

            self.last_checkpoint_step = step

    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.sampler.save(str(path / "plr_final.pkl"))

            stats = {
                'total_episodes': self.total_episodes,
                'total_steps': self.n_calls * self.num_envs,
                'returns': [float(r) for r in self.all_returns],
                'lengths': [int(l) for l in self.all_lengths],
                'plr_stats': self.sampler.get_stats(),
                'warmup_episodes': self.warmup_episodes,
                'use_accel': self.use_accel,
                'stages_mastered': self.stages_mastered,
                'final_stage': self.current_stage,
                'best_return_ever': float(self.best_return_ever),
                'timestamp': datetime.now().isoformat(),
            }
            with open(path / "training_stats.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            print(f"\n[ACCEL] Statistiche salvate in: {path / 'training_stats.json'}")
            print(f"[ACCEL] Stages padroneggiati: {len(self.stages_mastered)}")
            print(f"[ACCEL] Stage finale: {self.current_stage}/{self.generator.num_stages - 1}")


# =============================================================================
#  BEST MODEL CALLBACK (protezione contro collapse)
# =============================================================================

class BestModelCallback(BaseCallback):
    """
    Salva il modello quando la reward media (su una finestra) migliora.
    Previene la perdita di progressi in caso di collapse:
    best_model.zip contiene sempre il miglior modello visto.
    """

    def __init__(self, save_path: str, check_freq: int = 1000, window: int = 50, verbose: int = 1):
        super().__init__(verbose)
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.check_freq = check_freq
        self.window = window
        self.best_mean_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.saves_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])
                self.episode_lengths.append(info['episode']['l'])

        if len(self.episode_rewards) >= self.window and self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.episode_rewards[-self.window:])
            mean_length = np.mean(self.episode_lengths[-self.window:])

            if mean_reward > self.best_mean_reward:
                improvement = mean_reward - self.best_mean_reward
                self.best_mean_reward = mean_reward
                self.saves_count += 1
                self.model.save(str(self.save_path / "best_model"))

                info_data = {
                    "step": self.n_calls,
                    "mean_reward": float(mean_reward),
                    "mean_length": float(mean_length),
                    "improvement": float(improvement),
                    "saves_count": self.saves_count,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(self.save_path / "best_model_info.json", "w") as f:
                    json.dump(info_data, f, indent=2)

                if self.verbose:
                    print(f"\n  \u2605 BEST MODEL salvato! Reward: {mean_reward:.2f} (+{improvement:.2f}) "
                          f"| Len: {mean_length:.0f} | Step: {self.n_calls:,}")

        return True


# =============================================================================
#  MAIN TRAINING FUNCTION
# =============================================================================

def train_dqn_accel(
    total_timesteps: int = 500_000,
    num_envs: int = 8,
    use_accel: bool = True,
    use_fast_env: bool = True,
    # Modello pre-trainato
    pretrained_path: Optional[str] = None,
    # PLR hyperparams (DCD defaults)
    plr_buffer_size: int = 4000,
    plr_replay_prob: float = 0.95,
    plr_temperature: float = 0.1,
    plr_staleness_coef: float = 0.3,
    plr_rho: float = 0.1,
    # ACCEL hyperparams
    level_editor_prob: float = 0.5,
    num_edits: int = 3,
    warmup_episodes: int = 500,
    # Mastery hyperparams
    mastery_threshold: float = 0.80,
    mastery_window: int = 50,
    mastery_min_episodes: int = 50,
    max_episodes_per_stage: int = 500,
    # DQN hyperparams (ottimizzati per highway-env)
    learning_rate: float = 5e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.95,
    train_freq: int = 4,
    gradient_steps: int = 8,
    target_update_interval: int = 250,   # Allineato a baseline (50 era troppo aggressivo)
    exploration_fraction: float = 0.3,   # 30% dei timestep in esplorazione
    exploration_final_eps: float = 0.05,
    net_arch: Optional[List[int]] = None,
    # Generale
    save_dir: str = './dqn_accel_models',
    device: str = 'auto',
    seed: int = 42,
    verbose: int = 1,
    # Configurazione base dell'env (come la baseline dell'utente)
    base_env_config: Optional[Dict] = None,
    # Stage di partenza (per riprendere da uno stage specifico)
    start_stage: int = 0,
    # Proximity reward (distanza di sicurezza) — disabilitato di default
    safety_distance: float = 25.0,
    proximity_penalty_val: float = 0.0,
) -> Tuple[DQN, LevelSampler]:
    """
    Training DQN + ACCEL per highway-env.
    
    Returns:
        (modello_trainato, level_sampler)
    """
    if net_arch is None:
        net_arch = [256, 256]

    env_id = "highway-fast-v0" if use_fast_env else "highway-v0"

    print(f"\n{'='*65}")
    print(f"  DQN + {'ACCEL' if use_accel else 'Robust PLR'} Training")
    print(f"{'='*65}")
    print(f"  Environment:     {env_id}")
    print(f"  Timesteps:       {total_timesteps:,}")
    print(f"  Parallel envs:   {num_envs}")
    print(f"  PLR buffer:      {plr_buffer_size}")
    print(f"  Replay prob:     {plr_replay_prob}")
    accel_info = f"ON (prob={level_editor_prob}, edits={num_edits})" if use_accel else "OFF"
    print(f"  ACCEL mutations: {accel_info}")
    print(f"  Warmup:          {warmup_episodes} episodi")
    print(f"  Mastery:         soglia={mastery_threshold}, finestra={mastery_window} ep, max={max_episodes_per_stage} ep/stage")
    print(f"  Stages:          {len(ACCELGenerator.DIFFICULTY_STAGES)} (Trivial → Expert)")
    if start_stage > 0:
        print(f"  Start stage:     {start_stage} (salta warmup, curriculum attivo subito)")
    prox_status = f"ON (safety_dist={safety_distance}m, max_penalty={proximity_penalty_val})" if proximity_penalty_val > 0 else "OFF (segnale pulito: solo speed + crash)"
    print(f"  Proximity reward: {prox_status}")
    print(f"  Competence gate: ON (PLR disabilitato sotto {50}% survival)")
    if pretrained_path:
        print(f"  Pre-trained:     {pretrained_path}")
    else:
        print(f"  Pre-trained:     Nessuno (training da zero)")
    print(f"  DQN lr:          {learning_rate}")
    print(f"  DQN batch_size:  {batch_size}")
    print(f"  DQN buffer:      {buffer_size:,}")
    print(f"  DQN train_freq:  {train_freq}")
    print(f"  DQN gamma:       {gamma}")
    print(f"  Net arch:        {net_arch}")
    print(f"{'='*65}\n")

    # Setup
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Generatore livelli ACCEL
    generator = ACCELGenerator(base_config=base_env_config, seed=seed)

    # PLR Sampler
    sampler = LevelSampler(
        buffer_size=plr_buffer_size,
        replay_prob=plr_replay_prob,
        score_transform='rank',
        temperature=plr_temperature,
        staleness_coef=plr_staleness_coef,
        rho=plr_rho,
        seed=seed,
    )

    # Configurazione stage di partenza e stage finale
    effective_start = min(start_stage, generator.num_stages - 1)
    base_config = generator.get_stage_config(effective_start)
    print(f"\nStage {effective_start} (partenza): lanes={base_config.get('lanes_count')}, "
          f"vehicles={base_config.get('vehicles_count')}, "
          f"density={base_config.get('vehicles_density', 1.0):.1f}, "
          f"duration={base_config.get('duration')}")
    final_config = generator.get_stage_config(generator.num_stages - 1)
    print(f"Stage {generator.num_stages-1} (finale):   lanes={final_config.get('lanes_count')}, "
          f"vehicles={final_config.get('vehicles_count')}, "
          f"density={final_config.get('vehicles_density', 1.0):.1f}, "
          f"duration={final_config.get('duration')}")

    # SubprocVecEnv = vero parallelismo (ogni env in un processo separato)
    # ~6x speedup rispetto a DummyVecEnv con 6 envs
    def _make_env(idx, config=base_config):
        def _init():
            import highway_env as _henv  # Registra env nel subprocess
            env = ConfigurableHighwayEnv(
                env_id=env_id,
                initial_config=config,
                safety_distance=safety_distance,
                proximity_penalty=proximity_penalty_val,
            )
            return Monitor(env)
        return _init

    vec_env_type = "SubprocVecEnv"
    try:
        vec_env = SubprocVecEnv([_make_env(i) for i in range(num_envs)])
    except Exception as e:
        if verbose:
            print(f"  SubprocVecEnv non disponibile ({e}), fallback DummyVecEnv")
        vec_env = DummyVecEnv([_make_env(i) for i in range(num_envs)])
        vec_env_type = "DummyVecEnv"
    print(f"Creati {num_envs} environment paralleli ({vec_env_type})\n")

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # Crea o carica modello DQN
    if pretrained_path:
        print(f"\nCaricamento modello pre-trainato: {pretrained_path}")
        model = DQN.load(
            pretrained_path,
            env=vec_env,
            device=device,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=0,  # Inizia subito (il modello sa già guidare)
            batch_size=batch_size,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            target_update_interval=target_update_interval,
            exploration_fraction=min(exploration_fraction, 0.3),  # Pretrained: meno esplorazione
            exploration_final_eps=exploration_final_eps,
            exploration_initial_eps=0.15,  # Bassa: il modello sa già cosa fare
            verbose=0,
            tensorboard_log=str(save_path / 'tensorboard'),
            seed=seed,
        )
        print(f"  Modello caricato! Esplorazione iniziale ridotta a 0.15")
    else:
        model = DQN(
            'MlpPolicy',
            vec_env,
            policy_kwargs=dict(net_arch=net_arch),
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
            tensorboard_log=str(save_path / 'tensorboard'),
            device=device,
            seed=seed,
        )

    # Callback ACCEL
    callback = ACCELCallback(
        vec_env=vec_env,
        sampler=sampler,
        generator=generator,
        num_envs=num_envs,
        use_accel=use_accel,
        level_editor_prob=level_editor_prob,
        num_edits=num_edits,
        warmup_episodes=warmup_episodes,
        mastery_threshold=mastery_threshold,
        mastery_window=mastery_window,
        mastery_min_episodes=mastery_min_episodes,
        max_episodes_per_stage=max_episodes_per_stage,
        log_interval=25,
        save_dir=str(save_path),
        save_interval=25000,
        use_fast_env=use_fast_env,
        verbose=verbose,
        start_stage=effective_start,
    )

    # Best model callback (protezione contro collapse)
    best_callback = BestModelCallback(
        save_path=str(save_path),
        check_freq=1000,
        window=50,
        verbose=verbose,
    )

    # Training
    print("Avvio training...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[callback, best_callback])
    training_time = time.time() - t0

    # Salva modello
    model_path = save_path / "dqn_accel_final.zip"
    model.save(str(model_path))

    # Salva config di training
    training_config = {
        'algorithm': 'DQN',
        'method': 'ACCEL' if use_accel else 'Robust PLR',
        'pretrained_path': pretrained_path,
        'total_timesteps': total_timesteps,
        'num_envs': num_envs,
        'training_time_seconds': training_time,
        'env_id': env_id,
        'base_env_config': base_config,
        'dqn_params': {
            'learning_rate': learning_rate,
            'buffer_size': buffer_size,
            'batch_size': batch_size,
            'gamma': gamma,
            'train_freq': train_freq,
            'net_arch': net_arch,
        },
        'plr_params': {
            'buffer_size': plr_buffer_size,
            'replay_prob': plr_replay_prob,
            'temperature': plr_temperature,
            'staleness_coef': plr_staleness_coef,
        },
        'accel_params': {
            'level_editor_prob': level_editor_prob,
            'num_edits': num_edits,
        },
        'proximity_reward': {
            'safety_distance': safety_distance,
            'proximity_penalty': proximity_penalty_val,
        },
        'warmup_episodes': warmup_episodes,
        'timestamp': datetime.now().isoformat(),
    }
    with open(save_path / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Training completato!")
    print(f"  Tempo: {training_time/60:.1f} minuti")
    print(f"  Modello: {model_path}")
    print(f"{'='*65}\n")

    vec_env.close()
    return model, sampler


# =============================================================================
#  EVALUATION
# =============================================================================

def evaluate_model(
    model_path: str,
    n_episodes: int = 50,
    device: str = 'auto',
    render: bool = True,
    use_metrics_tracker: bool = True,
) -> Dict:
    """
    Valuta il modello DQN su configurazioni diverse per testare la generalizzazione.
    """
    from metrics_tracker import evaluate as mt_evaluate, HighwayMetrics

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DQN.load(model_path, device=device)

    # Configurazioni di test (usa stessi FIXED_PARAMS + DIFFICULTY_STAGES del training)
    fixed = ACCELGenerator.FIXED_PARAMS.copy()
    test_configs = [
        {
            'name': 'Easy (Stage 0)',
            'config': {**fixed, 'lanes_count': 3, 'vehicles_count': 10, 'vehicles_density': 0.8, 'duration': 40},
        },
        {
            'name': 'Medium (Stage 2)',
            'config': {**fixed, 'lanes_count': 4, 'vehicles_count': 22, 'vehicles_density': 1.0, 'duration': 60},
        },
        {
            'name': 'Hard (Stage 4)',
            'config': {**fixed, 'lanes_count': 3, 'vehicles_count': 40, 'vehicles_density': 1.6, 'duration': 70},
        },
        {
            'name': 'Expert (Stage 5)',
            'config': {**fixed, 'lanes_count': 3, 'vehicles_count': 50, 'vehicles_density': 2.0, 'duration': 80},
        },
    ]

    all_results = {}

    for tc in test_configs:
        name = tc['name']
        config = tc['config']

        print(f"\n{'='*55}")
        print(f" Testing: {name}")
        print(f" Config: lanes={config['lanes_count']}, vehicles={config['vehicles_count']}, "
              f"density={config['vehicles_density']}")
        print(f"{'='*55}")

        eval_env_id = "highway-v0" if render else "highway-fast-v0"  # fast per coerenza col training
        env = gymnasium.make(eval_env_id, config=config, render_mode='rgb_array')

        if use_metrics_tracker:
            metrics_to_use = {
                'collision_rate', 'survival_rate', 'avg_reward',
                'cars_overtaken', 'avg_speed', 'max_speed',
                'distance_traveled', 'lane_changes',
            }
            results = mt_evaluate(
                model=model, env=env, n_episodes=n_episodes,
                metrics=metrics_to_use, render=render, verbose=False, seed=42,
            )
        else:
            # Valutazione semplice
            returns = []
            for ep in tqdm(range(n_episodes), desc=name):
                obs, _ = env.reset()
                done = False
                ep_return = 0
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_return += reward
                returns.append(ep_return)
            results = {
                'avg_reward': np.mean(returns),
                'std_reward': np.std(returns),
            }

        all_results[name] = results
        env.close()

    # Summary
    print(f"\n{'='*65}")
    print(f"{'EVALUATION SUMMARY':^65}")
    print(f"{'='*65}")
    for name, results in all_results.items():
        reward = results.get('avg_reward', 0)
        survival = results.get('survival_rate', 0)
        print(f" {name:.<25} Reward: {reward:>7.2f} | Survival: {survival:>5.1f}%")
    print(f"{'='*65}\n")

    return all_results


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN + ACCEL Training for Highway-Env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Training con modello pre-trainato (RACCOMANDATO)
  python dqn_accel.py --pretrained src/highway_dqn/model.zip --timesteps 500000

  # Training da zero (più lento)
  python dqn_accel.py --timesteps 500000 --num-envs 6

  # Mastery più aggressiva (avanza prima)
  python dqn_accel.py --pretrained src/highway_dqn/model.zip --mastery-threshold 0.7

  # Valutazione modello
  python dqn_accel.py --eval-only ./dqn_accel_models/dqn_accel_final.zip
        """
    )

    # Pre-trained model
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path al modello DQN pre-trainato (.zip)')

    # Training
    parser.add_argument('--timesteps', type=int, default=500_000,
                        help='Timesteps totali (default: 500000)')
    parser.add_argument('--num-envs', type=int, default=8,
                        help='Numero di environment paralleli (default: 8)')
    parser.add_argument('--accel', dest='use_accel', action='store_true', default=True,
                        help='Usa ACCEL (PLR + mutazioni) [default]')
    parser.add_argument('--no-accel', dest='use_accel', action='store_false',
                        help='Solo Robust PLR (senza mutazioni)')
    parser.add_argument('--fast-env', dest='use_fast', action='store_true', default=True,
                        help='Usa highway-fast-v0 (15x velocità) [default]')
    parser.add_argument('--no-fast-env', dest='use_fast', action='store_false',
                        help='Usa highway-v0 standard (più lento)')

    # PLR
    parser.add_argument('--plr-buffer', type=int, default=4000,
                        help='Dimensione buffer PLR (default: 4000)')
    parser.add_argument('--replay-prob', type=float, default=0.95,
                        help='Probabilità replay (default: 0.95)')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Episodi di warmup (default: 500)')

    # Mastery
    parser.add_argument('--mastery-threshold', type=float, default=0.80,
                        help='Soglia mastery (0-1, default: 0.80 = 80%% survival)')
    parser.add_argument('--mastery-window', type=int, default=50,
                        help='Episodi finestra per valutare mastery (default: 50)')
    parser.add_argument('--mastery-min-ep', type=int, default=50,
                        help='Minimo episodi per stage prima di avanzare (default: 50)')
    parser.add_argument('--max-ep-per-stage', type=int, default=500,
                        help='Max episodi per stage (default: 500, poi avanza)')
    parser.add_argument('--start-stage', type=int, default=0,
                        help='Stage di partenza (default: 0). Se >0, salta il warmup.')

    # Proximity reward (distanza di sicurezza)
    parser.add_argument('--safety-distance', type=float, default=25.0,
                        help='Distanza di sicurezza in metri (default: 25.0). '
                             'Sotto questa distanza si attiva la penalità.')
    parser.add_argument('--proximity-penalty', type=float, default=0.0,
                        help='Penalità massima per prossimità (default: 0.0 = OFF). '
                             'Se attivata (es. 0.3): a 10m ~0.11/step, a 5m ~0.19/step.')

    # ACCEL
    parser.add_argument('--editor-prob', type=float, default=0.5,
                        help='Probabilità mutazione livello (default: 0.5)')
    parser.add_argument('--num-edits', type=int, default=3,
                        help='Numero edit per mutazione (default: 3)')

    # DQN
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate DQN (default: 5e-4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size DQN (default: 64)')
    parser.add_argument('--buffer', type=int, default=100_000,
                        help='Replay buffer size (default: 100000)')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='Discount factor (default: 0.95)')
    parser.add_argument('--train-freq', type=int, default=4,
                        help='Training frequency (default: 4)')

    # Generale
    parser.add_argument('--save-dir', type=str, default='./dqn_accel_models',
                        help='Directory per salvare (default: ./dqn_accel_models)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'],
                        help='Device (default: auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosità')

    # Evaluation
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path al modello per sola valutazione')
    parser.add_argument('--eval-episodes', type=int, default=50,
                        help='Episodi di valutazione (default: 50)')

    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path a file JSON di configurazione (sovrascrive CLI)')

    args = parser.parse_args()

    # Carica config da file se specificato
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
        print(f"Config caricata da: {args.config}")
        # Sovrascrivi args con valori dal config
        for key, val in config.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), val)

    if args.eval_only:
        # Modalità valutazione
        results = evaluate_model(
            model_path=args.eval_only,
            n_episodes=args.eval_episodes,
            device=args.device,
        )

        # Salva risultati
        results_path = Path(args.eval_only).parent / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Risultati salvati: {results_path}")
    else:
        # Modalità training
        model, sampler = train_dqn_accel(
            total_timesteps=args.timesteps,
            num_envs=args.num_envs,
            use_accel=args.use_accel,
            use_fast_env=args.use_fast,
            pretrained_path=args.pretrained,
            plr_buffer_size=args.plr_buffer,
            plr_replay_prob=args.replay_prob,
            level_editor_prob=args.editor_prob,
            num_edits=args.num_edits,
            warmup_episodes=args.warmup,
            mastery_threshold=args.mastery_threshold,
            mastery_window=args.mastery_window,
            mastery_min_episodes=args.mastery_min_ep,
            max_episodes_per_stage=args.max_ep_per_stage,
            learning_rate=args.lr,
            buffer_size=args.buffer,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            save_dir=args.save_dir,
            device=args.device,
            seed=args.seed,
            verbose=args.verbose,
            start_stage=args.start_stage,
            safety_distance=args.safety_distance,
            proximity_penalty_val=args.proximity_penalty,
        )

        print(f"\nPer valutare il modello:")
        print(f"  python src/dqn_accel.py --eval-only {args.save_dir}/dqn_accel_final.zip")
