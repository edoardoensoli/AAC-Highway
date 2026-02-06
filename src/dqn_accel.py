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
    Wrapper per highway-env che supporta il cambio di configurazione lazy.
    
    La nuova configurazione viene applicata al prossimo reset(),
    evitando di ricreare l'env durante un episodio attivo.
    Questo è il pattern corretto per PLR/ACCEL dove i livelli cambiano
    solo tra episodi.
    """

    def __init__(self, env_id: str = "highway-fast-v0", initial_config: Optional[Dict] = None):
        self.env_id = env_id
        self._config = initial_config or {}
        self._next_config = None

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
        return self.env.step(action)


def _get_configurable_env(vec_env: DummyVecEnv, idx: int) -> ConfigurableHighwayEnv:
    """Accede al ConfigurableHighwayEnv dentro DummyVecEnv -> Monitor -> Wrapper."""
    monitor = vec_env.envs[idx]
    if isinstance(monitor, Monitor):
        return monitor.env
    return monitor


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
    PARAM_SPACE = {
        'vehicles_count':  {'type': 'int',   'range': (10, 50),   'step': 5},
        'vehicles_density': {'type': 'float', 'range': (0.5, 2.0), 'step': 0.2},
        'lanes_count':     {'type': 'int',   'range': (3, 6),     'step': 1},
        'initial_spacing': {'type': 'float', 'range': (1.0, 3.0), 'step': 0.5},
        'duration':        {'type': 'int',   'range': (30, 80),   'step': 10},
    }

    # Parametri fissi (reward e simulazione - mai mutati per consistenza)
    FIXED_PARAMS = {
        'simulation_frequency': 15,
        'policy_frequency': 1,
        'collision_reward': -1.0,
        'high_speed_reward': 0.4,
        'right_lane_reward': 0.1,
        'reward_speed_range': [20, 30],
        'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
    }

    def __init__(self, base_config: Optional[Dict] = None, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.base_config = base_config or {
            'lanes_count': 4,
            'vehicles_count': 25,
            'vehicles_density': 1.0,
            'initial_spacing': 2,
            'duration': 60,
        }

    def base_level(self) -> Dict:
        """Config base del livello (uguale alla baseline dell'utente)."""
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

    def mutate_level(self, config: Dict, num_edits: int = 3) -> Dict:
        """
        Muta un livello con perturbazioni ACCEL-style.
        
        Come nel paper DCD/ACCEL:
        - Seleziona num_edits parametri random
        - Applica perturbazione ±step in una direzione random
        - Clip ai range validi
        """
        mutated = config.copy()
        params = list(self.PARAM_SPACE.keys())
        edit_params = self.rng.choice(params, size=min(num_edits, len(params)), replace=False)

        for param in edit_params:
            spec = self.PARAM_SPACE[param]
            direction = self.rng.choice([-1, 1])
            current = mutated.get(param, self.base_config.get(param, spec['range'][0]))
            new_val = current + direction * spec['step']

            # Clip al range valido
            new_val = max(spec['range'][0], min(spec['range'][1], new_val))

            if spec['type'] == 'int':
                mutated[param] = int(round(new_val))
            else:
                mutated[param] = round(float(new_val), 2)

        # Assicura parametri fissi
        mutated.update(self.FIXED_PARAMS)
        return mutated


# =============================================================================
#  ACCEL CALLBACK (SB3 Integration)
# =============================================================================

class ACCELCallback(BaseCallback):
    """
    Callback SB3 che implementa il loop ACCEL con progressione basata su mastery.
    
    Logica:
    1. Il modello (pre-trainato) inizia sul livello base
    2. Traccia la reward rolling media per livello corrente
    3. Quando la reward supera la soglia di mastery per N episodi consecutivi,
       il livello è considerato padroneggiato
    4. Solo allora ACCEL muta il livello (rendendolo un po' più difficile)
    5. Il PLR buffer tiene traccia dei livelli con alto learning potential
       per riproporli se il modello regredisce
    
    Questo evita il problema di cambiare livello troppo spesso,
    dando al modello tempo di consolidare ciò che impara.
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
        warmup_episodes: int = 50,
        mastery_threshold: float = 0.85,
        mastery_window: int = 15,
        mastery_min_episodes: int = 10,
        log_interval: int = 25,
        save_dir: Optional[str] = None,
        save_interval: int = 50000,
        use_fast_env: bool = True,
        verbose: int = 1,
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
        self.log_interval = log_interval
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.use_fast_env = use_fast_env

        # Stato per-env: quale livello sta eseguendo ogni env
        self.env_level_seeds = [None] * num_envs

        # Tracking globale
        self.total_episodes = 0
        self.all_returns = deque(maxlen=2000)
        self.all_lengths = deque(maxlen=2000)
        self.warmup_done = False

        # === MASTERY TRACKING ===
        # Per-livello: rolling window di returns per misurare la mastery
        self.current_level_returns: List[float] = []  # returns del livello corrente
        self.current_level_seed: Optional[int] = None  # seed del livello condiviso
        self.current_difficulty: int = 0  # contatore di livelli superati
        self.levels_mastered: List[Dict] = []  # storico livelli padroneggiati
        self.best_return_ever: float = float('-inf')
        self.last_checkpoint_step: int = 0

        # Timing
        self.start_time = None
        self.pbar = None

        # Aggiungi livello base e assegnalo a tutti gli env
        base_config = generator.base_level()
        base_seed = sampler.add_level(base_config)
        self.current_level_seed = base_seed
        for i in range(num_envs):
            self.env_level_seeds[i] = base_seed

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
                self.current_level_returns.append(ep_return)

                # Aggiorna best return
                if ep_return > self.best_return_ever:
                    self.best_return_ever = ep_return

                # Calcola score per PLR
                level_seed = self.env_level_seeds[env_idx]
                if level_seed is not None:
                    mean_return = np.mean(list(self.all_returns)[-100:]) if len(self.all_returns) > 1 else 0
                    return_deviation = abs(ep_return - mean_return)
                    td_proxy = max(current_loss, return_deviation)
                    self.sampler.update_score(level_seed, td_proxy, ep_return)

                # Dopo il warmup, controlla mastery e gestisci progressione
                if self.total_episodes >= self.warmup_episodes:
                    if not self.warmup_done:
                        self.warmup_done = True
                        if self.verbose:
                            print(f"\n{'='*55}")
                            print(f" WARMUP COMPLETO ({self.warmup_episodes} ep)")
                            print(f" Curriculum ACCEL con mastery attivo")
                            print(f" Soglia mastery: {self.mastery_threshold*100:.0f}% del max")
                            print(f"{'='*55}")

                    # Controlla se il livello corrente è padroneggiato
                    if self._check_mastery():
                        self._advance_level()

                # Log periodico
                if self.total_episodes % self.log_interval == 0:
                    self._log_progress()

        # Salvataggio periodico (ogni save_interval steps di callback)
        if self.save_dir and self.n_calls > 0 and self.n_calls % (self.save_interval // self.num_envs) == 0:
            self._save_checkpoint()

        return True

    def _check_mastery(self) -> bool:
        """
        Verifica se il livello corrente è padroneggiato.
        
        Usa una soglia basata sulla SOPRAVVIVENZA: in highway-env se sopravvivi
        fino alla fine dell'episodio ottieni reward alta. Se crashi, reward bassa.
        
        Condizioni:
        1. Almeno mastery_min_episodes episodi sul livello
        2. La percentuale di episodi "buoni" supera la soglia di mastery
           (un episodio è "buono" se il return supera la mediana globale)
        3. Performance consistente (pochi crash recenti)
        """
        if len(self.current_level_returns) < self.mastery_min_episodes:
            return False

        recent = self.current_level_returns[-self.mastery_window:]
        avg = np.mean(recent)

        # Soglia basata sul 50° percentile di TUTTI i returns visti finora
        # Se non c'è abbastanza storia, usa un minimo ragionevole
        if len(self.all_returns) >= 20:
            p50 = np.percentile(list(self.all_returns), 50)
        else:
            p50 = 15.0  # fallback conservativo

        # Survival threshold: un episodio "buono" = return > p50
        good_episodes = sum(1 for r in recent if r > p50)
        survival_rate = good_episodes / len(recent)

        # Mastery = alta % di episodi buoni (es. 85% = max 2 crash su 15)
        is_mastered = survival_rate >= self.mastery_threshold

        return is_mastered

    @property
    def mastery_info(self) -> Dict:
        """Informazioni sullo stato di mastery corrente per il logging."""
        if not self.current_level_returns:
            return {'survival_rate': 0, 'threshold_return': 0, 'avg': 0}
        recent = self.current_level_returns[-self.mastery_window:]
        if len(self.all_returns) >= 20:
            p50 = np.percentile(list(self.all_returns), 50)
        else:
            p50 = 15.0
        good = sum(1 for r in recent if r > p50)
        return {
            'survival_rate': good / len(recent) * 100,
            'threshold_return': p50,
            'avg': np.mean(recent),
        }

    def _advance_level(self):
        """
        Il livello è padroneggiato: avanza al prossimo.
        
        ACCEL: muta il livello corrente per renderlo un po' più difficile.
        PLR: oppure campiona un livello con alto learning potential dal buffer.
        """
        self.current_difficulty += 1
        avg_return = np.mean(self.current_level_returns[-self.mastery_window:])

        # Salva il livello padroneggiato
        self.levels_mastered.append({
            'difficulty': self.current_difficulty - 1,
            'seed': self.current_level_seed,
            'avg_return': float(avg_return),
            'episodes_on_level': len(self.current_level_returns),
            'total_episodes': self.total_episodes,
        })

        if self.verbose:
            print(f"\n{'*'*55}")
            print(f" ★ LIVELLO {self.current_difficulty-1} PADRONEGGIATO!")
            print(f"   Reward media: {avg_return:.2f}")
            print(f"   Episodi sul livello: {len(self.current_level_returns)}")
            print(f"   Avanzo al livello {self.current_difficulty}...")
            print(f"{'*'*55}")

        # Resetta tracking per il nuovo livello
        self.current_level_returns = []

        # Decidi come generare il prossimo livello
        old_seed = self.current_level_seed
        old_config = self.sampler.levels[old_seed].config.copy() if old_seed in self.sampler.levels else self.generator.base_level()

        if self.use_accel:
            # ACCEL: muta il livello corrente per renderlo leggermente più difficile
            new_config = self.generator.mutate_level(old_config, self.num_edits)
            new_seed = self.sampler.add_level(new_config, parent_seed=old_seed)
            self.sampler.stats['mutation_count'] += 1
        else:
            # PLR puro: campiona dal buffer il livello con più learning potential
            if self.sampler.is_warm:
                new_seed = self.sampler.sample_replay_level()
                new_config = self.sampler.levels[new_seed].config.copy()
                self.sampler.stats['replay_count'] += 1
            else:
                new_config = self.generator.random_level()
                new_seed = self.sampler.add_level(new_config)
                self.sampler.stats['new_count'] += 1

        self.current_level_seed = new_seed

        # Aggiorna TUTTI gli env al nuovo livello (tutti lavorano sullo stesso)
        for i in range(self.num_envs):
            self.env_level_seeds[i] = new_seed
            try:
                self.vec_env.env_method('set_next_config', new_config, indices=[i])
            except Exception as e:
                if self.verbose > 1:
                    print(f"[WARN] Impossibile aggiornare env {i}: {e}")

        if self.verbose:
            print(f"   Nuovo livello: lanes={new_config.get('lanes_count')}, "
                  f"vehicles={new_config.get('vehicles_count')}, "
                  f"density={new_config.get('vehicles_density', 1.0):.1f}")

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

        mode = "WARMUP (livello fisso)" if not self.warmup_done else "ACCEL CURRICULUM"
        print(f" Modalità:  {mode}")

        # Mastery progress
        level_eps = len(self.current_level_returns)
        if level_eps > 0:
            mi = self.mastery_info
            print(f" Livello attuale:   #{self.current_difficulty} "
                  f"({level_eps} ep, avg={mi['avg']:.1f}, "
                  f"sopravvivenza={mi['survival_rate']:.0f}%, "
                  f"target={self.mastery_threshold*100:.0f}%)")
        print(f" Livelli superati:  {len(self.levels_mastered)}")

        # PLR buffer stats
        plr = self.sampler.get_stats()
        print(f" PLR Buffer: {plr['buffer_size']}/{plr['buffer_capacity']}")
        print(f" Mutazioni ACCEL: {plr.get('mutation_count', self.sampler.stats['mutation_count'])}")

        # Info livello corrente
        if self.current_level_seed and self.current_level_seed in self.sampler.levels:
            cfg = self.sampler.levels[self.current_level_seed].config
            print(f" Config attuale: lanes={cfg.get('lanes_count')}, "
                  f"vehicles={cfg.get('vehicles_count')}, "
                  f"density={cfg.get('vehicles_density', 1.0):.1f}, "
                  f"duration={cfg.get('duration', 60)}")

        print(f"{'='*65}")

    def _save_checkpoint(self):
        if self.save_dir:
            path = Path(self.save_dir)
            path.mkdir(parents=True, exist_ok=True)

            step = self.n_calls * self.num_envs

            # Salva modello DQN (checkpoint ripristinabile)
            self.model.save(str(path / f"checkpoint_step{step}.zip"))
            # Salva anche come 'latest' per ripristino rapido
            self.model.save(str(path / "checkpoint_latest.zip"))

            # Salva PLR sampler
            self.sampler.save(str(path / "plr_latest.pkl"))

            # Salva stats incrementali
            stats = {
                'step': step,
                'total_episodes': self.total_episodes,
                'current_difficulty': self.current_difficulty,
                'levels_mastered': len(self.levels_mastered),
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
                'levels_mastered': self.levels_mastered,
                'final_difficulty': self.current_difficulty,
                'best_return_ever': float(self.best_return_ever),
                'timestamp': datetime.now().isoformat(),
            }
            with open(path / "training_stats.json", 'w') as f:
                json.dump(stats, f, indent=2, default=str)

            print(f"\n[ACCEL] Statistiche salvate in: {path / 'training_stats.json'}")
            print(f"[ACCEL] Livelli padroneggiati: {len(self.levels_mastered)}")
            print(f"[ACCEL] Difficoltà finale: {self.current_difficulty}")


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
    warmup_episodes: int = 50,
    # Mastery hyperparams
    mastery_threshold: float = 0.85,
    mastery_window: int = 15,
    mastery_min_episodes: int = 10,
    # DQN hyperparams (ottimizzati per highway-env)
    learning_rate: float = 5e-4,
    buffer_size: int = 100_000,
    learning_starts: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.8,
    train_freq: int = 16,
    gradient_steps: int = 1,
    target_update_interval: int = 50,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.05,
    net_arch: Optional[List[int]] = None,
    # Generale
    save_dir: str = './dqn_accel_models',
    device: str = 'auto',
    seed: int = 42,
    verbose: int = 1,
    # Configurazione base dell'env (come la baseline dell'utente)
    base_env_config: Optional[Dict] = None,
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
    print(f"  Mastery:         soglia={mastery_threshold}, finestra={mastery_window} ep")
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

    # Configurazione base
    base_config = generator.base_level()
    print(f"Config base: lanes={base_config.get('lanes_count')}, "
          f"vehicles={base_config.get('vehicles_count')}, "
          f"density={base_config.get('vehicles_density', 1.0):.1f}")

    # SubprocVecEnv = vero parallelismo (ogni env in un processo separato)
    # ~6x speedup rispetto a DummyVecEnv con 6 envs
    def _make_env(idx, config=base_config):
        def _init():
            import highway_env as _henv  # Registra env nel subprocess
            env = ConfigurableHighwayEnv(env_id=env_id, initial_config=config)
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
            exploration_fraction=exploration_fraction,
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
        log_interval=25,
        save_dir=str(save_path),
        save_interval=25000,
        use_fast_env=use_fast_env,
        verbose=verbose,
    )

    # Training
    print("Avvio training...\n")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback)
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
    render: bool = False,
    use_metrics_tracker: bool = True,
) -> Dict:
    """
    Valuta il modello DQN su configurazioni diverse per testare la generalizzazione.
    """
    from metrics_tracker import evaluate as mt_evaluate, HighwayMetrics

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DQN.load(model_path, device=device)

    # Configurazioni di test (Easy -> Hard)
    test_configs = [
        {
            'name': 'Easy',
            'config': {
                'lanes_count': 4, 'vehicles_count': 15, 'vehicles_density': 0.8,
                'duration': 60, 'simulation_frequency': 15, 'policy_frequency': 1,
            }
        },
        {
            'name': 'Medium (Baseline)',
            'config': {
                'lanes_count': 4, 'vehicles_count': 25, 'vehicles_density': 1.0,
                'duration': 60, 'simulation_frequency': 15, 'policy_frequency': 1,
            }
        },
        {
            'name': 'Hard',
            'config': {
                'lanes_count': 3, 'vehicles_count': 35, 'vehicles_density': 1.5,
                'duration': 60, 'simulation_frequency': 15, 'policy_frequency': 1,
            }
        },
        {
            'name': 'Very Hard',
            'config': {
                'lanes_count': 3, 'vehicles_count': 45, 'vehicles_density': 2.0,
                'duration': 60, 'simulation_frequency': 15, 'policy_frequency': 1,
            }
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

        env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')

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
    parser.add_argument('--warmup', type=int, default=50,
                        help='Episodi di warmup (default: 50)')

    # Mastery
    parser.add_argument('--mastery-threshold', type=float, default=0.85,
                        help='Soglia mastery (0-1, default: 0.85 = 85%% del max)')
    parser.add_argument('--mastery-window', type=int, default=15,
                        help='Episodi finestra per valutare mastery (default: 15)')
    parser.add_argument('--mastery-min-ep', type=int, default=10,
                        help='Minimo episodi per livello prima di avanzare (default: 10)')

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
    parser.add_argument('--gamma', type=float, default=0.8,
                        help='Discount factor (default: 0.8)')
    parser.add_argument('--train-freq', type=int, default=16,
                        help='Training frequency (default: 16)')

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
            learning_rate=args.lr,
            buffer_size=args.buffer,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            save_dir=args.save_dir,
            device=args.device,
            seed=args.seed,
            verbose=args.verbose,
        )

        print(f"\nPer valutare il modello:")
        print(f"  python src/dqn_accel.py --eval-only {args.save_dir}/dqn_accel_final.zip")
