"""
Hyperparameter Tuner per DQN + ACCEL su Highway-Env
=====================================================
Rileva automaticamente le specifiche hardware (CPU, RAM, cache) e calcola
gli hyperparameter ottimali per il training DQN + ACCEL.

Funzionalità:
- Rilevamento hardware (cores, RAM, frequenza CPU, cache)
- Benchmark throughput dell'environment
- Calcolo automatico di: num_envs, batch_size, buffer_size, learning_rate, ecc.
- Output JSON pronto per dqn_accel.py (--config hyperparams.json)

Uso:
    python hyperparameter_tuner.py                  # Analisi + output config
    python hyperparameter_tuner.py --benchmark      # Include benchmark env
    python hyperparameter_tuner.py --output config.json  # Salva su file
    python hyperparameter_tuner.py --profile quick  # Profilo veloce (meno test)
"""

import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
#  HARDWARE DETECTION
# =============================================================================

@dataclass
class HardwareInfo:
    """Informazioni hardware del sistema."""
    # CPU
    cpu_name: str = "Unknown"
    physical_cores: int = 1
    logical_cores: int = 1
    cpu_freq_mhz: float = 0.0
    cpu_freq_max_mhz: float = 0.0
    # RAM
    total_ram_gb: float = 0.0
    available_ram_gb: float = 0.0
    # Cache (se disponibile)
    l1_cache_kb: int = 0
    l2_cache_kb: int = 0
    l3_cache_mb: int = 0
    # OS
    os_name: str = ""
    os_version: str = ""
    python_version: str = ""
    # Score complessivo per il sizing
    perf_tier: str = "medium"  # "low", "medium", "high", "ultra"

    def summary(self) -> str:
        lines = [
            f"{'='*55}",
            f"  HARDWARE PROFILE",
            f"{'='*55}",
            f"  CPU:           {self.cpu_name}",
            f"  Core fisici:   {self.physical_cores}",
            f"  Core logici:   {self.logical_cores}",
            f"  Frequenza:     {self.cpu_freq_mhz:.0f} MHz (max: {self.cpu_freq_max_mhz:.0f} MHz)",
            f"  RAM totale:    {self.total_ram_gb:.1f} GB",
            f"  RAM disponib.: {self.available_ram_gb:.1f} GB",
            f"  OS:            {self.os_name} {self.os_version}",
            f"  Python:        {self.python_version}",
            f"  Perf tier:     {self.perf_tier.upper()}",
            f"{'='*55}",
        ]
        if self.l3_cache_mb > 0:
            lines.insert(6, f"  L3 Cache:      {self.l3_cache_mb} MB")
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """Rileva le specifiche hardware del sistema."""
    info = HardwareInfo()

    # OS
    info.os_name = platform.system()
    info.os_version = platform.version()
    info.python_version = platform.python_version()

    # CPU cores
    info.logical_cores = os.cpu_count() or 1
    info.physical_cores = info.logical_cores

    # Prova ad ottenere info più dettagliate con psutil
    try:
        import psutil
        info.physical_cores = psutil.cpu_count(logical=False) or info.logical_cores
        info.logical_cores = psutil.cpu_count(logical=True) or info.logical_cores

        freq = psutil.cpu_freq()
        if freq:
            info.cpu_freq_mhz = freq.current
            info.cpu_freq_max_mhz = freq.max if freq.max else freq.current

        mem = psutil.virtual_memory()
        info.total_ram_gb = mem.total / (1024 ** 3)
        info.available_ram_gb = mem.available / (1024 ** 3)
    except ImportError:
        # Fallback senza psutil
        info.total_ram_gb = _get_ram_fallback()
        info.available_ram_gb = info.total_ram_gb * 0.6  # Stima conservativa

    # CPU name
    info.cpu_name = _get_cpu_name()

    # Cache info (best effort)
    cache = _get_cache_info()
    info.l1_cache_kb = cache.get('l1', 0)
    info.l2_cache_kb = cache.get('l2', 0)
    info.l3_cache_mb = cache.get('l3', 0)

    # Performance tier
    info.perf_tier = _calculate_perf_tier(info)

    return info


def _get_cpu_name() -> str:
    """Ottieni il nome della CPU."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'Name'],
                capture_output=True, text=True, timeout=5
            )
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip() != 'Name']
            if lines:
                return lines[0]
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        elif platform.system() == "Linux":
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if 'model name' in line:
                        return line.split(':')[1].strip()
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _get_ram_fallback() -> float:
    """RAM in GB senza psutil."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['wmic', 'ComputerSystem', 'get', 'TotalPhysicalMemory'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if line.isdigit():
                    return int(line) / (1024 ** 3)
        elif platform.system() == "Darwin":
            result = subprocess.run(
                ['sysctl', '-n', 'hw.memsize'],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip()) / (1024 ** 3)
        elif platform.system() == "Linux":
            with open('/proc/meminfo') as f:
                for line in f:
                    if 'MemTotal' in line:
                        kb = int(line.split()[1])
                        return kb / (1024 ** 2)
    except Exception:
        pass
    return 8.0  # Default conservativo


def _get_cache_info() -> Dict[str, int]:
    """Ottieni info sulla cache CPU (best effort)."""
    cache = {'l1': 0, 'l2': 0, 'l3': 0}
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'L2CacheSize,L3CacheSize'],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                values = lines[1].strip().split()
                if len(values) >= 1 and values[0].isdigit():
                    cache['l2'] = int(values[0])
                if len(values) >= 2 and values[1].isdigit():
                    cache['l3'] = int(values[1]) // 1024  # KB -> MB
        elif platform.system() == "Darwin":
            for level, key in [('l1', 'hw.l1dcachesize'), ('l2', 'hw.l2cachesize'), ('l3', 'hw.l3cachesize')]:
                result = subprocess.run(
                    ['sysctl', '-n', key],
                    capture_output=True, text=True, timeout=5
                )
                val = result.stdout.strip()
                if val.isdigit():
                    if level == 'l3':
                        cache[level] = int(val) // (1024 * 1024)
                    else:
                        cache[level] = int(val) // 1024
    except Exception:
        pass
    return cache


def _calculate_perf_tier(info: HardwareInfo) -> str:
    """Calcola il tier di performance basato sull'hardware."""
    score = 0

    # CPU cores (max 30 punti)
    score += min(info.physical_cores * 3, 30)

    # CPU frequency (max 20 punti)
    freq = info.cpu_freq_max_mhz or info.cpu_freq_mhz
    if freq >= 4000:
        score += 20
    elif freq >= 3000:
        score += 15
    elif freq >= 2000:
        score += 10
    else:
        score += 5

    # RAM (max 30 punti)
    if info.total_ram_gb >= 32:
        score += 30
    elif info.total_ram_gb >= 16:
        score += 20
    elif info.total_ram_gb >= 8:
        score += 10
    else:
        score += 5

    # Cache L3 (max 20 punti)
    if info.l3_cache_mb >= 16:
        score += 20
    elif info.l3_cache_mb >= 8:
        score += 15
    elif info.l3_cache_mb >= 4:
        score += 10

    if score >= 80:
        return "ultra"
    elif score >= 55:
        return "high"
    elif score >= 30:
        return "medium"
    else:
        return "low"


# =============================================================================
#  ENVIRONMENT BENCHMARK
# =============================================================================

@dataclass
class BenchmarkResult:
    """Risultati del benchmark dell'environment."""
    env_id: str = ""
    steps_per_second: float = 0.0
    episodes_per_minute: float = 0.0
    avg_episode_length: float = 0.0
    reset_time_ms: float = 0.0
    step_time_ms: float = 0.0
    total_steps: int = 0
    total_episodes: int = 0
    benchmark_time: float = 0.0

    def summary(self) -> str:
        return "\n".join([
            f"{'='*55}",
            f"  BENCHMARK: {self.env_id}",
            f"{'='*55}",
            f"  Steps/secondo:    {self.steps_per_second:,.0f}",
            f"  Episodi/minuto:   {self.episodes_per_minute:.1f}",
            f"  Step medio:       {self.step_time_ms:.3f} ms",
            f"  Reset medio:      {self.reset_time_ms:.3f} ms",
            f"  Lungh. episodio:  {self.avg_episode_length:.0f} steps",
            f"  Totale:           {self.total_steps:,} steps in {self.benchmark_time:.1f}s",
            f"{'='*55}",
        ])


def benchmark_environment(
    env_id: str = "highway-fast-v0",
    config: Optional[Dict] = None,
    duration_seconds: float = 15.0,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Esegue un benchmark dell'environment per misurare il throughput.
    """
    import gymnasium
    import highway_env

    if config is None:
        config = {
            'lanes_count': 4,
            'vehicles_count': 25,
            'vehicles_density': 1.0,
            'duration': 60,
            'simulation_frequency': 15,
            'policy_frequency': 1,
        }

    result = BenchmarkResult(env_id=env_id)

    if verbose:
        print(f"\nBenchmark {env_id} in corso ({duration_seconds}s)...")

    env = gymnasium.make(env_id, config=config, render_mode='rgb_array')

    total_steps = 0
    total_episodes = 0
    reset_times = []
    step_times = []
    episode_lengths = []

    t_start = time.perf_counter()

    while (time.perf_counter() - t_start) < duration_seconds:
        # Reset
        t0 = time.perf_counter()
        obs, _ = env.reset()
        reset_times.append(time.perf_counter() - t0)

        ep_steps = 0
        done = False
        while not done:
            action = env.action_space.sample()
            t1 = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            step_times.append(time.perf_counter() - t1)
            done = terminated or truncated
            ep_steps += 1
            total_steps += 1

        total_episodes += 1
        episode_lengths.append(ep_steps)

    t_end = time.perf_counter()
    result.benchmark_time = t_end - t_start

    env.close()

    result.total_steps = total_steps
    result.total_episodes = total_episodes
    result.steps_per_second = total_steps / result.benchmark_time if result.benchmark_time > 0 else 0
    result.episodes_per_minute = (total_episodes / result.benchmark_time) * 60 if result.benchmark_time > 0 else 0
    result.avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
    result.reset_time_ms = np.mean(reset_times) * 1000 if reset_times else 0
    result.step_time_ms = np.mean(step_times) * 1000 if step_times else 0

    if verbose:
        print(result.summary())

    return result


def benchmark_vec_env(
    num_envs_list: Optional[List[int]] = None,
    env_id: str = "highway-fast-v0",
    config: Optional[Dict] = None,
    steps_per_test: int = 2000,
    verbose: bool = True,
) -> Dict[int, float]:
    """
    Benchmark DummyVecEnv con diversi numeri di env paralleli.
    Trova il numero ottimale di env.
    """
    import gymnasium
    import highway_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    if config is None:
        config = {
            'lanes_count': 4, 'vehicles_count': 25, 'vehicles_density': 1.0,
            'duration': 60, 'simulation_frequency': 15, 'policy_frequency': 1,
        }

    if num_envs_list is None:
        max_envs = min(os.cpu_count() or 4, 16)
        num_envs_list = [1, 2, 4, 8]
        if max_envs >= 12:
            num_envs_list.append(12)
        if max_envs >= 16:
            num_envs_list.append(16)

    results = {}

    if verbose:
        print(f"\nBenchmark DummyVecEnv con {num_envs_list} envs...")

    for n in num_envs_list:
        def _make(cfg=config):
            env = gymnasium.make(env_id, config=cfg, render_mode='rgb_array')
            return Monitor(env)

        vec_env = DummyVecEnv([_make for _ in range(n)])
        obs = vec_env.reset()

        t0 = time.perf_counter()
        for _ in range(steps_per_test):
            actions = [vec_env.action_space.sample() for _ in range(n)]
            obs, rewards, dones, infos = vec_env.step(actions)
        dt = time.perf_counter() - t0

        sps = (steps_per_test * n) / dt
        results[n] = sps

        if verbose:
            print(f"  {n:>2} envs: {sps:>8,.0f} steps/sec ({dt:.2f}s)")

        vec_env.close()

    # Trova ottimale
    best_n = max(results, key=results.get)
    if verbose:
        print(f"\n  Ottimale: {best_n} envs ({results[best_n]:,.0f} steps/sec)")

    return results


# =============================================================================
#  HYPERPARAMETER CALCULATOR
# =============================================================================

@dataclass
class HyperparameterConfig:
    """Configurazione completa di hyperparameter calcolata."""
    # Training
    num_envs: int = 8
    total_timesteps: int = 500_000
    # DQN
    learning_rate: float = 5e-4
    buffer_size: int = 100_000
    learning_starts: int = 1000
    batch_size: int = 64
    gamma: float = 0.8
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 50
    exploration_fraction: float = 0.3
    exploration_final_eps: float = 0.05
    net_arch: List[int] = field(default_factory=lambda: [256, 256])
    # PLR
    plr_buffer_size: int = 4000
    plr_replay_prob: float = 0.95
    plr_temperature: float = 0.1
    plr_staleness_coef: float = 0.3
    plr_rho: float = 0.5
    # ACCEL
    use_accel: bool = True
    level_editor_prob: float = 0.5
    num_edits: int = 3
    warmup_episodes: int = 50
    # Env
    use_fast_env: bool = True
    # Meta
    perf_tier: str = "medium"
    estimated_time_minutes: float = 0.0

    def to_dict(self) -> Dict:
        return {
            'num_envs': self.num_envs,
            'timesteps': self.total_timesteps,
            'use_accel': self.use_accel,
            'use_fast': self.use_fast_env,
            'lr': self.learning_rate,
            'buffer': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'train_freq': self.train_freq,
            'plr_buffer': self.plr_buffer_size,
            'replay_prob': self.plr_replay_prob,
            'editor_prob': self.level_editor_prob,
            'num_edits': self.num_edits,
            'warmup': self.warmup_episodes,
        }

    def to_cli_args(self) -> str:
        """Genera i parametri CLI per dqn_accel.py."""
        args = [
            f"--timesteps {self.total_timesteps}",
            f"--num-envs {self.num_envs}",
            f"--lr {self.learning_rate}",
            f"--batch-size {self.batch_size}",
            f"--buffer {self.buffer_size}",
            f"--gamma {self.gamma}",
            f"--train-freq {self.train_freq}",
            f"--plr-buffer {self.plr_buffer_size}",
            f"--replay-prob {self.plr_replay_prob}",
            f"--editor-prob {self.level_editor_prob}",
            f"--num-edits {self.num_edits}",
            f"--warmup {self.warmup_episodes}",
        ]
        if not self.use_accel:
            args.append("--no-accel")
        if not self.use_fast_env:
            args.append("--no-fast-env")
        return "python src/dqn_accel.py " + " ".join(args)

    def summary(self) -> str:
        return "\n".join([
            f"{'='*55}",
            f"  HYPERPARAMETERS CALCOLATI ({self.perf_tier.upper()})",
            f"{'='*55}",
            f"  Parallel envs:     {self.num_envs}",
            f"  Timesteps:         {self.total_timesteps:,}",
            f"  Env:               {'highway-fast-v0' if self.use_fast_env else 'highway-v0'}",
            f"  ACCEL:             {'ON' if self.use_accel else 'OFF'}",
            f"  ---",
            f"  DQN lr:            {self.learning_rate}",
            f"  DQN batch:         {self.batch_size}",
            f"  DQN buffer:        {self.buffer_size:,}",
            f"  DQN train_freq:    {self.train_freq}",
            f"  DQN gamma:         {self.gamma}",
            f"  DQN net_arch:      {self.net_arch}",
            f"  ---",
            f"  PLR buffer:        {self.plr_buffer_size}",
            f"  PLR replay prob:   {self.plr_replay_prob}",
            f"  ACCEL editor prob: {self.level_editor_prob}",
            f"  ACCEL edits:       {self.num_edits}",
            f"  Warmup episodi:    {self.warmup_episodes}",
            f"  ---",
            f"  Tempo stimato:     ~{self.estimated_time_minutes:.0f} minuti",
            f"{'='*55}",
        ])


def calculate_hyperparameters(
    hw: HardwareInfo,
    benchmark: Optional[BenchmarkResult] = None,
    vec_benchmark: Optional[Dict[int, float]] = None,
    target_timesteps: Optional[int] = None,
) -> HyperparameterConfig:
    """
    Calcola gli hyperparameter ottimali basati sull'hardware e benchmark.
    
    Logica:
    - num_envs: basato su core fisici e benchmark throughput
    - buffer_size: scala con RAM disponibile
    - batch_size: scala con cache L3 e cores
    - learning_rate: stabile, non dipende dall'hardware
    - train_freq: ottimizzato per bilanciare data collection vs training
    """
    config = HyperparameterConfig()
    config.perf_tier = hw.perf_tier

    # ---- NUM ENVS ----
    # Per highway-env con DummyVecEnv (GIL bound), il throughput migliore
    # è tipicamente con 4-8 envs. Più envs = più data per step ma overhead.
    if vec_benchmark:
        # Usa il benchmark per scegliere il numero ottimale
        config.num_envs = max(vec_benchmark, key=vec_benchmark.get)
    else:
        # Stima basata su hardware
        tier_envs = {
            'low': 4,
            'medium': 6,
            'high': 8,
            'ultra': 12,
        }
        config.num_envs = tier_envs.get(hw.perf_tier, 8)

    # Limita a un multiplo ragionevole dei core
    config.num_envs = min(config.num_envs, max(hw.physical_cores, 4))

    # ---- TIMESTEPS ----
    if target_timesteps:
        config.total_timesteps = target_timesteps
    else:
        tier_timesteps = {
            'low': 200_000,
            'medium': 500_000,
            'high': 1_000_000,
            'ultra': 2_000_000,
        }
        config.total_timesteps = tier_timesteps.get(hw.perf_tier, 500_000)

    # ---- BUFFER SIZE ----
    # Buffer DQN: ~100 byte per transizione (obs + action + reward + next_obs + done)
    # highway-env obs: 5*5 = 25 float32 = 100 byte + overhead = ~200 byte
    # 100k transizioni ≈ 20 MB. Scala con RAM.
    max_buffer_by_ram = int(hw.available_ram_gb * 1024 * 1024 * 1024 * 0.1 / 200)  # 10% della RAM
    tier_buffer = {
        'low': 50_000,
        'medium': 100_000,
        'high': 200_000,
        'ultra': 500_000,
    }
    config.buffer_size = min(tier_buffer.get(hw.perf_tier, 100_000), max_buffer_by_ram)

    # ---- BATCH SIZE ----
    # Batch più grandi = gradient estimates migliori, ma più memoria.
    # Per CPU, batch 64-128 è ottimale (fits in L3 cache).
    if hw.l3_cache_mb >= 12:
        config.batch_size = 128
    elif hw.l3_cache_mb >= 6:
        config.batch_size = 64
    else:
        config.batch_size = 32

    # ---- LEARNING STARTS ----
    # Deve raccogliere abbastanza dati prima di iniziare. Con num_envs env paralleli,
    # raccogliamo num_envs transizioni per step.
    config.learning_starts = max(500, config.buffer_size // 100)

    # ---- TRAIN FREQ ----
    # Con DummyVecEnv, ogni step raccoglie num_envs transizioni.
    # train_freq=4 significa 1 gradient update ogni 4*num_envs transizioni.
    config.train_freq = 4

    # ---- NET ARCH ----
    # [256, 256] è lo standard per highway-env DQN. Reti più grandi non servono.
    config.net_arch = [256, 256]

    # ---- PLR BUFFER ----
    # Più grande = più diversità di livelli nel curriculum.
    # Con più steps, possiamo esplorare più livelli.
    if config.total_timesteps >= 1_000_000:
        config.plr_buffer_size = 5000
    elif config.total_timesteps >= 500_000:
        config.plr_buffer_size = 4000
    else:
        config.plr_buffer_size = 2000

    # ---- WARMUP ----
    # Più episodi di warmup = baseline più stabile prima del curriculum.
    # Con num_envs env, gli episodi si completano più velocemente.
    config.warmup_episodes = max(30, min(100, config.total_timesteps // 5000))

    # ---- EXPLORATION ----
    # Exploration fraction: parte iniziale con alta esplorazione.
    config.exploration_fraction = 0.3
    config.exploration_final_eps = 0.05

    # ---- TEMPO STIMATO ----
    if benchmark:
        sps = benchmark.steps_per_second * config.num_envs * 0.3  # 0.3x per overhead DQN training
        config.estimated_time_minutes = (config.total_timesteps / sps) / 60 if sps > 0 else 0
    else:
        # Stima conservativa basata su tier
        tier_sps = {'low': 200, 'medium': 400, 'high': 800, 'ultra': 1500}
        sps = tier_sps.get(hw.perf_tier, 400)
        config.estimated_time_minutes = (config.total_timesteps / sps) / 60

    return config


# =============================================================================
#  MAIN TUNER FLOW
# =============================================================================

def run_tuner(
    run_benchmark: bool = True,
    run_vec_benchmark: bool = False,
    target_timesteps: Optional[int] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> HyperparameterConfig:
    """
    Esegue il tuner completo e restituisce la configurazione ottimale.
    
    Steps:
    1. Rileva hardware
    2. (Opzionale) Benchmark environment
    3. (Opzionale) Benchmark DummyVecEnv scaling
    4. Calcola hyperparameters
    5. Salva configurazione
    """
    # 1. Hardware
    if verbose:
        print("\n[1/4] Rilevamento hardware...")
    hw = detect_hardware()
    if verbose:
        print(hw.summary())

    # 2. Benchmark singolo env
    bench = None
    if run_benchmark:
        if verbose:
            print("\n[2/4] Benchmark environment...")
        try:
            bench = benchmark_environment(
                env_id="highway-fast-v0",
                duration_seconds=10.0,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                print(f"  Benchmark fallito: {e}")
                print("  Usando stime conservative.")
    else:
        if verbose:
            print("\n[2/4] Benchmark saltato (usa --benchmark per includerlo)")

    # 3. Benchmark VecEnv (opzionale, più lungo)
    vec_bench = None
    if run_vec_benchmark:
        if verbose:
            print("\n[3/4] Benchmark DummyVecEnv scaling...")
        try:
            vec_bench = benchmark_vec_env(verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"  Benchmark VecEnv fallito: {e}")
    else:
        if verbose:
            print("\n[3/4] Benchmark VecEnv saltato (usa --benchmark-vec per includerlo)")

    # 4. Calcola hyperparameters
    if verbose:
        print("\n[4/4] Calcolo hyperparameters...")
    config = calculate_hyperparameters(
        hw=hw,
        benchmark=bench,
        vec_benchmark=vec_bench,
        target_timesteps=target_timesteps,
    )
    if verbose:
        print(config.summary())

    # 5. Salva
    if output_path:
        save_path = Path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'hardware': {
                'cpu': hw.cpu_name,
                'physical_cores': hw.physical_cores,
                'logical_cores': hw.logical_cores,
                'ram_gb': round(hw.total_ram_gb, 1),
                'perf_tier': hw.perf_tier,
            },
            'hyperparameters': config.to_dict(),
            'cli_command': config.to_cli_args(),
            'estimated_time_minutes': round(config.estimated_time_minutes, 1),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        if bench:
            output['benchmark'] = {
                'steps_per_second': round(bench.steps_per_second, 1),
                'episodes_per_minute': round(bench.episodes_per_minute, 1),
            }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        if verbose:
            print(f"\nConfig salvata: {save_path}")

    # Mostra comando
    if verbose:
        print(f"\n{'='*55}")
        print(f"  COMANDO PER AVVIARE IL TRAINING")
        print(f"{'='*55}")
        print(f"\n  {config.to_cli_args()}\n")

        if output_path:
            print(f"  oppure con file config:")
            print(f"  python src/dqn_accel.py --config {output_path}\n")

    return config


# =============================================================================
#  CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter Tuner per DQN + ACCEL su Highway-Env",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
  # Analisi rapida
  python hyperparameter_tuner.py

  # Con benchmark environment
  python hyperparameter_tuner.py --benchmark

  # Con benchmark VecEnv scaling (più lungo, più preciso)
  python hyperparameter_tuner.py --benchmark --benchmark-vec

  # Salva config per dqn_accel.py
  python hyperparameter_tuner.py --benchmark --output hyperparams.json

  # Target specifico di timesteps
  python hyperparameter_tuner.py --timesteps 1000000 --output config.json

  # Usa la config generata:
  python dqn_accel.py --config hyperparams.json
        """
    )

    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Esegui benchmark singolo environment')
    parser.add_argument('--benchmark-vec', action='store_true', default=False,
                        help='Esegui benchmark DummyVecEnv con diversi num_envs')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Target timesteps (se non specificato, auto-calcolato)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path file JSON output (default: stampa a console)')
    parser.add_argument('--quiet', '-q', action='store_true', default=False,
                        help='Output minimo')

    args = parser.parse_args()

    config = run_tuner(
        run_benchmark=args.benchmark,
        run_vec_benchmark=args.benchmark_vec,
        target_timesteps=args.timesteps,
        output_path=args.output,
        verbose=not args.quiet,
    )
