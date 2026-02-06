"""
DQN + PLR (Prioritized Level Replay) — MPS-Optimised, Vectorised
=================================================================

Optimisations over the original script
---------------------------------------
1. **MPS backend**  – auto-detects Apple-Silicon GPU; all tensors stay on
   the same device to avoid host-to-device round-trips.
2. **SubprocVecEnv** – 4 parallel highway-env instances; one reset never
   blocks the others.
3. **Vectorised PLR scoring** – a single batched forward pass replaces the
   sequential per-episode Q-value loop.
4. **Staleness caching** – level scores are reused for `score_reuse_window`
   iterations instead of being recalculated every step.
5. **Aggressive NPC traffic** – AggressiveVehicle, POLITENESS=0,
   TIME_WANTED=1.0.
6. **float32 everywhere** – MPS has poor float64 support.
7. **FPS-tracking tqdm bar** – live verification of throughput.
"""

import gymnasium
import highway_env
import torch
import numpy as np
import json
import time
import pickle

from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from highway_env.vehicle.behavior import IDMVehicle

# ---------------------------------------------------------------------------
# 1. Global highway-env behaviour tweaks
# ---------------------------------------------------------------------------
IDMVehicle.POLITENESS = 0.0
IDMVehicle.TIME_WANTED = 1.0

# ---------------------------------------------------------------------------
# 2. Device detection  (mps → cuda → cpu)
# ---------------------------------------------------------------------------
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# ---------------------------------------------------------------------------
# 3. Environment helpers
# ---------------------------------------------------------------------------
NUM_ENVS = 4


def _make_env(config: dict, rank: int = 0):
    """Return a callable that creates one highway-v0 instance."""
    def _init():
        env = gymnasium.make("highway-v0", config=config)
        return env
    return _init


def make_vec_env(config: dict, n_envs: int = NUM_ENVS):
    """Create a SubprocVecEnv with `n_envs` copies of highway-v0."""
    return SubprocVecEnv([_make_env(config, i) for i in range(n_envs)])


def make_single_env(config: dict, render: bool = False):
    render_mode = "human" if render else None
    return gymnasium.make("highway-v0", config=config, render_mode=render_mode)


# ---------------------------------------------------------------------------
# 4. Configuration generation  (aggressive traffic)
# ---------------------------------------------------------------------------
def generate_env_configs() -> List[dict]:
    configs = []
    for lanes in [2, 3, 4]:
        for density in [0.8, 1.0, 1.5, 2.0]:
            for vehicles in [15, 20, 25, 30, 35]:
                configs.append({
                    "lanes_count": lanes,
                    "vehicles_density": density,
                    "vehicles_count": vehicles,
                    "collision_reward": -5,
                    "duration": 40,
                    "simulation_frequency": 15,
                    "policy_frequency": 5,
                    "other_vehicles_type":
                        "highway_env.vehicle.behavior.AggressiveVehicle",
                })
    return configs


# ---------------------------------------------------------------------------
# 5. Optimised PLR  (vectorised scoring, staleness caching)
# ---------------------------------------------------------------------------
class FastLevelSampler:
    """
    Streamlined PLR sampler.

    Key changes vs. the original LevelSampler
    ------------------------------------------
    * Scores stored as a contiguous float32 numpy array for fast indexing.
    * `score_reuse_window`: skip re-evaluation for N iterations after a
      score is computed.
    * `batch_score()` accepts pre-computed Q-values from a single batched
      forward pass.
    """

    def __init__(
        self,
        num_levels: int,
        rho: float = 0.8,
        staleness_coef: float = 0.1,
        score_transform: str = "rank",
        temperature: float = 0.1,
        eps: float = 0.05,
        alpha: float = 1.0,
        ema_alpha: float = 0.1,
        score_reuse_window: int = 3,
    ):
        self.num_levels = num_levels
        self.rho = rho
        self.staleness_coef = staleness_coef
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha
        self.ema_alpha = ema_alpha
        self.score_reuse_window = score_reuse_window

        # Storage  (float32 for MPS compat)
        self.scores = np.zeros(num_levels, dtype=np.float32)
        self.staleness = np.zeros(num_levels, dtype=np.float32)
        self.seen = np.zeros(num_levels, dtype=bool)
        self.last_scored_iter = np.full(num_levels, -score_reuse_window - 1,
                                        dtype=np.int64)
        self._iter = 0

    # ----- sampling --------------------------------------------------------
    def sample(self) -> int:
        n_seen = int(self.seen.sum())
        if n_seen == 0:
            return self._sample_unseen()

        replay = (n_seen == self.num_levels) or (np.random.rand() < self.rho)
        if replay and n_seen > 0:
            return self._sample_replay()
        return self._sample_unseen()

    def _sample_unseen(self) -> int:
        unseen = np.where(~self.seen)[0]
        if len(unseen) == 0:
            return np.random.randint(self.num_levels)
        idx = np.random.choice(unseen)
        self.seen[idx] = True
        return int(idx)

    def _sample_replay(self) -> int:
        seen_idx = np.where(self.seen)[0]
        scores = self.scores[seen_idx]

        # Score transform
        if self.score_transform == "rank":
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(order) + 1)
            weights = 1.0 / ranks.astype(np.float32)
        elif self.score_transform == "power":
            weights = np.power(np.maximum(scores, 1e-8), self.alpha)
        elif self.score_transform == "softmax":
            s = scores / max(self.temperature, 1e-8)
            s -= s.max()
            weights = np.exp(s)
        else:
            weights = np.maximum(scores, 1e-8)

        # Staleness bonus
        if self.staleness_coef > 0:
            st = self.staleness[seen_idx]
            st_w = np.exp(self.staleness_coef * st)
            weights = weights * st_w

        # Floor + normalise
        weights = weights / weights.sum()
        weights = (1 - self.eps) * weights + self.eps / len(weights)
        weights = weights / weights.sum()

        chosen = np.random.choice(seen_idx, p=weights)
        return int(chosen)

    # ----- scoring ---------------------------------------------------------
    def needs_scoring(self, level_idx: int) -> bool:
        return (self._iter - self.last_scored_iter[level_idx]
                >= self.score_reuse_window)

    def update_score(self, level_idx: int, new_score: float):
        self.seen[level_idx] = True
        if self.scores[level_idx] == 0.0:
            self.scores[level_idx] = new_score
        else:
            self.scores[level_idx] = (
                self.ema_alpha * new_score
                + (1 - self.ema_alpha) * self.scores[level_idx]
            )
        self.last_scored_iter[level_idx] = self._iter

    def step_staleness(self, active_idx: int):
        """Increment staleness for all, reset for the active level."""
        self.staleness += 1
        self.staleness[active_idx] = 0
        self._iter += 1

    # ----- stats / persistence ---------------------------------------------
    def get_stats(self) -> dict:
        seen_mask = self.seen
        n_seen = int(seen_mask.sum())
        if n_seen == 0:
            return {"plr/seen": 0, "plr/unseen": self.num_levels,
                    "plr/mean_score": 0.0, "plr/max_score": 0.0}
        s = self.scores[seen_mask]
        return {
            "plr/seen": n_seen,
            "plr/unseen": self.num_levels - n_seen,
            "plr/mean_score": float(s.mean()),
            "plr/max_score": float(s.max()),
            "plr/min_score": float(s.min()),
            "plr/std_score": float(s.std()),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "scores": self.scores,
                "staleness": self.staleness,
                "seen": self.seen,
                "last_scored_iter": self.last_scored_iter,
                "_iter": self._iter,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scores = d["scores"]
        self.staleness = d["staleness"]
        self.seen = d["seen"]
        self.last_scored_iter = d["last_scored_iter"]
        self._iter = d["_iter"]


# ---------------------------------------------------------------------------
# 6. Rolling statistics tracker
# ---------------------------------------------------------------------------
class RollingStats:
    """Track key agent performance metrics with rolling windows."""

    def __init__(self, window: int = 10):
        self.window = window
        self.rewards: List[float] = []
        self.crashes: List[bool] = []
        self.ep_lengths: List[int] = []
        self.best_reward = -float("inf")
        self.worst_reward = float("inf")
        self.total_evals = 0

    def update(self, mean_reward: float, crash_count: int,
               n_episodes: int, mean_ep_length: float):
        self.rewards.append(mean_reward)
        for _ in range(n_episodes):
            self.crashes.append(crash_count > 0)
        self.ep_lengths.append(mean_ep_length)
        self.total_evals += 1
        self.best_reward = max(self.best_reward, mean_reward)
        self.worst_reward = min(self.worst_reward, mean_reward)

    # -- rolling helpers --
    @property
    def avg_reward(self) -> float:
        w = self.rewards[-self.window:]
        return float(np.mean(w)) if w else 0.0

    @property
    def avg_reward_all(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def reward_trend(self) -> str:
        """Compare last-window avg to previous-window avg."""
        if len(self.rewards) < 2 * self.window:
            return "…"  # not enough data
        recent = np.mean(self.rewards[-self.window:])
        prev = np.mean(self.rewards[-2 * self.window:-self.window])
        diff = recent - prev
        if diff > 0.5:
            return f"▲ +{diff:.1f}"
        elif diff < -0.5:
            return f"▼ {diff:.1f}"
        return "━ stable"

    @property
    def crash_rate(self) -> float:
        w = self.crashes[-self.window * 3:]  # last N evals × episodes
        return 100.0 * sum(w) / max(len(w), 1)

    @property
    def survival_rate(self) -> float:
        return 100.0 - self.crash_rate

    @property
    def avg_ep_length(self) -> float:
        w = self.ep_lengths[-self.window:]
        return float(np.mean(w)) if w else 0.0


# ---------------------------------------------------------------------------
# 7. Vectorised PLR evaluation  (batched Q-value forward pass)
# ---------------------------------------------------------------------------
@torch.no_grad()
def vectorised_evaluate(
    model: DQN,
    config: dict,
    n_episodes: int = 3,
    gamma: float = 0.8,
) -> dict:
    """
    Evaluate a config over `n_episodes` sequentially but score with a
    single batched forward pass at the end.

    Returns dict with: returns, values, mean_reward, crash_count,
    mean_ep_length, episode_rewards.
    """
    device = model.device
    all_obs: List[np.ndarray] = []
    all_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_rewards: List[float] = []
    crash_count = 0

    env = make_single_env(config)
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = truncated = False
        ep_obs, ep_rew = [], []

        while not (done or truncated):
            ep_obs.append(obs.copy())
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_rew.append(float(reward))

        if done and not truncated:
            crash_count += 1

        all_obs.extend(ep_obs)
        all_rewards.extend(ep_rew)
        episode_lengths.append(len(ep_rew))
        episode_rewards.append(sum(ep_rew))
    env.close()

    if len(all_obs) == 0:
        return {"returns": np.zeros(1, dtype=np.float32),
                "values": np.zeros(1, dtype=np.float32),
                "mean_reward": 0.0, "crash_count": 0,
                "mean_ep_length": 0.0, "episode_rewards": []}

    # ---- batched Q-value pass (single .to(device) call) ----
    obs_t = torch.as_tensor(np.array(all_obs, dtype=np.float32),
                            device=device)
    q_values = model.policy.q_net(obs_t)
    values = q_values.max(dim=1)[0].cpu().numpy().astype(np.float32)

    # ---- compute returns ----
    returns = np.empty_like(values)
    idx = 0
    for length in episode_lengths:
        R = 0.0
        for t in reversed(range(length)):
            R = all_rewards[idx + t] + gamma * R
            returns[idx + t] = R
        idx += length

    return {
        "returns": returns,
        "values": values,
        "mean_reward": float(np.mean(episode_rewards)),
        "crash_count": crash_count,
        "mean_ep_length": float(np.mean(episode_lengths)),
        "episode_rewards": episode_rewards,
    }


# ---------------------------------------------------------------------------
# 8. Main training loop
# ---------------------------------------------------------------------------
def train_dqn_plr(
    total_timesteps: int = 100_000,
    steps_per_config: int = 5_000,
    eval_episodes: int = 3,
    plr_rho: float = 0.8,
    score_reuse_window: int = 3,
    save_dir: str = "highway_dqn_plr",
    verbose: int = 1,
):
    device = get_device()

    print(f"\n{'=' * 72}")
    print("  DQN + PLR  —  MPS-Optimised, Vectorised Training")
    print(f"{'=' * 72}")
    print(f"  Device .............. {device}")
    print(f"  Parallel envs ....... {NUM_ENVS}")
    print(f"  Total timesteps ..... {total_timesteps:,}")
    print(f"  Steps per config .... {steps_per_config:,}")
    print(f"  Eval episodes ....... {eval_episodes}")
    print(f"  PLR rho ............. {plr_rho}")
    print(f"  Score reuse window .. {score_reuse_window}")
    print(f"  IDM POLITENESS ...... {IDMVehicle.POLITENESS}")
    print(f"  IDM TIME_WANTED ..... {IDMVehicle.TIME_WANTED}")
    print(f"{'=' * 72}\n")

    # ---- configs & PLR ----------------------------------------------------
    env_configs = generate_env_configs()
    n_configs = len(env_configs)
    print(f"Generated {n_configs} environment configurations\n")

    sampler = FastLevelSampler(
        num_levels=n_configs,
        rho=plr_rho,
        score_reuse_window=score_reuse_window,
    )

    # ---- initial env & model ----------------------------------------------
    level_idx = sampler.sample()
    config = env_configs[level_idx]
    vec_env = make_vec_env(config, NUM_ENVS)

    model = DQN(
        "MlpPolicy",
        vec_env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15_000,
        learning_starts=200,
        batch_size=64,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=0,
        tensorboard_log=f"{save_dir}/tensorboard/",
        device=device,
    )

    # ---- bookkeeping ------------------------------------------------------
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history: Dict[str, list] = {
        "configs": [], "rewards": [], "crash_rates": [],
        "ep_lengths": [], "timesteps": [], "fps": [],
    }
    stats = RollingStats(window=10)
    timesteps_done = 0
    iteration = 0
    evals_done = 0
    chunk_size = 256  # small chunks for responsive FPS counter

    pbar = tqdm(total=total_timesteps, desc="Training", unit="step",
                dynamic_ncols=True)
    wall_start = time.perf_counter()

    # ---- training loop ----------------------------------------------------
    while timesteps_done < total_timesteps:
        iteration += 1
        steps_this_config = min(steps_per_config,
                                total_timesteps - timesteps_done)

        # ---- train in small chunks for live FPS ----------------------------
        trained = 0
        while trained < steps_this_config:
            chunk = min(chunk_size, steps_this_config - trained)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                        progress_bar=False)
            trained += chunk
            timesteps_done += chunk
            pbar.update(chunk)

        # ---- PLR scoring (skip if recently scored) -------------------------
        evaluated = sampler.needs_scoring(level_idx)
        if evaluated:
            eval_data = vectorised_evaluate(
                model, config, n_episodes=eval_episodes, gamma=model.gamma,
            )
            score = float(
                np.abs(eval_data["returns"] - eval_data["values"]).mean()
            )
            sampler.update_score(level_idx, score)
            mean_reward = eval_data["mean_reward"]
            evals_done += 1

            # Update rolling stats
            stats.update(
                mean_reward=mean_reward,
                crash_count=eval_data["crash_count"],
                n_episodes=eval_episodes,
                mean_ep_length=eval_data["mean_ep_length"],
            )
        else:
            mean_reward = stats.rewards[-1] if stats.rewards else 0.0

        sampler.step_staleness(level_idx)

        # ---- per-iteration log (compact, informative) ----------------------
        elapsed = time.perf_counter() - wall_start
        fps = timesteps_done / max(elapsed, 1e-6)

        if verbose and evaluated:
            cfg = env_configs[level_idx]
            plr = sampler.get_stats()
            crash_str = (
                f"{eval_data['crash_count']}/{eval_episodes} crashed"
                if eval_data["crash_count"] > 0
                else f"no crashes"
            )
            tqdm.write(
                f"\n  ┌─ Iter {iteration:>3d}  │  "
                f"{timesteps_done:,}/{total_timesteps:,} steps  │  "
                f"{elapsed:.0f}s elapsed"
            )
            tqdm.write(
                f"  │  Env: L={cfg['lanes_count']}  "
                f"V={cfg['vehicles_count']}  "
                f"D={cfg['vehicles_density']:.1f}  "
                f"(level {level_idx}, "
                f"{plr['plr/seen']}/{sampler.num_levels} seen)"
            )
            tqdm.write(
                f"  │  Reward: {mean_reward:>7.2f}  │  "
                f"Avg(last 10): {stats.avg_reward:>7.2f}  │  "
                f"Best: {stats.best_reward:.2f}  │  "
                f"Trend: {stats.reward_trend}"
            )
            tqdm.write(
                f"  │  Survival: {stats.survival_rate:>5.1f}%  │  "
                f"{crash_str}  │  "
                f"Avg length: {stats.avg_ep_length:.0f} steps"
            )
            tqdm.write(
                f"  └─ FPS: {fps:.0f}"
            )

        # Update tqdm postfix with rolling metrics
        pbar.set_postfix_str(
            f"R={stats.avg_reward:.1f}  "
            f"Surv={stats.survival_rate:.0f}%  "
            f"{fps:.0f} FPS"
        )

        # ---- history -------------------------------------------------------
        history["configs"].append(int(level_idx))
        history["rewards"].append(float(mean_reward))
        history["crash_rates"].append(float(stats.crash_rate))
        history["ep_lengths"].append(float(stats.avg_ep_length))
        history["timesteps"].append(int(timesteps_done))
        history["fps"].append(float(fps))

        # ---- sample next level, swap env ----------------------------------
        level_idx = sampler.sample()
        config = env_configs[level_idx]
        vec_env.close()
        vec_env = make_vec_env(config, NUM_ENVS)
        model.set_env(vec_env)

    pbar.close()
    vec_env.close()

    wall_total = time.perf_counter() - wall_start
    avg_fps = total_timesteps / max(wall_total, 1e-6)

    # ---- persist -----------------------------------------------------------
    model.save(save_path / "model")
    sampler.save(str(save_path / "plr_state.pkl"))

    with open(save_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "total_timesteps": total_timesteps,
        "steps_per_config": steps_per_config,
        "num_envs": NUM_ENVS,
        "device": device,
        "wall_time_s": round(wall_total, 1),
        "avg_fps": round(avg_fps, 1),
        "plr": {"rho": plr_rho, "score_reuse_window": score_reuse_window},
        "num_configs": n_configs,
    }
    with open(save_path / "training_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    plr_final = sampler.get_stats()
    print(f"\n{'=' * 72}")
    print("  TRAINING COMPLETE")
    print(f"{'=' * 72}")
    print(f"  Duration ............ {wall_total/60:.1f} min ({wall_total:.0f}s)")
    print(f"  Average FPS ......... {avg_fps:.0f}")
    print(f"  Total evaluations ... {evals_done}")
    print(f"{'=' * 72}")
    print(f"  AGENT PERFORMANCE")
    print(f"  {'─' * 40}")
    print(f"  Avg reward (all) .... {stats.avg_reward_all:.2f}")
    print(f"  Avg reward (last 10)  {stats.avg_reward:.2f}")
    print(f"  Best reward ......... {stats.best_reward:.2f}")
    print(f"  Worst reward ........ {stats.worst_reward:.2f}")
    print(f"  Survival rate ....... {stats.survival_rate:.1f}%")
    print(f"  Avg episode length .. {stats.avg_ep_length:.0f} steps")
    print(f"  Reward trend ........ {stats.reward_trend}")
    print(f"{'=' * 72}")
    print(f"  CURRICULUM (PLR)")
    print(f"  {'─' * 40}")
    print(f"  Levels seen ......... {plr_final['plr/seen']}/{sampler.num_levels}")
    print(f"  Levels unseen ....... {plr_final['plr/unseen']}")
    print(f"  Mean PLR score ...... {plr_final['plr/mean_score']:.4f}")
    print(f"  Max PLR score ....... {plr_final['plr/max_score']:.4f}")
    print(f"{'=' * 72}")
    print(f"  Saved to ............ {save_path}")
    print(f"{'=' * 72}\n")

    return model, sampler, history


# ---------------------------------------------------------------------------
# 9. Test evaluation
# ---------------------------------------------------------------------------
def evaluate_on_test_configs(model, test_configs, n_episodes=10):
    print(f"\n{'=' * 72}")
    print("  EVALUATING ON TEST CONFIGURATIONS")
    print(f"{'=' * 72}")
    print(f"  Episodes per config . {n_episodes}")
    print(f"  Test configs ........ {len(test_configs)}")
    print(f"{'=' * 72}\n")

    results = []
    all_rewards_global: List[float] = []
    all_crashes_global = 0
    all_lengths_global: List[int] = []
    total_episodes = 0

    for i, config in enumerate(test_configs):
        env = make_single_env(config)
        rewards: List[float] = []
        ep_lengths: List[int] = []
        crashes = 0

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False
            ep_r = 0.0
            steps = 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                ep_r += r
                steps += 1
            rewards.append(ep_r)
            ep_lengths.append(steps)
            if done and not truncated:
                crashes += 1
        env.close()

        avg = float(np.mean(rewards))
        std = float(np.std(rewards))
        med = float(np.median(rewards))
        best = float(np.max(rewards))
        worst = float(np.min(rewards))
        cr = 100.0 * crashes / n_episodes
        surv = 100.0 - cr
        avg_len = float(np.mean(ep_lengths))

        results.append({
            "config": config, "avg_reward": avg, "std_reward": std,
            "median_reward": med, "best_reward": best, "worst_reward": worst,
            "crash_rate": cr, "survival_rate": surv,
            "avg_ep_length": avg_len, "crashes": crashes,
        })

        all_rewards_global.extend(rewards)
        all_crashes_global += crashes
        all_lengths_global.extend(ep_lengths)
        total_episodes += n_episodes

        crash_str = (
            f"{crashes}/{n_episodes} crashed"
            if crashes > 0 else "no crashes"
        )

        print(
            f"  ┌─ Config {i+1}/{len(test_configs)}  │  "
            f"L={config['lanes_count']}  "
            f"V={config['vehicles_count']}  "
            f"D={config['vehicles_density']:.1f}"
        )
        print(
            f"  │  Reward: {avg:>7.2f} ± {std:.2f}  │  "
            f"Median: {med:.2f}  │  "
            f"Best: {best:.2f}  │  Worst: {worst:.2f}"
        )
        print(
            f"  └─ Survival: {surv:>5.1f}%  │  "
            f"{crash_str}  │  "
            f"Avg length: {avg_len:.0f} steps\n"
        )

    # ---- overall summary ---------------------------------------------------
    overall_avg = float(np.mean(all_rewards_global))
    overall_std = float(np.std(all_rewards_global))
    overall_med = float(np.median(all_rewards_global))
    overall_best = float(np.max(all_rewards_global))
    overall_worst = float(np.min(all_rewards_global))
    overall_cr = 100.0 * all_crashes_global / max(total_episodes, 1)
    overall_surv = 100.0 - overall_cr
    overall_len = float(np.mean(all_lengths_global))

    print(f"{'=' * 72}")
    print("  EVALUATION SUMMARY")
    print(f"  {'─' * 40}")
    print(f"  Total episodes ...... {total_episodes}")
    print(f"  Avg reward .......... {overall_avg:.2f} ± {overall_std:.2f}")
    print(f"  Median reward ....... {overall_med:.2f}")
    print(f"  Best reward ......... {overall_best:.2f}")
    print(f"  Worst reward ........ {overall_worst:.2f}")
    print(f"  Survival rate ....... {overall_surv:.1f}%  "
          f"({all_crashes_global}/{total_episodes} crashed)")
    print(f"  Avg episode length .. {overall_len:.0f} steps")
    print(f"{'=' * 72}\n")

    return results


# ---------------------------------------------------------------------------
# 10. Watch agent
# ---------------------------------------------------------------------------
def watch_agent(model, config, n_episodes=5):
    print(f"\n{'=' * 60}")
    print("  WATCHING AGENT  (Ctrl-C to stop)")
    print(f"{'=' * 60}\n")

    env = make_single_env(config, render=True)
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            env.render()
            done = truncated = False
            ep_r, steps = 0.0, 0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, _ = env.step(action)
                env.render()
                ep_r += r
                steps += 1
            tag = "CRASH" if (done and not truncated) else "SURVIVED"
            print(f"  Ep {ep+1}: {tag}  R={ep_r:.2f}  Steps={steps}")
    except KeyboardInterrupt:
        print("\n  Stopped by user")
    finally:
        env.close()
    print()


# ---------------------------------------------------------------------------
# 11. Entry point
# ---------------------------------------------------------------------------
def main():
    # ====== FLAGS ===========================================================
    TRAIN           = False
    SKIP_EVALUATION = True

    TOTAL_TIMESTEPS = 1_000_000
    STEPS_PER_CONFIG = 10_000
    EVAL_EPISODES   = 3
    PLR_RHO         = 0.8
    SCORE_REUSE     = 3        # reuse scores for N iterations
    SAVE_DIR        = "highway_dqn_plr"
    # ========================================================================

    if TRAIN:
        model, sampler, history = train_dqn_plr(
            total_timesteps=TOTAL_TIMESTEPS,
            steps_per_config=STEPS_PER_CONFIG,
            eval_episodes=EVAL_EPISODES,
            plr_rho=PLR_RHO,
            score_reuse_window=SCORE_REUSE,
            save_dir=SAVE_DIR,
            verbose=1,
        )
    else:
        device = get_device()
        model = DQN.load(f"{SAVE_DIR}/model", device=device)

    # ---- evaluation -------------------------------------------------------
    if not SKIP_EVALUATION:
        test_cfgs = [
            {"lanes_count": 2, "vehicles_count": 40, "vehicles_density": 2.5,
             "duration": 40, "simulation_frequency": 30, "policy_frequency": 5,
             "other_vehicles_type":
                 "highway_env.vehicle.behavior.AggressiveVehicle"},
            {"lanes_count": 3, "vehicles_count": 18, "vehicles_density": 0.9,
             "duration": 40, "simulation_frequency": 30, "policy_frequency": 5,
             "other_vehicles_type":
                 "highway_env.vehicle.behavior.AggressiveVehicle"},
            {"lanes_count": 4, "vehicles_count": 28, "vehicles_density": 1.3,
             "duration": 40, "simulation_frequency": 30, "policy_frequency": 5,
             "other_vehicles_type":
                 "highway_env.vehicle.behavior.AggressiveVehicle"},
            {"lanes_count": 5, "vehicles_count": 22, "vehicles_density": 1.1,
             "duration": 40, "simulation_frequency": 30, "policy_frequency": 5,
             "other_vehicles_type":
                 "highway_env.vehicle.behavior.AggressiveVehicle"},
        ]
        results = evaluate_on_test_configs(model, test_cfgs, n_episodes=10)

        save_path = Path(SAVE_DIR)
        with open(save_path / "test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        print("Skipping evaluation, going directly to visualization...")

    # ---- visualisation ----------------------------------------------------
    watch_cfg = {
        "lanes_count": 4,
        "vehicles_count": 25,
        "vehicles_density": 1,
        "duration": 60,
        "simulation_frequency": 30,
        "policy_frequency": 5,
        "other_vehicles_type":
            "highway_env.vehicle.behavior.AggressiveVehicle",
    }
    watch_agent(model, watch_cfg, n_episodes=5)


if __name__ == "__main__":
    main()