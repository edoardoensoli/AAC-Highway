"""
Highway Level Management — seed-based level parameterisation for PLR.

Self-contained module: imports from ``plr.plr_configs`` (same folder).
Adds ``HighwayVecEnv`` and ``SubprocVecEnv`` adapters that expose the
``set_level() / reset()`` interface consumed by the training loop and PLR
sampler.

Design goal:  **one persistent env** that changes configuration via
``set_level(seed)`` rather than creating/destroying envs per level.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
import torch

# Register highway-env environments
import highway_env  # noqa: F401

# Import from local plr_configs (same package) 
from plr_dqn.plr_configs import (
    seed_to_factors,
    seed_to_config,
    describe_level,
    config_for_gym,
    HighwayLevelWrapper,
    OBSERVATION_CFG,
    ACTION_CFG,
    REWARD_CFG,
    TEST_SEEDS,
)


# Observation shape (derived from Kinematics config)
_num_vehicles = OBSERVATION_CFG["vehicles_count"]   # 10
_num_features = len(OBSERVATION_CFG["features"])
OBS_SHAPE = (_num_vehicles * _num_features,)
NUM_ACTIONS = 5


def _gym_config(seed: int) -> dict:
    """Build a config dict suitable for gym.make (no meta keys)."""
    return config_for_gym(seed_to_config(seed))


def make_flat_env(seed: int = 0, render: bool = False) -> gym.Env:
    """
    Create a highway-v0 env with FlattenObservation for SB3 compatibility.

    Returns a standard Gymnasium env with flat observation space
    Box(shape=(OBS_SHAPE,)) suitable for ``stable_baselines3.DQN``.
    """
    cfg = _gym_config(seed)
    render_mode = "human" if render else None
    env = gym.make("highway-fast-v0", config=cfg, render_mode=render_mode)
    env = HighwayLevelWrapper(env, seed=seed)
    env = gym.wrappers.FlattenObservation(env)
    return env


class HighwayVecEnv:
    """
    Minimal single-process "vector" env that wraps one HighwayLevelWrapper.

    Compatible with the DCD-style training loop:

    - ``set_level(seed)``: change config for the next reset
    - ``reset()``: reset env with current config, return obs tensor
    - ``step(action)``: take a step, return (obs, reward, done, info) tensors
    - ``obs_to_tensor(obs)``: numpy → float torch tensor

    All returned observations have shape ``(1, C, H, W)`` (batch dim = 1).
    """

    def __init__(self, seed: int = 0, device: str = "cpu", render: bool = False):
        self.device = device
        self._render_mode = "human" if render else None
        self._current_seed = seed

        # Build the underlying env
        self._env = self._make_env(seed)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _make_env(self, seed: int) -> gym.Env:
        cfg = _gym_config(seed)
        env = gym.make("highway-fast-v0", config=cfg, render_mode=self._render_mode)
        return HighwayLevelWrapper(env, seed=seed)

    def set_level(self, seed: int):
        """Configure the next episode to use ``seed``."""
        self._current_seed = seed
        self._env.set_level(seed)

    def get_level_seed(self) -> int:
        return self._current_seed

    def reset(self) -> torch.Tensor:
        """Reset and return obs tensor (1, C, H, W)."""
        obs, _info = self._env.reset()
        return self._obs_to_tensor(obs)

    def step(self, action: int | torch.Tensor):
        """
        Step the environment.

        Returns
        -------
        obs : Tensor (1, C, H, W)
        reward : Tensor (1, 1)
        done : Tensor (1, 1)  — float mask (0 = done, 1 = alive)
        info : dict
        """
        if isinstance(action, torch.Tensor):
            action = action.item()

        obs, reward, terminated, truncated, info = self._env.step(int(action))
        done = terminated or truncated

        obs_t = self._obs_to_tensor(obs)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        # DCD convention: mask = 0 at episode end, 1 otherwise
        mask = torch.tensor(
            [[0.0 if done else 1.0]], dtype=torch.float32, device=self.device
        )
        # bad_mask: 0 if truncated (time limit) but not truly terminal
        bad_mask = torch.tensor(
            [[0.0 if truncated and not terminated else 1.0]],
            dtype=torch.float32,
            device=self.device,
        )

        info["done"] = done
        info["terminated"] = terminated
        info["truncated"] = truncated
        info["level_seed"] = self._current_seed

        return obs_t, reward_t, mask, bad_mask, info

    def step_and_maybe_reset(self, action: int | torch.Tensor, next_seed: int | None = None):
        """
        Step; if episode ends, automatically reset with ``next_seed``.

        Returns
        -------
        obs : Tensor  — next obs (possibly from new episode)
        reward, mask, bad_mask : Tensor
        info : dict  (includes ``episode_return`` if ended)
        """
        obs, reward, mask, bad_mask, info = self.step(action)
        if info["done"]:
            # Store episode stats before reset
            if next_seed is not None:
                self.set_level(next_seed)
            obs = self.reset()
        return obs, reward, mask, bad_mask, info

    def _obs_to_tensor(self, obs: np.ndarray) -> torch.Tensor:
        """Convert observation to float tensor with batch dim, flattened."""
        t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).flatten()
        t = t.unsqueeze(0)  # add batch dim → (1, flat)
        return t.to(self.device)

    def close(self):
        self._env.close()

    def describe_level(self, seed: int | None = None) -> str:
        seed = seed or self._current_seed
        f = seed_to_factors(seed)
        return (
            f"seed={seed}  L={f['lanes_count']}  V={f['vehicles_count']}  "
            f"D={f['vehicles_density']:.2f}  P={f['politeness']:.3f}"
        )


import multiprocessing as mp


def _worker(remote, parent_remote, seed):
    """Worker process: runs one HighwayLevelWrapper in a loop."""
    parent_remote.close()
    cfg = _gym_config(seed)
    env = gym.make("highway-fast-v0", config=cfg, render_mode=None)
    env = HighwayLevelWrapper(env, seed=seed)
    current_seed = seed

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            done = terminated or truncated
            info["done"] = done
            info["terminated"] = terminated
            info["truncated"] = truncated
            info["level_seed"] = current_seed
            if done:
                # Auto-reset
                obs, _info = env.reset()
            remote.send((obs, reward, done, truncated, terminated, info))

        elif cmd == "reset":
            obs, info = env.reset()
            remote.send(obs)

        elif cmd == "set_level":
            current_seed = data
            env.set_level(data)
            remote.send(None)

        elif cmd == "get_seed":
            remote.send(current_seed)

        elif cmd == "close":
            env.close()
            remote.close()
            break


class SubprocVecEnv:
    """
    Multiprocessing vectorised env for N parallel highway-env workers.

    Each worker runs in its own process. Returns batched tensors with
    shape ``(N, ...)`` compatible with the RolloutStorage.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments.
    initial_seeds : list[int] | None
        Initial level seeds (one per env). Defaults to [0, 1, ..., N-1].
    device : str
        Device for output tensors.
    """

    def __init__(
        self,
        num_envs: int = 4,
        initial_seeds: list[int] | None = None,
        device: str = "cpu",
    ):
        self.num_envs = num_envs
        self.device = device
        self._current_seeds = list(initial_seeds or range(num_envs))

        # Fork-safe on macOS
        ctx = mp.get_context("spawn")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)]
        )
        self.processes = []
        for i, (work_remote, remote, seed) in enumerate(
            zip(self.work_remotes, self.remotes, self._current_seeds)
        ):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, seed),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            work_remote.close()

    def set_levels(self, seeds: list[int]):
        """Set level seed for each env."""
        assert len(seeds) == self.num_envs
        self._current_seeds = list(seeds)
        for remote, seed in zip(self.remotes, seeds):
            remote.send(("set_level", seed))
        for remote in self.remotes:
            remote.recv()

    def set_level_all(self, seed: int):
        """Set the same seed for all envs."""
        self.set_levels([seed] * self.num_envs)

    def get_seeds(self) -> list[int]:
        return list(self._current_seeds)

    def reset(self) -> torch.Tensor:
        """Reset all envs. Returns (N, C, H, W) tensor."""
        for remote in self.remotes:
            remote.send(("reset", None))
        obs_list = [remote.recv() for remote in self.remotes]
        return self._stack_obs(obs_list)

    def step(self, actions: torch.Tensor):
        """
        Step all envs with a (N, 1) action tensor.

        Returns
        -------
        obs : (N, C, H, W)
        rewards : (N, 1)
        masks : (N, 1)     — 0 at episode end
        bad_masks : (N, 1)  — 0 if truncated
        infos : list[dict]
        """
        for i, remote in enumerate(self.remotes):
            a = actions[i].item() if isinstance(actions, torch.Tensor) else int(actions[i])
            remote.send(("step", int(a)))

        results = [remote.recv() for remote in self.remotes]

        obs_list, rewards, masks, bad_masks, infos = [], [], [], [], []
        for i, (obs, reward, done, truncated, terminated, info) in enumerate(results):
            obs_list.append(obs)
            rewards.append(reward)
            masks.append(0.0 if done else 1.0)
            bad_masks.append(0.0 if truncated and not terminated else 1.0)
            infos.append(info)

        obs_t = self._stack_obs(obs_list)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        masks_t = torch.tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)
        bad_masks_t = torch.tensor(bad_masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        return obs_t, rewards_t, masks_t, bad_masks_t, infos

    def _stack_obs(self, obs_list) -> torch.Tensor:
        # Flatten each obs and stack → (N, flat)
        stacked = np.stack([np.asarray(o, dtype=np.float32).flatten() for o in obs_list])
        return torch.from_numpy(stacked).to(self.device)

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)


EVAL_SEEDS = TEST_SEEDS


if __name__ == "__main__":
    print(f"OBS_SHAPE  = {OBS_SHAPE}")
    print(f"NUM_ACTIONS = {NUM_ACTIONS}")

    env = HighwayVecEnv(seed=42)
    obs = env.reset()
    print(f"obs shape  = {obs.shape}")

    for t in range(5):
        a = env.action_space.sample()
        obs, rew, mask, bad_mask, info = env.step(a)
        print(f"  t={t}  a={a}  rew={rew.item():.3f}  mask={mask.item()}")

    env.close()
    print("Smoke test passed.")
