"""
Level Sampler — Prioritized Level Replay  (PLR / PLR⊥).

Faithful adaptation of ``facebookresearch/dcd/level_replay/level_sampler.py``
for the *infinite seed* setting (``sample_full_distribution=True``).

Key concepts
~~~~~~~~~~~~
* **Staging set** — newly generated seeds awaiting their first full episode.
* **Working buffer** — fixed-size buffer of the best-scored seeds.
* **Replay decision** — flip a coin: replay from buffer or explore new seed.
* **Score function** — ``value_l1`` (|return − value_pred|) is the default.
* **Score transform** — ``rank`` converts scores to sampling weights.
* **Staleness** — encourages revisiting stale levels.
* **Robust PLR⊥** — ``no_exploratory_grad_updates=True`` discards gradients
  on newly explored (non-replay) levels.

Reference
~~~~~~~~~
Jiang, Grefenstette & Rocktäschel (2021)
*Prioritized Level Replay*  —  arXiv:2010.03934
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


INT32_MAX = 2_147_483_647


class LevelSampler:
    """
    PLR level sampler with infinite seed support.

    Parameters match the DCD default configuration for Robust PLR⊥::

        strategy            = "value_l1"
        score_transform     = "rank"
        temperature         = 0.1
        replay_schedule     = "fixed"
        replay_prob         = 0.95
        rho                 = 1.0
        staleness_coef      = 0.3
        staleness_transform = "power"
        seed_buffer_size    = 4000
        no_exploratory_grad_updates = True   (handled in train loop)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_actors: int = 1,
        strategy: str = "value_l1",
        replay_schedule: str = "fixed",
        score_transform: str = "rank",
        temperature: float = 0.1,
        eps: float = 0.05,
        rho: float = 1.0,
        replay_prob: float = 0.95,
        alpha: float = 1.0,
        staleness_coef: float = 0.3,
        staleness_transform: str = "power",
        staleness_temperature: float = 1.0,
        seed_buffer_size: int = 4000,
        seed_buffer_priority: str = "replay_support",
        gamma: float = 0.999,
    ):
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.replay_prob = replay_prob
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        self.gamma = gamma

        self.seed_buffer_size = seed_buffer_size
        self.seed_buffer_priority = seed_buffer_priority

        # ── Working buffer arrays ────────────────────────────────────────
        N = seed_buffer_size
        self.seeds = np.zeros(N, dtype=np.int64) - 1  # -1 = empty slot
        self.seed2index: Dict[int, int] = {}
        self.unseen_seed_weights = np.ones(N, dtype=np.float64)
        self.seed_scores = np.zeros(N, dtype=np.float64)
        self.partial_seed_scores = np.zeros((num_actors, N), dtype=np.float64)
        self.partial_seed_steps = np.zeros((num_actors, N), dtype=np.int32)
        self.seed_staleness = np.zeros(N, dtype=np.float64)

        self.running_sample_count = 0
        self.working_seed_buffer_size = 0

        # ── Staging data (infinite seed support) ─────────────────────────
        self.staging_seed_set: Set[int] = set()
        self.working_seed_set: Set[int] = set()
        self.seed2actor: Dict[int, set] = defaultdict(set)
        self.seed2timestamp_buffer: Dict[int, int] = {}
        self.partial_seed_scores_buffer: List[Dict[int, float]] = [
            {} for _ in range(num_actors)
        ]
        self.partial_seed_steps_buffer: List[Dict[int, int]] = [
            {} for _ in range(num_actors)
        ]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def _proportion_filled(self) -> float:
        return self.working_seed_buffer_size / self.seed_buffer_size

    @property
    def _has_working_seed_buffer(self) -> bool:
        return self.seed_buffer_size > 0

    @property
    def is_warm(self) -> bool:
        return self._proportion_filled >= self.rho

    @property
    def _next_buffer_index(self) -> int:
        if self._proportion_filled < 1.0:
            return self.working_seed_buffer_size
        else:
            if self.seed_buffer_priority == "replay_support":
                return self.sample_weights().argmin()
            else:
                return self.seed_scores.argmin()

    # ── External unseen sample (called when new levels are generated) ────

    def observe_external_unseen_sample(self, seeds, solvable=None):
        """Register newly generated seeds in the staging set."""
        for i, seed in enumerate(seeds):
            self.running_sample_count += 1
            if not (seed in self.staging_seed_set or seed in self.working_seed_set):
                self.seed2timestamp_buffer[seed] = self.running_sample_count
                self.staging_seed_set.add(seed)
            else:
                seed_idx = self.seed2index.get(seed, None)
                if seed_idx is not None:
                    self._update_staleness(seed_idx)

    # ── Replay decision ──────────────────────────────────────────────────

    def sample_replay_decision(self) -> bool:
        """Decide whether to replay (True) or explore (False)."""
        proportion_filled = self._proportion_filled
        if self.seed_buffer_size > 0:
            if self.replay_schedule == "fixed":
                if proportion_filled >= self.rho and np.random.rand() < self.replay_prob:
                    return True
                else:
                    return False
            else:
                if proportion_filled >= self.rho and np.random.rand() < min(
                    proportion_filled, self.replay_prob
                ):
                    return True
                else:
                    return False
        else:
            return False

    # ── Level sampling ───────────────────────────────────────────────────

    def sample_replay_level(self) -> int:
        """Sample a seed from the working buffer."""
        return self._sample_replay_level()

    def _sample_replay_level(self, update_staleness: bool = True) -> int:
        weights = self.sample_weights()
        if np.isclose(np.sum(weights), 0):
            weights = np.ones_like(self.seeds, dtype=np.float64) / len(self.seeds)
            weights = weights * (1 - self.unseen_seed_weights)
            s = np.sum(weights)
            if s > 0:
                weights /= s
            else:
                weights = np.ones(len(self.seeds)) / len(self.seeds)
        elif np.sum(weights) != 1.0:
            weights = weights / np.sum(weights)

        seed_idx = np.random.choice(len(self.seeds), p=weights)
        seed = self.seeds[seed_idx]
        if update_staleness:
            self._update_staleness(seed_idx)
        return int(seed)

    def sample_unseen_level(self) -> int:
        """Generate a new random seed (infinite distribution)."""
        seed = int(np.random.randint(1, INT32_MAX))
        while seed in self.staging_seed_set or seed in self.working_seed_set:
            seed = int(np.random.randint(1, INT32_MAX))
        self.seed2timestamp_buffer[seed] = self.running_sample_count
        self.staging_seed_set.add(seed)
        return seed

    # ── Score updates ────────────────────────────────────────────────────

    def update_seed_score(
        self, actor_index: int, seed: int, score: float, num_steps: int
    ):
        """Full update for a completed episode on ``seed``."""
        if seed in self.staging_seed_set:
            self._partial_update_seed_score_buffer(
                actor_index, seed, score, num_steps, done=True
            )
        else:
            self._partial_update_seed_score(
                actor_index, seed, score, num_steps, done=True
            )

    def _partial_update_seed_score(
        self,
        actor_index: int,
        seed: int,
        score: float,
        num_steps: int,
        done: bool = False,
    ):
        seed_idx = self.seed2index.get(seed, -1)
        if seed_idx < 0:
            return

        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps
        if running_num_steps > 0:
            merged_score = merged_score / float(running_num_steps)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0.0
            self.partial_seed_steps[actor_index][seed_idx] = 0
            self.unseen_seed_weights[seed_idx] = 0.0
            old_score = self.seed_scores[seed_idx]
            self.seed_scores[seed_idx] = (
                (1 - self.alpha) * old_score + self.alpha * merged_score
            )
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

    def _partial_update_seed_score_buffer(
        self,
        actor_index: int,
        seed: int,
        score: float,
        num_steps: int,
        done: bool = False,
    ):
        """Update a staging-set seed. Promotes to working buffer when done."""
        self.seed2actor[seed].add(actor_index)
        partial_score = self.partial_seed_scores_buffer[actor_index].get(seed, 0)
        partial_num_steps = self.partial_seed_steps_buffer[actor_index].get(seed, 0)

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score) * num_steps
        if running_num_steps > 0:
            merged_score = merged_score / float(running_num_steps)

        if done:
            # Promote to working buffer if score is high enough
            seed_idx = self._next_buffer_index
            if (
                self.seed_scores[seed_idx] <= merged_score
                or self.unseen_seed_weights[seed_idx] > 0
            ):
                self.unseen_seed_weights[seed_idx] = 0.0
                self.working_seed_set.discard(self.seeds[seed_idx])
                self.working_seed_set.add(seed)
                self.seeds[seed_idx] = seed
                self.seed2index[seed] = seed_idx
                self.seed_scores[seed_idx] = merged_score
                self.partial_seed_scores[:, seed_idx] = 0.0
                self.partial_seed_steps[:, seed_idx] = 0
                self.seed_staleness[seed_idx] = (
                    self.running_sample_count
                    - self.seed2timestamp_buffer.get(seed, self.running_sample_count)
                )
                self.working_seed_buffer_size = min(
                    self.working_seed_buffer_size + 1, self.seed_buffer_size
                )
            # Clean up staging data
            for a in self.seed2actor[seed]:
                self.partial_seed_scores_buffer[a].pop(seed, None)
                self.partial_seed_steps_buffer[a].pop(seed, None)
            self.seed2timestamp_buffer.pop(seed, None)
            self.seed2actor.pop(seed, None)
            self.staging_seed_set.discard(seed)
        else:
            self.partial_seed_scores_buffer[actor_index][seed] = merged_score
            self.partial_seed_steps_buffer[actor_index][seed] = running_num_steps

    # ── Update with rollout data (called after each rollout) ─────────────

    @property
    def requires_value_buffers(self) -> bool:
        return self.strategy in [
            "gae",
            "value_l1",
            "signed_value_loss",
            "positive_value_loss",
            "one_step_td_error",
        ]

    def update_with_rollouts(self, rollouts):
        """
        Score levels using the training rollout data.

        Iterates through the rollout, identifies episode boundaries (done),
        and computes a score for each completed episode.
        """
        if not self._has_working_seed_buffer:
            return

        if self.strategy in ["random", "off"]:
            return

        score_function = self._get_score_function()

        level_seeds = rollouts.level_seeds
        policy_logits = rollouts.action_log_dist
        total_steps, num_actors = policy_logits.shape[:2]
        done = ~(rollouts.masks > 0)

        for actor_index in range(num_actors):
            start_t = 0
            done_steps = done[:, actor_index].nonzero()[:, 0]

            for t in done_steps:
                if not start_t < total_steps:
                    break
                if t == 0:
                    continue

                seed_t = level_seeds[start_t, actor_index].item()

                # Build kwargs for score function
                kwargs = self._build_score_kwargs(
                    rollouts, actor_index, start_t, t, seed_t, done=True
                )
                score = score_function(**kwargs)
                num_steps = len(kwargs["episode_logits"])

                if seed_t in self.staging_seed_set:
                    self._partial_update_seed_score_buffer(
                        actor_index, seed_t, score, num_steps, done=True
                    )
                else:
                    self._partial_update_seed_score(
                        actor_index, seed_t, score, num_steps, done=True
                    )
                start_t = t.item()

            # Handle partial episode (not yet done)
            if start_t < total_steps:
                seed_t = level_seeds[start_t, actor_index].item()
                kwargs = self._build_score_kwargs(
                    rollouts, actor_index, start_t, total_steps, seed_t, done=False
                )
                score = score_function(**kwargs)
                num_steps = len(kwargs["episode_logits"])

                if seed_t in self.staging_seed_set:
                    self._partial_update_seed_score_buffer(
                        actor_index, seed_t, score, num_steps, done=False
                    )
                else:
                    self._partial_update_seed_score(
                        actor_index, seed_t, score, num_steps, done=False
                    )

    def _build_score_kwargs(
        self, rollouts, actor_index, start_t, end_t, seed, done
    ) -> dict:
        episode_logits = rollouts.action_log_dist[start_t:end_t, actor_index]
        kwargs = {
            "actor_index": actor_index,
            "done": done,
            "episode_logits": torch.log_softmax(episode_logits, -1),
            "seed": seed,
        }
        if self.requires_value_buffers:
            kwargs["returns"] = rollouts.returns[start_t:end_t, actor_index]
            kwargs["rewards"] = rollouts.rewards[start_t:end_t, actor_index]
            kwargs["value_preds"] = rollouts.value_preds[start_t:end_t, actor_index]
        return kwargs

    def _get_score_function(self):
        if self.strategy == "value_l1":
            return self._value_l1
        elif self.strategy == "gae":
            return self._gae
        elif self.strategy == "positive_value_loss":
            return self._positive_value_loss
        elif self.strategy == "signed_value_loss":
            return self._signed_value_loss
        elif self.strategy == "one_step_td_error":
            return self._one_step_td_error
        elif self.strategy == "policy_entropy":
            return self._policy_entropy
        elif self.strategy == "least_confidence":
            return self._least_confidence
        elif self.strategy == "min_margin":
            return self._min_margin
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

    # ── Score functions (from DCD) ───────────────────────────────────────

    @staticmethod
    def _value_l1(**kwargs) -> float:
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        return (returns - value_preds).abs().mean().item()

    @staticmethod
    def _gae(**kwargs) -> float:
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        return (returns - value_preds).mean().item()

    @staticmethod
    def _positive_value_loss(**kwargs) -> float:
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        return (returns - value_preds).clamp(0).mean().item()

    @staticmethod
    def _signed_value_loss(**kwargs) -> float:
        returns = kwargs["returns"]
        value_preds = kwargs["value_preds"]
        return (returns - value_preds).mean().item()

    @staticmethod
    def _one_step_td_error(**kwargs) -> float:
        rewards = kwargs["rewards"]
        value_preds = kwargs["value_preds"]
        if len(rewards) < 2:
            return abs((rewards[0] - value_preds[0]).item())
        td = (rewards[:-1] + 0.999 * value_preds[1:] - value_preds[:-1]).abs()
        return td.mean().item()

    @staticmethod
    def _policy_entropy(**kwargs) -> float:
        logits = kwargs["episode_logits"]
        probs = logits.exp()
        entropy = -(probs * logits).sum(-1)
        num_actions = logits.shape[-1]
        max_ent = np.log(num_actions) if num_actions > 1 else 1.0
        return (entropy / max_ent).mean().item()

    @staticmethod
    def _least_confidence(**kwargs) -> float:
        logits = kwargs["episode_logits"]
        return (1 - logits.exp().max(-1)[0]).mean().item()

    @staticmethod
    def _min_margin(**kwargs) -> float:
        logits = kwargs["episode_logits"]
        top2 = logits.exp().topk(2, dim=-1)[0]
        margin = (top2[:, 0] - top2[:, 1]).mean()
        return (1 - margin).item()

    # ── After PPO update ─────────────────────────────────────────────────

    def after_update(self):
        """Reset partial scores (weights changed, logits are stale)."""
        if not self._has_working_seed_buffer:
            return

        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self._partial_update_seed_score(
                        actor_index, self.seeds[seed_idx], 0, 0, done=True
                    )

        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

        # Reset staging buffer partial scores
        for actor_index in range(self.num_actors):
            staging_seeds = list(self.partial_seed_scores_buffer[actor_index].keys())
            for seed in staging_seeds:
                if self.partial_seed_scores_buffer[actor_index][seed] > 0:
                    self._partial_update_seed_score_buffer(
                        actor_index, seed, 0, 0, done=True
                    )

    # ── Staleness ────────────────────────────────────────────────────────

    def _update_staleness(self, selected_idx: int):
        if self.staleness_coef > 0:
            self.seed_staleness += 1
            self.seed_staleness[selected_idx] = 0

    # ── Sampling weights ─────────────────────────────────────────────────

    def sample_weights(self) -> np.ndarray:
        weights = self._score_transform_fn(
            self.score_transform, self.temperature, self.seed_scores
        )
        weights = weights * (1 - self.unseen_seed_weights)

        z = np.sum(weights)
        if z > 0:
            weights /= z
        else:
            weights = np.ones_like(weights) / len(weights)
            weights = weights * (1 - self.unseen_seed_weights)
            s = np.sum(weights)
            if s > 0:
                weights /= s
            else:
                return np.ones(len(weights)) / len(weights)

        if self.staleness_coef > 0:
            stale_w = self._score_transform_fn(
                self.staleness_transform,
                self.staleness_temperature,
                self.seed_staleness,
            )
            stale_w = stale_w * (1 - self.unseen_seed_weights)
            z = np.sum(stale_w)
            if z > 0:
                stale_w /= z
            else:
                stale_w = (1.0 / len(stale_w)) * (1 - self.unseen_seed_weights)

            weights = (1 - self.staleness_coef) * weights + self.staleness_coef * stale_w

        return weights

    @staticmethod
    def _score_transform_fn(
        transform: str, temperature: float, scores: np.ndarray
    ) -> np.ndarray:
        if transform == "constant":
            return np.ones_like(scores)
        elif transform == "max":
            w = np.zeros_like(scores)
            valid = scores.copy()
            argmax = np.random.choice(np.flatnonzero(np.isclose(valid, valid.max())))
            w[argmax] = 1.0
            return w
        elif transform == "eps_greedy":
            w = np.zeros_like(scores)
            w[scores.argmax()] = 0.95
            w += 0.05 / len(scores)
            return w
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            return 1.0 / ranks.astype(np.float64) ** (1.0 / max(temperature, 1e-8))
        elif transform == "power":
            eps = 1e-3
            return (np.maximum(scores, 0) + eps) ** (1.0 / max(temperature, 1e-8))
        elif transform == "softmax":
            return np.exp(scores / max(temperature, 1e-8))
        else:
            return np.ones_like(scores)

    # ── Stats / Diagnostics ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, float]:
        seen = self.unseen_seed_weights == 0
        n_seen = int(seen.sum())
        if n_seen == 0:
            return {
                "plr/buffer_filled": 0.0,
                "plr/buffer_size": self.working_seed_buffer_size,
                "plr/mean_score": 0.0,
                "plr/max_score": 0.0,
                "plr/staging_size": len(self.staging_seed_set),
            }
        s = self.seed_scores[seen]
        return {
            "plr/buffer_filled": self._proportion_filled,
            "plr/buffer_size": self.working_seed_buffer_size,
            "plr/mean_score": float(s.mean()),
            "plr/max_score": float(s.max()),
            "plr/staging_size": len(self.staging_seed_set),
        }
