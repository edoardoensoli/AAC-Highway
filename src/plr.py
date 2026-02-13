"""
Prioritized Level Replay (PLR) — Modular Implementation
=========================================================

Based on: Jiang, Grefenstette & Rocktäschel (2021)
"Prioritized Level Replay" — arXiv:2010.03934

This module provides algorithm-agnostic PLR components that can be used
with any RL algorithm (PPO, DQN, etc).

Components
----------
1. LevelSampler       — core PLR sampling logic (score transforms, staleness,
                         unseen/replay split)
2. LevelScorer        — scoring strategies (value_l1, gae, td_error,
                         policy_entropy, min_margin)
3. LevelBuffer        — per-level rollout storage for on-policy scoring
4. RollingStats       — lightweight performance tracker

Reference: github.com/facebookresearch/level-replay
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# 1. Level Sampler  (algorithm-agnostic)
# ──────────────────────────────────────────────────────────────────────────────
class LevelSampler:
    """
    Prioritized Level Replay sampler.

    Given N levels, maintains scores and produces a categorical distribution
    over levels biased toward those with highest estimated learning potential.

    Parameters
    ----------
    num_levels : int
        Total number of training levels (environment configs).
    strategy : str
        Scoring strategy name. Only affects how `update_with_rollout` picks
        the score function.  One of:
            value_l1, gae, one_step_td_error, policy_entropy,
            least_confidence, min_margin.
    rho : float
        Proportion of levels that must be seen before replay kicks in.
    nu : float
        When replay is active, probability of sampling a *new* unseen level
        (exploration in level space).  Paper default: 0.5 for fixed schedule.
    replay_schedule : str
        'fixed' — replay once rho fraction seen, with prob 1-nu.
        'proportional' — replay prob = proportion_seen.
    score_transform : str
        How raw scores → sampling weights.  One of:
            rank, power, softmax, eps_greedy, max.
    temperature : float
        Temperature for rank / power / softmax transforms.
    eps : float
        Floor probability for eps-greedy mixing.
    alpha : float
        EMA coefficient for score updates:
            score ← alpha * new + (1-alpha) * old
        Paper default: 1.0 (no smoothing — latest score only).
    staleness_coef : float
        Weight of staleness bonus in sampling distribution.
        Paper default: 0.1 for value_l1 + rank.
    staleness_transform : str
        Transform for staleness values (same options as score_transform).
    staleness_temperature : float
        Temperature for the staleness transform.
    """

    def __init__(
        self,
        num_levels: int,
        strategy: str = "value_l1",
        rho: float = 0.5,
        nu: float = 0.5,
        replay_schedule: str = "fixed",
        score_transform: str = "rank",
        temperature: float = 0.1,
        eps: float = 0.05,
        alpha: float = 1.0,
        staleness_coef: float = 0.1,
        staleness_transform: str = "power",
        staleness_temperature: float = 1.0,
    ):
        self.num_levels = num_levels
        self.strategy = strategy
        self.rho = rho
        self.nu = nu
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        # ── storage ──
        self.scores = np.zeros(num_levels, dtype=np.float64)
        self.staleness = np.zeros(num_levels, dtype=np.float64)
        self.unseen_weights = np.ones(num_levels, dtype=np.float64)  # 1 = unseen
        self.max_returns = np.full(num_levels, -np.inf, dtype=np.float64)

    # ── public API ───────────────────────────────────────────────────────────

    def sample(self) -> int:
        """Sample a level index according to PLR."""
        idx, _ = self.sample_with_decision()
        return idx

    def sample_with_decision(self) -> Tuple[int, bool]:
        """
        PLR⊥ (Robust PLR) sampling decision.

        Returns
        -------
        level_idx : int
            The level to interact with.
        is_replay : bool
            ``True`` if this is a **replay** level — the caller should
            perform gradient updates (``model.learn()``) on this level.
            ``False`` if this is an **explore / discovery** level — the
            caller should only collect a rollout for scoring, with **no**
            gradient updates, to avoid noise from random levels.

        In standard PLR both branches train; in PLR⊥ only replay trains.
        """
        num_unseen = int(self.unseen_weights.sum())
        proportion_seen = (self.num_levels - num_unseen) / self.num_levels

        if self.replay_schedule == "fixed":
            if proportion_seen >= self.rho:
                if np.random.rand() > self.nu or num_unseen == 0:
                    return self._sample_replay(), True    # REPLAY
            return self._sample_unseen(), False             # EXPLORE
        else:  # proportional schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay(), True
            return self._sample_unseen(), False

    def update_score(self, level_idx: int, score: float):
        """Update the score for a level using EMA."""
        self.unseen_weights[level_idx] = 0.0  # mark as seen
        old = self.scores[level_idx]
        self.scores[level_idx] = self.alpha * score + (1 - self.alpha) * old

    def update_with_maxmc(
        self,
        level_idx: int,
        max_episode_return: float,
        mean_reward: float,
    ):
        """
        Update score using MaxMC regret.

        MaxMC tracks the **maximum** episodic return ever achieved on each
        level as a "ground truth" upper bound.  The score (regret) is then:

        .. math::
            \\text{score} = \\max(\\text{max\_return}[i] - \\bar{R}, \\; 0)

        This avoids the bias of ``estimated_regret`` which relies on the
        (potentially inaccurate) learned value function.  MaxMC aligns
        more closely with the true minimax-regret objective used in the
        theoretical proofs for PLR⊥.
        """
        self.max_returns[level_idx] = max(
            self.max_returns[level_idx], max_episode_return
        )
        regret = max(self.max_returns[level_idx] - mean_reward, 0.0)
        self.update_score(level_idx, regret)

    def update_staleness(self, selected_idx: int):
        """Increment staleness for all levels, reset for the active one."""
        if self.staleness_coef > 0:
            self.staleness += 1
            self.staleness[selected_idx] = 0

    @property
    def num_seen(self) -> int:
        return int((self.unseen_weights == 0).sum())

    # ── sampling internals ───────────────────────────────────────────────────

    def _sample_replay(self) -> int:
        weights = self._compute_sample_weights()
        if np.isclose(weights.sum(), 0):
            weights = np.ones_like(weights) / len(weights)
        idx = np.random.choice(self.num_levels, p=weights)
        self.update_staleness(idx)
        return int(idx)

    def _sample_unseen(self) -> int:
        if self.unseen_weights.sum() == 0:
            return np.random.randint(self.num_levels)
        w = self.unseen_weights / self.unseen_weights.sum()
        idx = np.random.choice(self.num_levels, p=w)
        self.update_staleness(idx)
        return int(idx)

    def _compute_sample_weights(self) -> np.ndarray:
        """Compute the full sampling distribution over all levels."""
        # Score component (only for seen levels)
        score_w = self._apply_transform(
            self.score_transform, self.temperature, self.scores
        )
        score_w = score_w * (1 - self.unseen_weights)  # zero out unseen
        z = score_w.sum()
        if z > 0:
            score_w /= z

        # Staleness component
        if self.staleness_coef > 0:
            stale_w = self._apply_transform(
                self.staleness_transform,
                self.staleness_temperature,
                self.staleness,
            )
            stale_w = stale_w * (1 - self.unseen_weights)
            z = stale_w.sum()
            if z > 0:
                stale_w /= z
            weights = (1 - self.staleness_coef) * score_w + self.staleness_coef * stale_w
        else:
            weights = score_w

        return weights

    @staticmethod
    def _apply_transform(transform: str, temperature: float, scores: np.ndarray) -> np.ndarray:
        """Apply a score transform to produce sampling weights."""
        if transform == "constant":
            return np.ones_like(scores)
        elif transform == "max":
            w = np.zeros_like(scores)
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            w[argmax] = 1.0
            return w
        elif transform == "eps_greedy":
            w = np.zeros_like(scores)
            w[scores.argmax()] = 1.0
            return w
        elif transform == "rank":
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            return 1.0 / ranks.astype(np.float64) ** (1.0 / max(temperature, 1e-8))
        elif transform == "power":
            return (np.maximum(scores, 1e-8)) ** (1.0 / max(temperature, 1e-8))
        elif transform == "softmax":
            s = scores / max(temperature, 1e-8)
            s = s - s.max()
            return np.exp(s)
        else:
            return np.maximum(scores, 1e-8)

    # ── stats / persistence ──────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, float]:
        seen_mask = self.unseen_weights == 0
        n_seen = int(seen_mask.sum())
        if n_seen == 0:
            return {
                "plr/seen": 0,
                "plr/unseen": self.num_levels,
                "plr/mean_score": 0.0,
                "plr/max_score": 0.0,
                "plr/min_score": 0.0,
                "plr/std_score": 0.0,
            }
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
            pickle.dump(
                {
                    "scores": self.scores,
                    "staleness": self.staleness,
                    "unseen_weights": self.unseen_weights,
                    "max_returns": self.max_returns,
                    "strategy": self.strategy,
                    "rho": self.rho,
                    "nu": self.nu,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.scores = d["scores"]
        self.staleness = d["staleness"]
        self.unseen_weights = d["unseen_weights"]
        self.max_returns = d.get(
            "max_returns",
            np.full(self.num_levels, -np.inf, dtype=np.float64),
        )

    def __repr__(self) -> str:
        return (
            f"LevelSampler(levels={self.num_levels}, seen={self.num_seen}, "
            f"strategy={self.strategy}, rho={self.rho}, nu={self.nu})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Level Scorer  (computes learning-potential scores from rollout data)
# ──────────────────────────────────────────────────────────────────────────────
class LevelScorer:
    """
    Compute PLR scores from rollout data.

    All methods are static and take numpy/torch arrays.
    The score should reflect *learning potential* — how much the agent
    can still learn from this level.
    """

    @staticmethod
    def value_l1(
        returns: np.ndarray,
        value_preds: np.ndarray,
    ) -> float:
        """
        L1 advantage: mean |G_t - V(s_t)|.
        High when value function is inaccurate → high learning potential.
        This is the recommended strategy from the PLR paper.
        """
        advantages = np.abs(returns - value_preds)
        return float(advantages.mean())

    @staticmethod
    def gae(
        returns: np.ndarray,
        value_preds: np.ndarray,
    ) -> float:
        """
        Mean signed advantage A_t = G_t - V(s_t).
        Levels with high positive advantage = agent underestimates.
        """
        advantages = returns - value_preds
        return float(np.abs(advantages.mean()))

    @staticmethod
    def estimated_regret(
        returns: np.ndarray,
        value_preds: np.ndarray,
    ) -> float:
        """
        Estimated regret (positive value loss): mean max(G_t − V(s_t), 0).

        Only counts states where the agent **underestimates** value,
        capturing missed opportunities.  This is the "positive part"
        of the advantage, which the PLR paper identifies as a proxy
        for regret.
        """
        advantages = returns - value_preds
        positive = np.maximum(advantages, 0)
        return float(positive.mean())

    @staticmethod
    def one_step_td_error(
        rewards: np.ndarray,
        value_preds: np.ndarray,
        gamma: float = 0.99,
    ) -> float:
        """
        Mean |r_t + γ V(s_{t+1}) - V(s_t)|.
        Directly measures temporal-difference surprise.
        """
        if len(rewards) < 2:
            return 0.0
        td = np.abs(
            rewards[:-1] + gamma * value_preds[1:] - value_preds[:-1]
        )
        return float(td.mean())

    @staticmethod
    def policy_entropy(
        action_logits: np.ndarray,
    ) -> float:
        """
        Normalised average entropy of the policy.
        High entropy → agent is uncertain → high learning potential.
        """
        # logits → log_softmax → entropy
        logits = torch.as_tensor(action_logits, dtype=torch.float32)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        num_actions = logits.shape[-1]
        max_entropy = np.log(num_actions)
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    @staticmethod
    def min_margin(
        action_logits: np.ndarray,
    ) -> float:
        """
        1 - mean margin between top-2 action probabilities.
        Small margin → agent is indecisive → high learning potential.
        """
        logits = torch.as_tensor(action_logits, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        top2 = probs.topk(2, dim=-1)[0]
        margin = (top2[:, 0] - top2[:, 1]).mean()
        return float(1.0 - margin)

    @staticmethod
    def least_confidence(
        action_logits: np.ndarray,
    ) -> float:
        """
        1 - mean max probability.
        Low confidence → high learning potential.
        """
        logits = torch.as_tensor(action_logits, dtype=torch.float32)
        probs = torch.softmax(logits, dim=-1)
        return float((1.0 - probs.max(dim=-1)[0]).mean())

    @staticmethod
    def max_mc(
        episode_rewards: List[float],
    ) -> float:
        """
        Maximum Monte-Carlo return from the rollout’s episodes.

        Used with :meth:`LevelSampler.update_with_maxmc` to implement
        MaxMC scoring.  The sampler tracks the historical maximum and
        computes regret = max_ever − current_mean, which provides a
        "ground truth" learning-potential signal free from value-function
        bias.
        """
        if not episode_rewards:
            return 0.0
        return float(max(episode_rewards))


# ──────────────────────────────────────────────────────────────────────────────
# 3. Level Rollout Buffer  (stores per-level data for scoring)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class LevelRollout:
    """Data collected from one level for PLR scoring."""

    level_idx: int
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    value_preds: List[float] = field(default_factory=list)
    action_logits: List[np.ndarray] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    crash_count: int = 0

    def add_step(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float = 0.0,
        logits: Optional[np.ndarray] = None,
    ):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.value_preds.append(value)
        if logits is not None:
            self.action_logits.append(logits)

    def end_episode(self, crashed: bool = False):
        ep_start = sum(self.episode_lengths)
        ep_len = len(self.rewards) - ep_start
        if ep_len > 0:
            self.episode_lengths.append(ep_len)
            self.episode_rewards.append(sum(self.rewards[ep_start:]))
        if crashed:
            self.crash_count += 1

    @property
    def total_steps(self) -> int:
        return len(self.rewards)

    @property
    def mean_reward(self) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))

    @property
    def mean_ep_length(self) -> float:
        if not self.episode_lengths:
            return 0.0
        return float(np.mean(self.episode_lengths))

    @property
    def max_episode_return(self) -> float:
        """Maximum episodic return observed in this rollout."""
        if not self.episode_rewards:
            return 0.0
        return float(max(self.episode_rewards))

    def compute_returns(self, gamma: float = 0.99) -> np.ndarray:
        """Compute discounted returns per step, respecting episode boundaries."""
        returns = np.zeros(len(self.rewards), dtype=np.float64)
        idx = 0
        for length in self.episode_lengths:
            R = 0.0
            for t in reversed(range(length)):
                R = self.rewards[idx + t] + gamma * R
                returns[idx + t] = R
            idx += length
        return returns

    def compute_score(
        self,
        strategy: str,
        gamma: float = 0.99,
    ) -> float:
        """Compute PLR score using the specified strategy."""
        if self.total_steps == 0:
            return 0.0

        values = np.array(self.value_preds, dtype=np.float64)
        returns = self.compute_returns(gamma)
        rewards = np.array(self.rewards, dtype=np.float64)

        if strategy == "value_l1":
            return LevelScorer.value_l1(returns, values)
        elif strategy == "estimated_regret":
            return LevelScorer.estimated_regret(returns, values)
        elif strategy == "gae":
            return LevelScorer.gae(returns, values)
        elif strategy == "one_step_td_error":
            return LevelScorer.one_step_td_error(rewards, values, gamma)
        elif strategy == "policy_entropy":
            if not self.action_logits:
                return 0.0
            logits = np.array(self.action_logits)
            return LevelScorer.policy_entropy(logits)
        elif strategy == "min_margin":
            if not self.action_logits:
                return 0.0
            logits = np.array(self.action_logits)
            return LevelScorer.min_margin(logits)
        elif strategy == "least_confidence":
            if not self.action_logits:
                return 0.0
            logits = np.array(self.action_logits)
            return LevelScorer.least_confidence(logits)
        elif strategy == "max_mc":
            return LevelScorer.max_mc(self.episode_rewards)
        else:
            raise ValueError(f"Unknown scoring strategy: {strategy}")


# ──────────────────────────────────────────────────────────────────────────────
# 4. Rolling Statistics Tracker
# ──────────────────────────────────────────────────────────────────────────────
class RollingStats:
    """Lightweight performance tracker with rolling windows."""

    def __init__(self, window: int = 20):
        self.window = window
        self.rewards: List[float] = []
        self.crash_counts: List[int] = []
        self.episode_counts: List[int] = []
        self.ep_lengths: List[float] = []
        self.best_reward = -float("inf")
        self.worst_reward = float("inf")
        self.total_evals = 0

    def update(
        self,
        mean_reward: float,
        crash_count: int,
        n_episodes: int,
        mean_ep_length: float,
    ):
        self.rewards.append(mean_reward)
        self.crash_counts.append(crash_count)
        self.episode_counts.append(n_episodes)
        self.ep_lengths.append(mean_ep_length)
        self.total_evals += 1
        self.best_reward = max(self.best_reward, mean_reward)
        self.worst_reward = min(self.worst_reward, mean_reward)

    @property
    def avg_reward(self) -> float:
        w = self.rewards[-self.window :]
        return float(np.mean(w)) if w else 0.0

    @property
    def avg_reward_all(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def reward_trend(self) -> str:
        if len(self.rewards) < 2 * self.window:
            return "…"
        recent = np.mean(self.rewards[-self.window :])
        prev = np.mean(self.rewards[-2 * self.window : -self.window])
        diff = recent - prev
        if diff > 0.5:
            return f"▲ +{diff:.1f}"
        elif diff < -0.5:
            return f"▼ {diff:.1f}"
        return "━ stable"

    @property
    def crash_rate(self) -> float:
        """Crash rate over last `window` evaluations."""
        w = self.window
        crashes = self.crash_counts[-w:]
        episodes = self.episode_counts[-w:]
        total_crashes = sum(crashes)
        total_eps = sum(episodes)
        return 100.0 * total_crashes / max(total_eps, 1)

    @property
    def survival_rate(self) -> float:
        return 100.0 - self.crash_rate

    @property
    def avg_ep_length(self) -> float:
        w = self.ep_lengths[-self.window :]
        return float(np.mean(w)) if w else 0.0
