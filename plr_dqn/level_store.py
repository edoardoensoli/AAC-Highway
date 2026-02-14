"""
Level Store — maps seed ↔ level data.

Faithful adaptation of ``facebookresearch/dcd/level_replay/level_store.py``.

A "level" in our case is a string encoding of the highway-env config
(e.g. "lanes=3,vehicles=25,density=1.2,politeness=0.5").

The store assigns each unique level a monotonically increasing integer seed,
tracks parent lineage (for ACCEL, unused here), and supports FIFO eviction
when a maximum capacity is set.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class LevelStore:
    """Manages a mapping: seed → level, where level is any hashable object."""

    def __init__(self, max_size: Optional[int] = None):
        self.max_size = max_size
        self.seed2level: Dict[int, Any] = {}
        self.level2seed: Dict[Any, int] = {}
        self.seed2parent: Dict[int, list] = {}
        self.next_seed = 1
        self.levels: set = set()

    def __len__(self) -> int:
        return len(self.levels)

    # ── insert ───────────────────────────────────────────────────────────────

    def _insert(self, level: Any, parent_seed: Optional[int] = None) -> Optional[int]:
        if level is None:
            return None

        if level not in self.levels:
            # FIFO eviction if at capacity
            if self.max_size is not None:
                while len(self.levels) >= self.max_size:
                    first_idx = next(iter(self.seed2level))
                    self._remove(first_idx)

            seed = self.next_seed
            self.seed2level[seed] = level
            if parent_seed is not None:
                self.seed2parent[seed] = (
                    self.seed2parent.get(parent_seed, [])
                    + [self.seed2level.get(parent_seed)]
                )
            else:
                self.seed2parent[seed] = []
            self.level2seed[level] = seed
            self.levels.add(level)
            self.next_seed += 1
            return seed
        else:
            return self.level2seed[level]

    def insert(
        self,
        level: Any,
        parent_seeds: Optional[Sequence[Optional[int]]] = None,
    ) -> Any:
        """Insert one level or a list of levels. Returns seed(s)."""
        if hasattr(level, "__iter__") and not isinstance(level, str):
            idx = []
            for i, lv in enumerate(level):
                ps = parent_seeds[i] if parent_seeds is not None else None
                idx.append(self._insert(lv, ps))
            return idx
        else:
            return self._insert(level)

    # ── remove ───────────────────────────────────────────────────────────────

    def _remove(self, level_seed: int):
        if level_seed is None or level_seed < 0:
            return
        level = self.seed2level[level_seed]
        self.levels.discard(level)
        self.seed2level.pop(level_seed, None)
        self.level2seed.pop(level, None)
        self.seed2parent.pop(level_seed, None)

    def remove(self, level_seed):
        if hasattr(level_seed, "__iter__"):
            for s in level_seed:
                self._remove(s)
        else:
            self._remove(level_seed)

    # ── query ────────────────────────────────────────────────────────────────

    def get_level(self, level_seed: int) -> Any:
        return self.seed2level[level_seed]

    def reconcile_seeds(self, level_seeds):
        """Remove seeds that are no longer in the given set."""
        old_seeds = set(self.seed2level)
        new_seeds = set(level_seeds)
        if len(new_seeds) == 1 and -1 in new_seeds:
            return
        for seed in old_seeds - new_seeds:
            self._remove(seed)
