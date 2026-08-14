"""Generic interfaces and helpers for archive sampling strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

Item = TypeVar("Item")
ScoreGetter = Callable[[Item], float]


class Sampler(ABC):
    """Select one item from a sequence using a particular sampling policy."""

    @abstractmethod
    def sample_index(
        self,
        rng: np.random.Generator,
        items: Sequence[Item],
        *,
        score_getter: ScoreGetter[Item] | None = None,
    ) -> int:
        """Return the index of one sampled item.

        Strategies that use weights require ``score_getter``.  Callers retain
        ownership of their items, making this interface usable for both cell
        identifiers and candidate objects.
        """

    def sample(
        self,
        rng: np.random.Generator,
        items: Sequence[Item],
        *,
        score_getter: ScoreGetter[Item] | None = None,
    ) -> Item:
        """Select and return one item."""
        if not items:
            raise ValueError("Cannot sample from an empty sequence")
        return items[self.sample_index(rng, items, score_getter=score_getter)]


def probabilities(values: Sequence[float]) -> np.ndarray:
    """Convert arbitrary numeric scores to stable non-negative probabilities."""
    weights = np.asarray(values, dtype=float)
    if weights.size == 0:
        raise ValueError("Cannot calculate probabilities for an empty sequence")
    min_value = float(np.min(weights))
    if min_value <= 0:
        weights = weights - min_value + 1e-9
    if float(np.sum(weights)) == 0.0:
        weights = np.ones_like(weights, dtype=float)
    return weights / np.sum(weights)


def required_scores(items: Sequence[Item], score_getter: ScoreGetter[Item] | None) -> list[float]:
    """Read scores for a weighted sampler or report a clear caller error."""
    if score_getter is None:
        raise ValueError("This sampler requires a score_getter")
    return [float(score_getter(item)) for item in items]
