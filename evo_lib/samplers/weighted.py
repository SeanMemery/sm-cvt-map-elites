from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evo_lib.samplers.sampler import Item, Sampler, ScoreGetter, probabilities, required_scores


class WeightedSampler(Sampler):
    """Base implementation for strategies that weight items by a score."""

    def sample_index(
        self,
        rng: np.random.Generator,
        items: Sequence[Item],
        *,
        score_getter: ScoreGetter[Item] | None = None,
    ) -> int:
        if not items:
            raise ValueError("Cannot sample from an empty sequence")
        weights = probabilities(required_scores(items, score_getter))
        return int(rng.choice(len(items), size=1, replace=True, p=weights)[0])
