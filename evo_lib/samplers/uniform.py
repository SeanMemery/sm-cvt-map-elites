from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from evo_lib.samplers.sampler import Item, Sampler, ScoreGetter


class UniformSampler(Sampler):
    """Sample every item with equal probability."""

    def sample_index(
        self,
        rng: np.random.Generator,
        items: Sequence[Item],
        *,
        score_getter: ScoreGetter[Item] | None = None,
    ) -> int:
        if not items:
            raise ValueError("Cannot sample from an empty sequence")
        return int(rng.integers(0, len(items)))
