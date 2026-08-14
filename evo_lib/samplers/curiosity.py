from __future__ import annotations

from evo_lib.samplers.weighted import WeightedSampler


class CuriosityWeightedSampler(WeightedSampler):
    """Sample using a caller-provided curiosity score."""
