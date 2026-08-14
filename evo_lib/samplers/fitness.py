from __future__ import annotations

from evo_lib.samplers.weighted import WeightedSampler


class FitnessWeightedSampler(WeightedSampler):
    """Sample using a caller-provided primary-fitness score."""
