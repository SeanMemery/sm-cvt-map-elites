from __future__ import annotations

from evo_lib.samplers.weighted import WeightedSampler


class EmitterCuriosityWeightedSampler(WeightedSampler):
    """Sample using curiosity accumulated by the active emitter."""
