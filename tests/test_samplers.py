from __future__ import annotations

import numpy as np
import pytest

from evo_lib.samplers import (
    CuriosityWeightedSampler,
    EmitterCuriosityWeightedSampler,
    FitnessWeightedSampler,
    Sampler,
    UniformSampler,
)


def test_sampling_strategies_share_the_generic_sampler_interface():
    for sampler in (
        UniformSampler(),
        FitnessWeightedSampler(),
        CuriosityWeightedSampler(),
        EmitterCuriosityWeightedSampler(),
    ):
        assert isinstance(sampler, Sampler)


def test_uniform_sampler_returns_an_item_from_the_input():
    result = UniformSampler().sample(np.random.default_rng(1), ["a", "b", "c"])
    assert result in {"a", "b", "c"}


def test_weighted_samplers_prefer_larger_scores():
    items = ["low", "high"]
    for sampler in (FitnessWeightedSampler(), CuriosityWeightedSampler(), EmitterCuriosityWeightedSampler()):
        rng = np.random.default_rng(17)
        samples = [
            sampler.sample(rng, items, score_getter=lambda item: 1.0 if item == "low" else 10.0)
            for _ in range(300)
        ]
        assert samples.count("high") > samples.count("low")


def test_weighted_samplers_require_scores():
    with pytest.raises(ValueError, match="score_getter"):
        FitnessWeightedSampler().sample(np.random.default_rng(1), ["candidate"])
