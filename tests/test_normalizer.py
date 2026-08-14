from __future__ import annotations

from evo_lib.config import NormalizerConfig
from evo_lib.normalizer import DescriptorNormalizer


def test_normalizer_quantiles_and_clipping():
    normalizer = DescriptorNormalizer(
        NormalizerConfig(dim=2, lower_quantile=0.0, upper_quantile=1.0, history_size=10, clip=True)
    )
    normalizer.update([0.0, 10.0])
    normalizer.update([10.0, 20.0])

    normalized = normalizer.normalize([5.0, 30.0])

    assert normalized == [0.5, 1.0]
    assert normalizer.bounds() == {"lower": [0.0, 10.0], "upper": [10.0, 20.0]}


def test_normalizer_zero_width_returns_midpoint_and_roundtrip():
    normalizer = DescriptorNormalizer(NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=5))
    normalizer.update([3.0])

    assert normalizer.normalize([3.0]) == [0.5]

    state = normalizer.state_dict()
    restored = DescriptorNormalizer(NormalizerConfig(dim=1))
    restored.load_state_dict(state)

    assert restored.normalize([3.0]) == [0.5]
