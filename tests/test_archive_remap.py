from __future__ import annotations

import numpy as np

from evo_lib.archive import CVTArchive, generate_centroids
from evo_lib.candidate import Candidate
from evo_lib.config import NormalizerConfig
from evo_lib.normalizer import DescriptorNormalizer


def test_generate_centroids_is_deterministic():
    first = generate_centroids(descriptor_dim=2, num_centroids=4, samples=16, seed=5)
    second = generate_centroids(descriptor_dim=2, num_centroids=4, samples=16, seed=5)
    assert np.allclose(first, second)


def test_archive_insert_replace_remove_and_remap_changes_cells():
    registry = {}
    rng = np.random.default_rng(3)
    archive = CVTArchive(np.array([[0.1], [0.9]], dtype=float), registry, rng, elites_per_cell=1)
    normalizer = DescriptorNormalizer(
        NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=16)
    )

    for value in [0.0, 1.0]:
        normalizer.update([value])

    low = Candidate(id="low", code="a", descriptor_raw=[1.0], primary_fitness=1.0)
    low.descriptor_norm = normalizer.normalize(low.descriptor_raw)
    registry[low.id] = low
    assert archive.insert(low, low.descriptor_norm, low.primary_fitness).inserted is True
    assert low.cell_id == 1

    high = Candidate(id="high", code="b", descriptor_raw=[1.0], primary_fitness=2.0)
    high.descriptor_norm = normalizer.normalize(high.descriptor_raw)
    registry[high.id] = high
    assert archive.insert(high, high.descriptor_norm, high.primary_fitness).inserted is True
    assert archive.get_cell_elite(1).id == "high"
    assert [candidate.id for candidate in archive.get_cell_candidates(1)] == ["low", "high"]

    archive.remove("high")
    assert archive.get_cell_elite(1).id == "low"
    assert [candidate.id for candidate in archive.get_cell_candidates(1)] == ["low"]

    registry[low.id] = low
    archive.insert(low, low.descriptor_norm, low.primary_fitness)
    for value in [100.0, 100.0]:
        normalizer.update([value])
    remap_result = archive.remap(archive.all_candidates(), normalizer)

    assert low.cell_id == 0
    assert archive.get_cell_elite(0).id == "low"
    assert remap_result.moved_count >= 1


def test_archive_retains_top_k_elites_per_cell_in_order():
    registry = {}
    rng = np.random.default_rng(5)
    archive = CVTArchive(np.array([[0.9]], dtype=float), registry, rng, elites_per_cell=3)
    normalizer = DescriptorNormalizer(
        NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=16)
    )
    for value in [0.0, 1.0]:
        normalizer.update([value])

    candidates = []
    for candidate_id, fitness in [("a", 1.0), ("b", 4.0), ("c", 2.0), ("d", 3.0)]:
        candidate = Candidate(id=candidate_id, code=candidate_id, descriptor_raw=[1.0], primary_fitness=fitness)
        candidate.descriptor_norm = normalizer.normalize(candidate.descriptor_raw)
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
        candidates.append(candidate)

    assert [candidate.id for candidate in archive.get_cell_elites(0)] == ["b", "d", "c"]
    assert archive.get_cell_elite(0).id == "b"
    assert archive.elite_count() == 3
    assert [candidate.id for candidate in archive.get_cell_candidates(0)] == ["a", "b", "c", "d"]


def test_archive_retains_non_elites_and_remaps_them():
    registry = {}
    rng = np.random.default_rng(8)
    archive = CVTArchive(np.array([[0.1], [0.9]], dtype=float), registry, rng, elites_per_cell=1)
    normalizer = DescriptorNormalizer(
        NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=16)
    )
    for value in [0.0, 1.0]:
        normalizer.update([value])

    low = Candidate(id="low", code="low", descriptor_raw=[1.0], primary_fitness=1.0)
    low.descriptor_norm = normalizer.normalize(low.descriptor_raw)
    high = Candidate(id="high", code="high", descriptor_raw=[1.0], primary_fitness=2.0)
    high.descriptor_norm = normalizer.normalize(high.descriptor_raw)
    for candidate in [low, high]:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    assert archive.get_cell_elite(1).id == "high"
    assert [candidate.id for candidate in archive.get_cell_candidates(1)] == ["low", "high"]

    for value in [100.0, 100.0]:
        normalizer.update([value])
    remap_result = archive.remap(archive.all_candidates(), normalizer)

    assert low.cell_id == 0
    assert high.cell_id == 0
    assert archive.get_cell_elite(0).id == "high"
    assert [candidate.id for candidate in archive.get_cell_candidates(0)] == ["low", "high"]
    assert remap_result.moved_count >= 2
