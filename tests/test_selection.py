from __future__ import annotations

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.config import SamplingSchedulePhaseConfig, SamplingWeightConfig
from evo_lib.selection import sample_parents


def test_two_stage_parent_sampling_prefers_higher_fitness_elites_within_cell():
    registry = {}
    archive = CVTArchive(
        np.array([[0.1], [0.9]], dtype=float),
        registry,
        np.random.default_rng(11),
        elites_per_cell=3,
    )

    candidates = [
        Candidate(id="cell0_low", code="a", descriptor_norm=[0.1], descriptor_raw=[0.1], primary_fitness=1.0),
        Candidate(id="cell0_high", code="b", descriptor_norm=[0.1], descriptor_raw=[0.1], primary_fitness=5.0),
        Candidate(id="cell1_low", code="c", descriptor_norm=[0.9], descriptor_raw=[0.9], primary_fitness=1.0),
        Candidate(id="cell1_high", code="d", descriptor_norm=[0.9], descriptor_raw=[0.9], primary_fitness=6.0),
    ]
    for candidate in candidates:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    sampled = sample_parents(
        archive=archive,
        n=400,
        grid_strategy="uniform",
        elite_strategy="fitness_weighted",
        grid_distance_threshold=0.01,
    )

    counts = {}
    for candidate in sampled:
        counts[candidate.id] = counts.get(candidate.id, 0) + 1

    assert counts.get("cell0_high", 0) > counts.get("cell0_low", 0)
    assert counts.get("cell1_high", 0) > counts.get("cell1_low", 0)


def test_config_defaults_to_two_stage_sampling(base_config):
    loaded = base_config.from_dict(base_config.to_dict())
    assert loaded.grid_selection_strategy == "uniform"
    assert loaded.elite_selection_strategy == "fitness_weighted"
    assert loaded.grid_distance_threshold == 0.05


def test_config_round_trips_curiosity_and_weighted_schedule(base_config):
    payload = base_config.to_dict()
    payload["curiosity_default"] = 1.25
    payload["curiosity_reward"] = 2.0
    payload["curiosity_penalty"] = 0.4
    payload["elite_selection_strategy"] = "weighted"
    payload["elite_sampling_schedule"] = [
        {"until_step": 20, "weights": [{"strategy": "curiosity_weighted", "weight": 1.0}]},
        {"until_step": None, "weights": [{"strategy": "fitness_weighted", "weight": 1.0}]},
    ]
    loaded = base_config.from_dict(payload)

    assert loaded.curiosity_default == 1.25
    assert loaded.curiosity_reward == 2.0
    assert loaded.curiosity_penalty == 0.4
    assert loaded.elite_selection_strategy == "weighted"
    assert loaded.elite_sampling_schedule is not None
    assert loaded.elite_sampling_schedule[0].weights[0].strategy == "curiosity_weighted"
    assert loaded.elite_sampling_schedule[1].weights[0].strategy == "fitness_weighted"


def test_grid_sampling_prefers_cells_above_distance_threshold():
    registry = {}
    archive = CVTArchive(
        np.array([[0.0], [0.02], [0.5]], dtype=float),
        registry,
        np.random.default_rng(3),
        elites_per_cell=1,
    )
    candidates = [
        Candidate(id="a", code="a", descriptor_norm=[0.0], descriptor_raw=[0.0], primary_fitness=1.0),
        Candidate(id="b", code="b", descriptor_norm=[0.02], descriptor_raw=[0.02], primary_fitness=1.0),
        Candidate(id="c", code="c", descriptor_norm=[0.5], descriptor_raw=[0.5], primary_fitness=1.0),
    ]
    for candidate in candidates:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    sampled = sample_parents(
        archive=archive,
        n=2,
        grid_strategy="uniform",
        elite_strategy="uniform",
        grid_distance_threshold=0.1,
    )
    cell_ids = [candidate.cell_id for candidate in sampled]
    assert len(set(cell_ids)) == 2
    assert abs(archive.centroids[cell_ids[0]][0] - archive.centroids[cell_ids[1]][0]) >= 0.1


def test_curiosity_weighted_elite_sampling_prefers_higher_curiosity():
    registry = {}
    archive = CVTArchive(
        np.array([[0.1]], dtype=float),
        registry,
        np.random.default_rng(5),
        elites_per_cell=3,
    )
    candidates = [
        Candidate(
            id="low_curiosity",
            code="a",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=10.0,
            curiosity_score=0.0,
        ),
        Candidate(
            id="high_curiosity",
            code="b",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=1.0,
            curiosity_score=10.0,
        ),
    ]
    for candidate in candidates:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    sampled = sample_parents(
        archive=archive,
        n=400,
        grid_strategy="uniform",
        elite_strategy="curiosity_weighted",
        grid_distance_threshold=0.01,
    )

    counts = {}
    for candidate in sampled:
        counts[candidate.id] = counts.get(candidate.id, 0) + 1

    assert counts.get("high_curiosity", 0) > counts.get("low_curiosity", 0)


def test_weighted_sampling_schedule_can_shift_from_curiosity_to_fitness():
    registry = {}
    archive = CVTArchive(
        np.array([[0.1]], dtype=float),
        registry,
        np.random.default_rng(9),
        elites_per_cell=3,
    )
    candidates = [
        Candidate(
            id="fitness_favorite",
            code="a",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=10.0,
            curiosity_score=0.0,
        ),
        Candidate(
            id="curiosity_favorite",
            code="b",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=1.0,
            curiosity_score=10.0,
        ),
    ]
    for candidate in candidates:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    schedule = [
        SamplingSchedulePhaseConfig(
            until_step=10,
            weights=[SamplingWeightConfig(strategy="curiosity_weighted", weight=1.0)],
        ),
        SamplingSchedulePhaseConfig(
            until_step=None,
            weights=[SamplingWeightConfig(strategy="fitness_weighted", weight=1.0)],
        ),
    ]

    early = sample_parents(
        archive=archive,
        n=200,
        grid_strategy="uniform",
        elite_strategy="weighted",
        grid_distance_threshold=0.01,
        current_step=5,
        elite_sampling_schedule=schedule,
    )
    late = sample_parents(
        archive=archive,
        n=200,
        grid_strategy="uniform",
        elite_strategy="weighted",
        grid_distance_threshold=0.01,
        current_step=50,
        elite_sampling_schedule=schedule,
    )

    early_counts = {}
    for candidate in early:
        early_counts[candidate.id] = early_counts.get(candidate.id, 0) + 1
    late_counts = {}
    for candidate in late:
        late_counts[candidate.id] = late_counts.get(candidate.id, 0) + 1

    assert early_counts.get("curiosity_favorite", 0) > early_counts.get("fitness_favorite", 0)
    assert late_counts.get("fitness_favorite", 0) > late_counts.get("curiosity_favorite", 0)


def test_emitter_curiosity_weighted_elite_sampling_prefers_higher_emitter_curiosity():
    registry = {}
    archive = CVTArchive(
        np.array([[0.1]], dtype=float),
        registry,
        np.random.default_rng(12),
        elites_per_cell=3,
    )
    candidates = [
        Candidate(
            id="runtime_low",
            code="a",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=10.0,
            emitter_curiosity_scores={"runtime": 1.0, "algo": 6.0},
        ),
        Candidate(
            id="runtime_high",
            code="b",
            descriptor_norm=[0.1],
            descriptor_raw=[0.1],
            primary_fitness=1.0,
            emitter_curiosity_scores={"runtime": 9.0, "algo": 1.0},
        ),
    ]
    for candidate in candidates:
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)

    sampled = sample_parents(
        archive=archive,
        n=400,
        grid_strategy="uniform",
        elite_strategy="emitter_curiosity_weighted",
        grid_distance_threshold=0.01,
        emitter_name="runtime",
    )

    counts = {}
    for candidate in sampled:
        counts[candidate.id] = counts.get(candidate.id, 0) + 1

    assert counts.get("runtime_high", 0) > counts.get("runtime_low", 0)
