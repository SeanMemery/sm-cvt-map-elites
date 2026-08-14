from __future__ import annotations

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.secondary import (
    run_stage_end_evaluation,
    select_for_secondary_eval,
    select_stage_seeds,
)


def build_archive():
    registry = {}
    archive = CVTArchive(
        np.array([[0.0], [0.3], [0.6], [0.9]], dtype=float),
        registry,
        np.random.default_rng(2),
    )
    candidates = []
    for index, fitness in enumerate([1.0, 2.0, 3.0, 4.0]):
        candidate = Candidate(
            id=f"cand_{index}",
            code=f"code {index}",
            descriptor_raw=[float(index)],
            descriptor_norm=[float(index) / 3.0],
            primary_fitness=fitness,
            created_at_step=index,
        )
        registry[candidate.id] = candidate
        archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
        candidates.append(candidate)
    return archive, candidates


def test_select_for_secondary_eval_top_p1():
    archive, candidates = build_archive()
    rng = np.random.default_rng(0)

    selected = select_for_secondary_eval(archive, p1=0.5, rng=rng)

    # p1=0.5 of 4 elites → ceil(2) = 2, should be the two highest primary fitness
    assert len(selected) == 2
    assert {c.id for c in selected} == {"cand_2", "cand_3"}


def test_select_for_secondary_eval_full():
    archive, candidates = build_archive()
    rng = np.random.default_rng(0)

    selected = select_for_secondary_eval(archive, p1=1.0, rng=rng)
    assert len(selected) == 4


def test_select_for_secondary_eval_zero_p1():
    archive, candidates = build_archive()
    rng = np.random.default_rng(0)

    selected = select_for_secondary_eval(archive, p1=0.0, rng=rng)
    assert selected == []


def test_select_stage_seeds_top_p2():
    _, candidates = build_archive()
    secondary_scores = {"cand_0": 0.5, "cand_1": 3.0, "cand_2": 1.0, "cand_3": 2.0}
    for c in candidates:
        c.secondary_fitness = secondary_scores[c.id]

    seeds = select_stage_seeds(candidates, p2=0.5)

    # p2=0.5 of 4 → ceil(2) = 2, top 2 by secondary fitness
    assert len(seeds) == 2
    assert {s.id for s in seeds} == {"cand_1", "cand_3"}


def test_select_stage_seeds_zero_p2():
    _, candidates = build_archive()
    for c in candidates:
        c.secondary_fitness = 1.0
    assert select_stage_seeds(candidates, p2=0.0) == []


def test_run_stage_end_evaluation_scores_and_seeds():
    archive, candidates = build_archive()
    rng = np.random.default_rng(0)

    secondary_scores = {"cand_0": 0.5, "cand_1": 1.0, "cand_2": 2.0, "cand_3": 3.0}
    secondary_validation_scores = {"cand_0": 10.0, "cand_1": 20.0, "cand_2": 30.0, "cand_3": 40.0}

    def secondary(c: Candidate) -> float:
        return secondary_scores[c.id]

    def secondary_validation(c: Candidate) -> float:
        return secondary_validation_scores[c.id]

    # p1=1.0 → all 4 evaluated; p2=0.5 → top 2 by secondary fitness become seeds
    evaluated, seeds = run_stage_end_evaluation(
        archive=archive,
        p1=1.0,
        p2=0.5,
        secondary_fitness=secondary,
        secondary_validation_fitness=secondary_validation,
        rng=rng,
    )

    assert len(evaluated) == 4
    assert all(c.secondary_fitness is not None for c in evaluated)
    assert all(c.secondary_validation_fitness is not None for c in evaluated)

    assert len(seeds) == 2
    assert {s.id for s in seeds} == {"cand_2", "cand_3"}


def test_run_stage_end_evaluation_no_validation_fitness():
    archive, _ = build_archive()
    rng = np.random.default_rng(0)

    evaluated, seeds = run_stage_end_evaluation(
        archive=archive,
        p1=0.5,
        p2=1.0,
        secondary_fitness=lambda c: 1.0,
        secondary_validation_fitness=None,
        rng=rng,
    )

    assert len(evaluated) == 2
    assert all(c.secondary_validation_fitness is None for c in evaluated)
    assert len(seeds) == 2
