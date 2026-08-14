from __future__ import annotations

import math
from typing import Callable

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.errors import validate_stats

CandidateProgressCallback = Callable[[Candidate, int, int], None]


def select_for_secondary_eval(
    archive: CVTArchive,
    p1: float,
    rng: np.random.Generator,
) -> list[Candidate]:
    """Select the top P1 fraction of archive elites by primary fitness for secondary evaluation."""
    elites = archive.all_elites()
    if not elites or p1 <= 0.0:
        return []
    n = max(1, math.ceil(p1 * len(elites)))
    return sorted(elites, key=lambda c: c.primary_fitness or float("-inf"), reverse=True)[:n]


def select_stage_seeds(
    evaluated: list[Candidate],
    p2: float,
    min_seeds: int = 4,
    p2_threshold: int = 8,
    rng: np.random.Generator | None = None,
) -> list[Candidate]:
    """Select seeds for the next stage from secondary-evaluated candidates.

    If more than p2_threshold candidates were evaluated, apply P2 (floored at
    min_seeds). Otherwise take all evaluated candidates up to min_seeds.

    If all secondary fitness scores are identical (including all-zero from failed
    evaluations), fall back to random sampling — this is rare but prevents an
    empty archive from killing the next stage.
    """
    if not evaluated:
        return []
    n_evaluated = len(evaluated)
    if n_evaluated > p2_threshold:
        n = max(min_seeds, math.ceil(p2 * n_evaluated))
    else:
        n = min(min_seeds, n_evaluated)

    scores = [c.secondary_fitness for c in evaluated]
    all_equal = len(set(s for s in scores if s is not None)) <= 1

    if all_equal:
        # All scores identical — sort by primary fitness, then shuffle within ties
        by_primary = sorted(evaluated, key=lambda c: c.primary_fitness or float("-inf"), reverse=True)
        if rng is not None:
            # Shuffle candidates with equal primary fitness (most likely all equal too)
            primary_scores = [c.primary_fitness for c in by_primary]
            if len(set(p for p in primary_scores if p is not None)) <= 1:
                indices = list(range(len(by_primary)))
                rng.shuffle(indices)
                by_primary = [by_primary[i] for i in indices]
        return by_primary[:n]

    return sorted(evaluated, key=lambda c: c.secondary_fitness or float("-inf"), reverse=True)[:n]


def run_stage_end_evaluation(
    *,
    archive: CVTArchive,
    p1: float,
    p2: float,
    secondary_fitness: Callable[[Candidate], float],
    secondary_validation_fitness: Callable[[Candidate], float] | None,
    rng: np.random.Generator,
    min_seeds: int = 4,
    p2_threshold: int = 8,
    per_candidate_callback: CandidateProgressCallback | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Run end-of-stage secondary evaluation.

    Returns:
        (evaluated, seeds) where evaluated is every candidate that received a
        secondary fitness score and seeds is the P2-fraction subset to carry
        forward as initial candidates for the next stage.
    """
    selected = select_for_secondary_eval(archive, p1, rng)
    if not selected:
        return [], []
    total = len(selected)
    for index, candidate in enumerate(selected):
        raw = secondary_fitness(candidate)
        if isinstance(raw, tuple):
            score, stats = raw
            candidate.secondary_fitness = float(score)
            if stats is not None:
                secondary_stats = validate_stats(stats)
                if candidate.stats is None:
                    candidate.stats = secondary_stats
                else:
                    candidate.stats.update(secondary_stats)
        else:
            candidate.secondary_fitness = float(raw)
        if secondary_validation_fitness is not None:
            val_raw = secondary_validation_fitness(candidate)
            candidate.secondary_validation_fitness = float(val_raw[0] if isinstance(val_raw, tuple) else val_raw)
        if per_candidate_callback is not None:
            per_candidate_callback(candidate, index + 1, total)
    seeds = select_stage_seeds(selected, p2, min_seeds=min_seeds, p2_threshold=p2_threshold, rng=rng)
    return selected, seeds
