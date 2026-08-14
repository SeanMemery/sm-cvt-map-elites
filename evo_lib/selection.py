from __future__ import annotations

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.config import SamplingSchedulePhaseConfig, SamplingStrategy, SamplingWeightConfig


def sample_parents(
    archive: CVTArchive,
    n: int,
    grid_strategy: SamplingStrategy,
    elite_strategy: SamplingStrategy,
    grid_distance_threshold: float,
    *,
    current_step: int = 0,
    grid_sampling_weights: list[SamplingWeightConfig] | None = None,
    elite_sampling_weights: list[SamplingWeightConfig] | None = None,
    grid_sampling_schedule: list[SamplingSchedulePhaseConfig] | None = None,
    elite_sampling_schedule: list[SamplingSchedulePhaseConfig] | None = None,
    emitter_name: str | None = None,
    curiosity_temperature: float = 0.7,
) -> list[Candidate]:
    cell_ids = archive.occupied_cells()
    if not cell_ids:
        raise ValueError("Archive is empty")
    sampled_cells = _sample_cells(
        archive=archive,
        cell_ids=cell_ids,
        n=n,
        strategy=grid_strategy,
        distance_threshold=grid_distance_threshold,
        current_step=current_step,
        sampling_weights=grid_sampling_weights,
        sampling_schedule=grid_sampling_schedule,
        curiosity_temperature=curiosity_temperature,
        emitter_name=emitter_name,
    )
    return [
        _sample_cell_elite(
            archive=archive,
            cell_id=cell_id,
            strategy=elite_strategy,
            current_step=current_step,
            sampling_weights=elite_sampling_weights,
            sampling_schedule=elite_sampling_schedule,
            emitter_name=emitter_name,
            curiosity_temperature=curiosity_temperature,
        )
        for cell_id in sampled_cells
    ]


def _sample_cells(
    archive: CVTArchive,
    cell_ids: list[int],
    n: int,
    strategy: SamplingStrategy,
    distance_threshold: float,
    *,
    current_step: int,
    sampling_weights: list[SamplingWeightConfig] | None,
    sampling_schedule: list[SamplingSchedulePhaseConfig] | None,
    curiosity_temperature: float = 0.7,
    emitter_name: str | None = None,
) -> list[int]:
    if n <= 0:
        return []
    available_ids = list(cell_ids)
    sampled_cells: list[int] = []
    while available_ids and len(sampled_cells) < n:
        candidate_ids = _eligible_cells(
            archive=archive,
            candidate_ids=available_ids,
            selected_ids=sampled_cells,
            distance_threshold=distance_threshold,
        )
        if not candidate_ids:
            candidate_ids = list(available_ids)
        chosen_cell = _sample_single_cell(
            archive=archive,
            cell_ids=candidate_ids,
            strategy=strategy,
            current_step=current_step,
            sampling_weights=sampling_weights,
            sampling_schedule=sampling_schedule,
            curiosity_temperature=curiosity_temperature,
            emitter_name=emitter_name,
        )
        sampled_cells.append(chosen_cell)
        available_ids = [cell_id for cell_id in available_ids if cell_id != chosen_cell]
    if not sampled_cells:
        raise ValueError("Archive has no sampleable cells")
    while len(sampled_cells) < n:
        sampled_cells.append(sampled_cells[-1])
    return sampled_cells


def _eligible_cells(
    archive: CVTArchive,
    candidate_ids: list[int],
    selected_ids: list[int],
    distance_threshold: float,
) -> list[int]:
    if not selected_ids:
        return list(candidate_ids)
    eligible: list[int] = []
    for cell_id in candidate_ids:
        centroid = archive.centroids[cell_id]
        if all(
            float(np.linalg.norm(centroid - archive.centroids[selected_id])) >= distance_threshold
            for selected_id in selected_ids
        ):
            eligible.append(cell_id)
    return eligible


def _sample_single_cell(
    archive: CVTArchive,
    cell_ids: list[int],
    strategy: SamplingStrategy,
    *,
    current_step: int,
    sampling_weights: list[SamplingWeightConfig] | None,
    sampling_schedule: list[SamplingSchedulePhaseConfig] | None,
    curiosity_temperature: float = 0.7,
    emitter_name: str | None = None,
) -> int:
    resolved_strategy = _resolve_strategy(
        archive=archive,
        strategy=strategy,
        current_step=current_step,
        sampling_weights=sampling_weights,
        sampling_schedule=sampling_schedule,
    )
    if resolved_strategy == "uniform":
        index = int(archive._rng.integers(0, len(cell_ids)))
        return cell_ids[index]
    if resolved_strategy == "fitness_weighted":
        fitnesses = np.asarray(
            [archive.get_cell_elite(cell_id).primary_fitness or 0.0 for cell_id in cell_ids],
            dtype=float,
        )
        index = int(archive._rng.choice(len(cell_ids), size=1, replace=True, p=_to_probabilities(fitnesses))[0])
        return cell_ids[index]
    if resolved_strategy == "curiosity_weighted":
        curiosity = np.asarray(
            [
                archive.get_cell_elite(cell_id).curiosity_score
                if archive.get_cell_elite(cell_id).curiosity_score is not None
                else 0.0
                for cell_id in cell_ids
            ],
            dtype=float,
        )
        index = int(archive._rng.choice(len(cell_ids), size=1, replace=True, p=_to_probabilities(curiosity, curiosity_temperature))[0])
        return cell_ids[index]
    if resolved_strategy == "emitter_curiosity_weighted":
        # For grid sampling: use the best elite's emitter curiosity score for each cell
        curiosity = np.asarray(
            [
                float(archive.get_cell_elite(cell_id).emitter_curiosity_scores.get(emitter_name, 0.0))
                if emitter_name and archive.get_cell_elite(cell_id).emitter_curiosity_scores
                else (archive.get_cell_elite(cell_id).curiosity_score or 0.0)
                for cell_id in cell_ids
            ],
            dtype=float,
        )
        index = int(archive._rng.choice(len(cell_ids), size=1, replace=True, p=_to_probabilities(curiosity, curiosity_temperature))[0])
        return cell_ids[index]
    raise ValueError(f"Unsupported grid sampling strategy: {strategy}")


def _sample_cell_elite(
    archive: CVTArchive,
    cell_id: int,
    strategy: SamplingStrategy,
    *,
    current_step: int,
    sampling_weights: list[SamplingWeightConfig] | None,
    sampling_schedule: list[SamplingSchedulePhaseConfig] | None,
    emitter_name: str | None,
    curiosity_temperature: float = 0.7,
) -> Candidate:
    elites = archive.get_cell_elites(cell_id)
    if not elites:
        raise ValueError(f"Cell {cell_id} has no elites")
    resolved_strategy = _resolve_strategy(
        archive=archive,
        strategy=strategy,
        current_step=current_step,
        sampling_weights=sampling_weights,
        sampling_schedule=sampling_schedule,
    )
    if resolved_strategy == "uniform":
        index = int(archive._rng.integers(0, len(elites)))
        return elites[index]
    if resolved_strategy == "fitness_weighted":
        fitnesses = np.asarray([elite.primary_fitness or 0.0 for elite in elites], dtype=float)
        index = int(archive._rng.choice(len(elites), size=1, replace=True, p=_to_probabilities(fitnesses))[0])
        return elites[index]
    if resolved_strategy == "best":
        # Deterministic: always return the elite with the highest primary fitness
        return max(elites, key=lambda e: e.primary_fitness or 0.0)
    if resolved_strategy == "curiosity_weighted":
        curiosity = np.asarray(
            [elite.curiosity_score if elite.curiosity_score is not None else 0.0 for elite in elites],
            dtype=float,
        )
        index = int(archive._rng.choice(len(elites), size=1, replace=True, p=_to_probabilities(curiosity, curiosity_temperature))[0])
        return elites[index]
    if resolved_strategy == "emitter_curiosity_weighted":
        if not emitter_name:
            raise ValueError("emitter_curiosity_weighted elite sampling requires emitter_name")
        curiosity = np.asarray(
            [float(elite.emitter_curiosity_scores.get(emitter_name, 0.0)) for elite in elites],
            dtype=float,
        )
        index = int(archive._rng.choice(len(elites), size=1, replace=True, p=_to_probabilities(curiosity, curiosity_temperature))[0])
        return elites[index]
    raise ValueError(f"Unsupported elite sampling strategy: {strategy}")


def _resolve_strategy(
    *,
    archive: CVTArchive,
    strategy: SamplingStrategy,
    current_step: int,
    sampling_weights: list[SamplingWeightConfig] | None,
    sampling_schedule: list[SamplingSchedulePhaseConfig] | None,
) -> str:
    if strategy != "weighted":
        return strategy
    active_weights = _resolve_active_weights(
        current_step=current_step,
        sampling_weights=sampling_weights,
        sampling_schedule=sampling_schedule,
    )
    probabilities = _to_probabilities(np.asarray([entry.weight for entry in active_weights], dtype=float))
    index = int(archive._rng.choice(len(active_weights), size=1, replace=True, p=probabilities)[0])
    return active_weights[index].strategy


def _resolve_active_weights(
    *,
    current_step: int,
    sampling_weights: list[SamplingWeightConfig] | None,
    sampling_schedule: list[SamplingSchedulePhaseConfig] | None,
) -> list[SamplingWeightConfig]:
    if sampling_schedule:
        for phase in sampling_schedule:
            if phase.until_step is None or current_step <= phase.until_step:
                return phase.weights
        return sampling_schedule[-1].weights
    if sampling_weights:
        return sampling_weights
    raise ValueError("Weighted sampling requires configured sampling weights")


def _to_probabilities(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Convert raw scores to a probability distribution with optional temperature scaling.

    Temperature < 1 sharpens the distribution (favours high-scoring candidates).
    Temperature > 1 flattens it (more uniform).  Temperature = 1 gives raw proportional weights.
    """
    min_value = float(np.min(values))
    if min_value <= 0:
        values = values - min_value + 1e-9
    if float(np.sum(values)) == 0.0:
        values = np.ones_like(values, dtype=float)
    if temperature != 1.0 and temperature > 0:
        values = values ** (1.0 / temperature)
    return values / np.sum(values)
