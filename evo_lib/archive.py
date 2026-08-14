from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

from evo_lib.candidate import Candidate
from evo_lib.normalizer import DescriptorNormalizer


def generate_centroids(
    *,
    descriptor_dim: int,
    num_centroids: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = rng.random((samples, descriptor_dim))
    kmeans = KMeans(init="k-means++", n_clusters=num_centroids, n_init=1, random_state=seed)
    kmeans.fit(points)
    return np.asarray(kmeans.cluster_centers_, dtype=float)


@dataclass
class ArchiveState:
    centroids: np.ndarray
    cell_to_elite_ids: dict[int, list[str]]
    cell_to_candidate_ids: dict[int, list[str]]
    elites_per_cell: int


@dataclass
class InsertResult:
    inserted: bool
    cell_id: int
    replaced_candidate_id: str | None = None


@dataclass
class RemapResult:
    before_occupied_cells: int
    after_occupied_cells: int
    moved_count: int
    collision_count: int
    lost_candidate_ids: list[str]
    cell_changes: dict[str, dict[str, int | None]]


class CVTArchive:
    def __init__(
        self,
        centroids: np.ndarray,
        candidate_registry: dict[str, Candidate],
        rng: np.random.Generator,
        elites_per_cell: int = 1,
    ):
        self.centroids = np.asarray(centroids, dtype=float)
        self._candidate_registry = candidate_registry
        self._rng = rng
        self.elites_per_cell = int(elites_per_cell)
        self._cell_to_elite_ids: dict[int, list[str]] = {}
        self._cell_to_candidate_ids: dict[int, list[str]] = {}
        self._tree = KDTree(self.centroids, leaf_size=30, metric="euclidean")

    @classmethod
    def create(
        cls,
        *,
        descriptor_dim: int,
        num_centroids: int,
        samples: int,
        seed: int,
        candidate_registry: dict[str, Candidate],
        rng: np.random.Generator,
        elites_per_cell: int = 1,
    ) -> "CVTArchive":
        centroids = generate_centroids(
            descriptor_dim=descriptor_dim,
            num_centroids=num_centroids,
            samples=samples,
            seed=seed,
        )
        return cls(
            centroids=centroids,
            candidate_registry=candidate_registry,
            rng=rng,
            elites_per_cell=elites_per_cell,
        )

    def insert(
        self,
        candidate: Candidate,
        descriptor_norm: list[float],
        fitness: float,
    ) -> InsertResult:
        del fitness
        cell_id = self._nearest_cell_id(descriptor_norm)
        candidate.cell_id = cell_id
        cell_candidates = list(self._cell_to_candidate_ids.get(cell_id, []))
        if candidate.id not in cell_candidates:
            cell_candidates.append(candidate.id)
        self._cell_to_candidate_ids[cell_id] = cell_candidates
        previous_elite_ids = list(self._cell_to_elite_ids.get(cell_id, []))
        surviving_ids = self._recompute_cell_elites(cell_id)
        if candidate.id not in surviving_ids:
            return InsertResult(inserted=False, cell_id=cell_id, replaced_candidate_id=None)
        replaced_candidate_id = None
        demoted_ids = [
            candidate_id
            for candidate_id in previous_elite_ids
            if candidate_id not in surviving_ids and candidate_id != candidate.id
        ]
        if demoted_ids:
            replaced_candidate_id = demoted_ids[0]
        return InsertResult(inserted=True, cell_id=cell_id, replaced_candidate_id=replaced_candidate_id)

    def remove(self, candidate_id: str) -> None:
        for cell_id, candidate_ids in list(self._cell_to_candidate_ids.items()):
            if candidate_id not in candidate_ids:
                continue
            remaining_ids = [existing_id for existing_id in candidate_ids if existing_id != candidate_id]
            if remaining_ids:
                self._cell_to_candidate_ids[cell_id] = remaining_ids
                self._recompute_cell_elites(cell_id)
            else:
                del self._cell_to_candidate_ids[cell_id]
                self._cell_to_elite_ids.pop(cell_id, None)
            break

    def sample_elites(self, n: int, strategy: str = "uniform") -> list[Candidate]:
        elites = self.all_elites()
        if not elites:
            raise ValueError("Archive is empty")
        if strategy == "uniform":
            indexes = self._rng.integers(0, len(elites), size=n)
        elif strategy == "fitness_weighted":
            fitnesses = np.asarray([elite.primary_fitness or 0.0 for elite in elites], dtype=float)
            min_fitness = float(np.min(fitnesses))
            if min_fitness <= 0:
                fitnesses = fitnesses - min_fitness + 1e-9
            if float(np.sum(fitnesses)) == 0.0:
                fitnesses = np.ones_like(fitnesses, dtype=float)
            probabilities = fitnesses / np.sum(fitnesses)
            indexes = self._rng.choice(len(elites), size=n, replace=True, p=probabilities)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")
        return [elites[int(index)] for index in indexes]

    def all_elites(self) -> list[Candidate]:
        elites: list[Candidate] = []
        for cell_id in sorted(self._cell_to_elite_ids.keys()):
            elites.extend(self.get_cell_elites(cell_id))
        return elites

    def all_candidates(self) -> list[Candidate]:
        candidates: list[Candidate] = []
        for cell_id in sorted(self._cell_to_candidate_ids.keys()):
            candidates.extend(self.get_cell_candidates(cell_id))
        return candidates

    def occupied_cells(self) -> list[int]:
        return sorted(self._cell_to_elite_ids.keys())

    def elite_count(self) -> int:
        return sum(len(candidate_ids) for candidate_ids in self._cell_to_elite_ids.values())

    def population_count(self) -> int:
        return sum(len(candidate_ids) for candidate_ids in self._cell_to_candidate_ids.values())

    def get_cell_elite(self, cell_id: int) -> Candidate | None:
        elites = self.get_cell_elites(cell_id)
        if not elites:
            return None
        return elites[0]

    def get_cell_elites(self, cell_id: int) -> list[Candidate]:
        candidate_ids = self._cell_to_elite_ids.get(cell_id, [])
        return [self._candidate_registry[candidate_id] for candidate_id in candidate_ids]

    def get_cell_candidates(self, cell_id: int) -> list[Candidate]:
        candidate_ids = self._cell_to_candidate_ids.get(cell_id, [])
        return [self._candidate_registry[candidate_id] for candidate_id in candidate_ids]

    def clear(self) -> None:
        self._cell_to_elite_ids.clear()
        self._cell_to_candidate_ids.clear()

    def remap(
        self,
        candidates: list[Candidate],
        normalizer: DescriptorNormalizer,
    ) -> RemapResult:
        before_map = {candidate.id: candidate.cell_id for candidate in candidates}
        before_occupied_cells = len(self._cell_to_elite_ids)
        self.clear()
        collisions = 0
        lost_candidate_ids: list[str] = []
        for candidate in candidates:
            if candidate.descriptor_raw is None or candidate.primary_fitness is None:
                raise ValueError(f"Candidate {candidate.id} cannot be remapped without descriptor_raw and primary_fitness")
            candidate.descriptor_norm = normalizer.normalize(candidate.descriptor_raw)
            result = self.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
            if result.inserted and result.replaced_candidate_id is not None:
                collisions += 1
                lost_candidate_ids.append(result.replaced_candidate_id)
        active_ids = {candidate.id for candidate in self.all_candidates()}
        after_map = {
            candidate.id: self._candidate_registry[candidate.id].cell_id
            for candidate in candidates
            if candidate.id in active_ids
        }
        moved_count = 0
        cell_changes: dict[str, dict[str, int | None]] = {}
        for candidate_id, previous_cell in before_map.items():
            current_cell = after_map.get(candidate_id)
            cell_changes[candidate_id] = {"before": previous_cell, "after": current_cell}
            if previous_cell != current_cell:
                moved_count += 1
        return RemapResult(
            before_occupied_cells=before_occupied_cells,
            after_occupied_cells=len(self._cell_to_elite_ids),
            moved_count=moved_count,
            collision_count=collisions,
            lost_candidate_ids=lost_candidate_ids,
            cell_changes=cell_changes,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "centroids": self.centroids,
            "cell_to_elite_ids": {cell_id: list(candidate_ids) for cell_id, candidate_ids in self._cell_to_elite_ids.items()},
            "cell_to_candidate_ids": {cell_id: list(candidate_ids) for cell_id, candidate_ids in self._cell_to_candidate_ids.items()},
            "elites_per_cell": self.elites_per_cell,
        }

    @classmethod
    def from_state_dict(
        cls,
        state: dict[str, Any],
        *,
        candidate_registry: dict[str, Candidate],
        rng: np.random.Generator,
    ) -> "CVTArchive":
        raw_population_mapping = state.get("cell_to_candidate_ids")
        raw_elite_mapping = state.get("cell_to_elite_ids")
        if raw_population_mapping is None:
            raw_single = state.get("cell_to_candidate_id", {})
            raw_population_mapping = {int(k): [v] for k, v in raw_single.items()}
        if raw_elite_mapping is None:
            raw_elite_mapping = raw_population_mapping
        archive = cls(
            centroids=state["centroids"],
            candidate_registry=candidate_registry,
            rng=rng,
            elites_per_cell=int(state.get("elites_per_cell", 1)),
        )
        archive._cell_to_elite_ids = {
            int(k): list(v)
            for k, v in raw_elite_mapping.items()
        }
        archive._cell_to_candidate_ids = {
            int(k): list(v)
            for k, v in raw_population_mapping.items()
        }
        return archive

    def _nearest_cell_id(self, descriptor_norm: list[float]) -> int:
        query = np.asarray([descriptor_norm], dtype=float)
        return int(self._tree.query(query, k=1)[1][0][0])

    def _sort_candidate_ids_by_primary(self, candidate_ids: list[str]) -> list[str]:
        return sorted(
            candidate_ids,
            key=lambda candidate_id: self._sort_key(candidate_id),
            reverse=True,
        )

    def _recompute_cell_elites(self, cell_id: int) -> list[str]:
        candidate_ids = self._cell_to_candidate_ids.get(cell_id, [])
        if not candidate_ids:
            self._cell_to_elite_ids.pop(cell_id, None)
            return []
        ordered_ids = self._sort_candidate_ids_by_primary(candidate_ids)
        surviving_ids = ordered_ids[: self.elites_per_cell]
        self._cell_to_elite_ids[cell_id] = surviving_ids
        return surviving_ids

    def _sort_key(self, candidate_id: str) -> tuple[float, int]:
        candidate = self._candidate_registry[candidate_id]
        primary = float(candidate.primary_fitness if candidate.primary_fitness is not None else float("-inf"))
        created_at_step = int(candidate.created_at_step or 0)
        return (primary, -created_at_step)
