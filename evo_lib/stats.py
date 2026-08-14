from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evo_lib.archive import CVTArchive


@dataclass
class StatsTracker:
    generated_count: int = 0
    accepted_count: int = 0
    llm_error_count: int = 0
    parse_failure_count: int = 0
    validation_failure_count: int = 0
    descriptor_failure_count: int = 0
    fitness_failure_count: int = 0
    secondary_eval_count: int = 0
    prune_count: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "generated_count": self.generated_count,
            "accepted_count": self.accepted_count,
            "llm_error_count": self.llm_error_count,
            "parse_failure_count": self.parse_failure_count,
            "validation_failure_count": self.validation_failure_count,
            "descriptor_failure_count": self.descriptor_failure_count,
            "fitness_failure_count": self.fitness_failure_count,
            "secondary_eval_count": self.secondary_eval_count,
            "prune_count": self.prune_count,
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "StatsTracker":
        return cls(**data)

    def to_record(self, step: int, archive: CVTArchive, stage: int = 0) -> dict[str, Any]:
        elites = archive.all_elites()
        primary_values = np.asarray([elite.primary_fitness for elite in elites if elite.primary_fitness is not None], dtype=float)
        primary_validation_values = np.asarray(
            [elite.primary_validation_fitness for elite in elites if elite.primary_validation_fitness is not None],
            dtype=float,
        )
        secondary_values = np.asarray(
            [elite.secondary_fitness for elite in elites if elite.secondary_fitness is not None],
            dtype=float,
        )
        secondary_validation_values = np.asarray(
            [elite.secondary_validation_fitness for elite in elites if elite.secondary_validation_fitness is not None],
            dtype=float,
        )
        return {
            "step": step,
            "stage": stage,
            "generated_count": self.generated_count,
            "accepted_count": self.accepted_count,
            "archive_occupancy": len(archive.occupied_cells()),
            "archive_elite_count": len(elites),
            "best_primary_fitness": float(np.max(primary_values)) if primary_values.size else None,
            "best_primary_validation_fitness": float(np.max(primary_validation_values)) if primary_validation_values.size else None,
            "best_secondary_fitness": float(np.max(secondary_values)) if secondary_values.size else None,
            "best_secondary_validation_fitness": float(np.max(secondary_validation_values)) if secondary_validation_values.size else None,
            "mean_primary_fitness": float(np.mean(primary_values)) if primary_values.size else None,
            "mean_primary_validation_fitness": float(np.mean(primary_validation_values)) if primary_validation_values.size else None,
            "mean_secondary_fitness": float(np.mean(secondary_values)) if secondary_values.size else None,
            "mean_secondary_validation_fitness": float(np.mean(secondary_validation_values)) if secondary_validation_values.size else None,
            "llm_error_count": self.llm_error_count,
            "parse_failure_count": self.parse_failure_count,
            "validation_failure_count": self.validation_failure_count,
            "descriptor_failure_count": self.descriptor_failure_count,
            "fitness_failure_count": self.fitness_failure_count,
            "secondary_eval_count": self.secondary_eval_count,
            "prune_count": self.prune_count,
        }
