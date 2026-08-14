from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    id: str
    code: str
    metadata: dict[str, Any] = field(default_factory=dict)
    llm_call_id: str | None = None
    descriptor_raw: list[float] | None = None
    descriptor_norm: list[float] | None = None
    primary_fitness: float | None = None
    primary_validation_fitness: float | None = None
    secondary_fitness: float | None = None
    secondary_validation_fitness: float | None = None
    curiosity_score: float | None = None
    curiosity_updates: int = 0
    emitter_curiosity_scores: dict[str, float] = field(default_factory=dict)
    emitter_curiosity_updates: dict[str, int] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    created_at_step: int = 0
    stage: int = 0
    stats: dict[str, Any] | None = None
    cell_id: int | None = None
    is_active: bool = True
    record_version: int = 1
    # Per-cell sampling statistics (used by evo_lib.sampling)
    cell_n_trials: int = 0
    cell_reward_ema: float = 0.0
    cell_logit_exploit: float = 0.0
    cell_emitter_n_trials: dict = field(default_factory=dict)    # emitter -> int
    cell_emitter_reward_ema: dict = field(default_factory=dict)  # emitter -> float
    cell_emitter_successes: dict = field(default_factory=dict)   # emitter -> float (Beta alpha)
    cell_emitter_failures: dict = field(default_factory=dict)    # emitter -> float (Beta beta)
    cell_emitter_logits: dict = field(default_factory=dict)       # emitter -> float (softmax logit)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "code": self.code,
            "metadata": self.metadata,
            "llm_call_id": self.llm_call_id,
            "descriptor_raw": self.descriptor_raw,
            "descriptor_norm": self.descriptor_norm,
            "primary_fitness": self.primary_fitness,
            "primary_validation_fitness": self.primary_validation_fitness,
            "secondary_fitness": self.secondary_fitness,
            "secondary_validation_fitness": self.secondary_validation_fitness,
            "curiosity_score": self.curiosity_score,
            "curiosity_updates": self.curiosity_updates,
            "emitter_curiosity_scores": self.emitter_curiosity_scores,
            "emitter_curiosity_updates": self.emitter_curiosity_updates,
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "created_at_step": self.created_at_step,
            "stage": self.stage,
            "stats": self.stats,
            "cell_id": self.cell_id,
            "is_active": self.is_active,
            "record_version": self.record_version,
            "cell_n_trials": self.cell_n_trials,
            "cell_reward_ema": self.cell_reward_ema,
            "cell_logit_exploit": self.cell_logit_exploit,
            "cell_emitter_n_trials": dict(self.cell_emitter_n_trials),
            "cell_emitter_reward_ema": dict(self.cell_emitter_reward_ema),
            "cell_emitter_successes": dict(self.cell_emitter_successes),
            "cell_emitter_failures": dict(self.cell_emitter_failures),
            "cell_emitter_logits": dict(self.cell_emitter_logits),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        return cls(
            id=data["id"],
            code=data["code"],
            metadata=data.get("metadata", {}),
            llm_call_id=data.get("llm_call_id"),
            descriptor_raw=data.get("descriptor_raw"),
            descriptor_norm=data.get("descriptor_norm"),
            primary_fitness=data.get("primary_fitness"),
            primary_validation_fitness=data.get("primary_validation_fitness"),
            secondary_fitness=data.get("secondary_fitness"),
            secondary_validation_fitness=data.get("secondary_validation_fitness"),
            curiosity_score=data.get("curiosity_score"),
            curiosity_updates=data.get("curiosity_updates", 0),
            emitter_curiosity_scores=dict(data.get("emitter_curiosity_scores", {})),
            emitter_curiosity_updates={str(key): int(value) for key, value in dict(data.get("emitter_curiosity_updates", {})).items()},
            parent_ids=list(data.get("parent_ids", [])),
            generation=data.get("generation", 0),
            created_at_step=data.get("created_at_step", 0),
            stage=data.get("stage", 0),
            stats=data.get("stats"),
            cell_id=data.get("cell_id"),
            is_active=data.get("is_active", True),
            record_version=data.get("record_version", 1),
            cell_n_trials=int(data.get("cell_n_trials", 0)),
            cell_reward_ema=float(data.get("cell_reward_ema", 0.0)),
            cell_logit_exploit=float(data.get("cell_logit_exploit", 0.0)),
            cell_emitter_n_trials=dict(data.get("cell_emitter_n_trials", {})),
            cell_emitter_reward_ema=dict(data.get("cell_emitter_reward_ema", {})),
            cell_emitter_successes=dict(data.get("cell_emitter_successes", {})),
            cell_emitter_failures=dict(data.get("cell_emitter_failures", {})),
            cell_emitter_logits=dict(data.get("cell_emitter_logits", {})),
        )


@dataclass
class Elite:
    candidate_id: str
    cell_id: int
    primary_fitness: float
    primary_validation_fitness: float | None = None
    secondary_fitness: float | None = None
    secondary_validation_fitness: float | None = None
    curiosity_score: float | None = None
