from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.checkpoint import load_checkpoint as load_checkpoint_file
from evo_lib.checkpoint import save_checkpoint as save_checkpoint_file
from evo_lib.config import EvolutionConfig, StorageConfig
from evo_lib.errors import ConfigurationError
from evo_lib.errors import StorageError
from evo_lib.normalizer import DescriptorNormalizer


class RunStore:
    def __init__(
        self,
        config: StorageConfig,
        run_name: str,
        output_dir: str,
        run_id: str | None = None,
        existing_run_dir: str | None = None,
    ):
        self.config = config
        self.run_name = run_name
        self.output_dir = Path(output_dir)
        self.run_id = run_id or config.run_id or self._generate_run_id(run_name)
        self.run_dir = Path(existing_run_dir) if existing_run_dir else self.output_dir / self.run_id
        self.archive_snapshots_dir = self.run_dir / "archive_snapshots"
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.config_path = self.run_dir / "config.json"
        self.metadata_path = self.run_dir / "metadata.json"
        self.candidates_path = self.run_dir / "candidates.jsonl"
        self.events_path = self.run_dir / "events.jsonl"
        self.llm_calls_path = self.run_dir / "llm_calls.jsonl"
        self.stats_path = self.run_dir / "stats.jsonl"
        self.centroids_path = self.run_dir / "centroids.npy"

    def initialize_run(self, full_config: EvolutionConfig, centroids: np.ndarray) -> None:
        if not self.config.save_enabled:
            return
        try:
            if self.run_dir.exists():
                if self.config.overwrite:
                    shutil.rmtree(self.run_dir)
                else:
                    raise StorageError(f"Run directory already exists: {self.run_dir}")
            self.archive_snapshots_dir.mkdir(parents=True, exist_ok=False)
            self.checkpoints_dir.mkdir(parents=True, exist_ok=False)
            self._write_json(self.config_path, full_config.to_dict())
            self._write_json(
                self.metadata_path,
                {
                    "run_id": self.run_id,
                    "run_name": full_config.run_name,
                    "project_id": full_config.project_id,
                    "job_id": full_config.job_id,
                    "island_id": full_config.island_id,
                    "created_at": self._timestamp(),
                    "library_version": "0.1.0",
                    "status": "running",
                    "current_step": 0,
                    "current_stage": 0,
                    "total_stages": full_config.secondary_eval.stages,
                    "descriptor_dim": full_config.descriptor_dim,
                    "num_centroids": full_config.num_centroids,
                    "descriptor_labels": full_config.descriptor_labels,
                    "primary_metric_label": full_config.primary_metric_label,
                    "primary_validation_metric_label": full_config.primary_validation_metric_label,
                    "secondary_metric_label": full_config.secondary_metric_label,
                    "secondary_validation_metric_label": full_config.secondary_validation_metric_label,
                },
            )
            np.save(self.centroids_path, centroids)
        except OSError as exc:
            raise StorageError(f"Failed to initialize run storage: {exc}") from exc

    def attach_existing_run(self) -> None:
        if not self.config.save_enabled:
            return
        if not self.run_dir.exists():
            raise StorageError(f"Existing run directory does not exist: {self.run_dir}")

    def update_metadata(self, **updates: Any) -> None:
        if not self.config.save_enabled:
            return
        metadata = self._read_json(self.metadata_path)
        metadata.update(updates)
        self._write_json(self.metadata_path, metadata)

    def save_candidate(self, candidate: Candidate) -> None:
        if not self.config.save_enabled or not self.config.save_candidates:
            return
        self._append_jsonl(self.candidates_path, candidate.to_dict())

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.config.save_enabled or not self.config.save_events:
            return
        record = dict(payload)
        record["type"] = event_type
        self._append_jsonl(self.events_path, record)

    def log_llm_call(
        self,
        step: int,
        parent_ids: list[str],
        llm_call_id: str,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
        reasoning: str | None = None,
    ) -> None:
        if not self.config.save_enabled or not self.config.save_llm_calls:
            return
        record = {
            "llm_call_id": llm_call_id,
            "step": step,
            "parent_ids": parent_ids,
            "prompt": prompt,
            "response": response,
            "created_at": self._timestamp(),
        }
        if reasoning:
            record["reasoning"] = reasoning
        if metadata:
            record.update(metadata)
        self._append_jsonl(self.llm_calls_path, record)

    def save_archive_snapshot(
        self,
        step: int,
        archive: CVTArchive,
        normalizer: DescriptorNormalizer,
        stage: int = 0,
        sampling_global_state: "dict | None" = None,
    ) -> None:
        if not self.config.save_enabled:
            return
        occupied_cells = []
        cell_elites = []
        cell_candidates = []
        for cell_id in archive.occupied_cells():
            elites = archive.get_cell_elites(cell_id)
            if not elites:
                continue
            candidates = archive.get_cell_candidates(cell_id)
            candidate = elites[0]
            occupied_cells.append(
                {
                    "cell_id": cell_id,
                    "candidate_id": candidate.id,
                    "primary_fitness": candidate.primary_fitness,
                    "primary_validation_fitness": candidate.primary_validation_fitness,
                    "secondary_fitness": candidate.secondary_fitness,
                    "secondary_validation_fitness": candidate.secondary_validation_fitness,
                    "curiosity_score": candidate.curiosity_score,
                    "emitter_curiosity_scores": candidate.emitter_curiosity_scores,
                    "descriptor_raw": candidate.descriptor_raw,
                    "descriptor_norm": candidate.descriptor_norm,
                    "cell_n_trials": candidate.cell_n_trials,
                    "cell_reward_ema": candidate.cell_reward_ema,
                    "cell_logit_exploit": candidate.cell_logit_exploit,
                    "cell_emitter_n_trials": candidate.cell_emitter_n_trials,
                    "cell_emitter_reward_ema": candidate.cell_emitter_reward_ema,
                    "cell_emitter_logits": candidate.cell_emitter_logits,
                }
            )
            cell_elites.append(
                {
                    "cell_id": cell_id,
                    "elites": [
                        {
                            "candidate_id": elite.id,
                            "primary_fitness": elite.primary_fitness,
                            "primary_validation_fitness": elite.primary_validation_fitness,
                            "secondary_fitness": elite.secondary_fitness,
                            "secondary_validation_fitness": elite.secondary_validation_fitness,
                            "curiosity_score": elite.curiosity_score,
                            "emitter_curiosity_scores": elite.emitter_curiosity_scores,
                            "descriptor_raw": elite.descriptor_raw,
                            "descriptor_norm": elite.descriptor_norm,
                        }
                        for elite in elites
                    ],
                }
            )
            cell_candidates.append(
                {
                    "cell_id": cell_id,
                    "candidates": [
                        {
                            "candidate_id": entry.id,
                            "primary_fitness": entry.primary_fitness,
                            "primary_validation_fitness": entry.primary_validation_fitness,
                            "secondary_fitness": entry.secondary_fitness,
                            "secondary_validation_fitness": entry.secondary_validation_fitness,
                            "curiosity_score": entry.curiosity_score,
                            "emitter_curiosity_scores": entry.emitter_curiosity_scores,
                            "descriptor_raw": entry.descriptor_raw,
                            "descriptor_norm": entry.descriptor_norm,
                            "is_elite": entry.id in {elite.id for elite in elites},
                        }
                        for entry in candidates
                    ],
                }
            )
        try:
            normalizer_bounds = normalizer.bounds()
        except ConfigurationError:
            normalizer_bounds = None
        snapshot = {
            "step": step,
            "stage": stage,
            "normalizer_bounds": normalizer_bounds,
            "descriptor_labels": getattr(normalizer, "descriptor_labels", None),
            "occupied_cells": occupied_cells,
            "cell_elites": cell_elites,
            "cell_candidates": cell_candidates,
            "insertion_rate_ema": float((sampling_global_state or {}).get("insertion_rate_ema", 0.5)),
        }
        filename = self.archive_snapshots_dir / f"step_{step:06d}.json"
        self._write_json(filename, snapshot)

    def save_checkpoint(self, engine_state: dict[str, Any], step: int) -> None:
        if not self.config.save_enabled:
            return
        latest_path = self.checkpoints_dir / "latest.pkl"
        step_path = self.checkpoints_dir / f"step_{step:06d}.pkl"
        save_checkpoint_file(latest_path, engine_state)
        save_checkpoint_file(step_path, engine_state)

    def load_checkpoint(self, path: str) -> dict[str, Any]:
        return load_checkpoint_file(Path(path))

    def save_stats(self, record: dict[str, Any]) -> None:
        if not self.config.save_enabled:
            return
        self._append_jsonl(self.stats_path, record)

    @staticmethod
    def _generate_run_id(run_name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        return f"{run_name}_{timestamp}_{uuid4().hex[:6]}"

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")
        except OSError as exc:
            raise StorageError(f"Failed to append JSONL to {path}: {exc}") from exc

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            tmp_path.replace(path)
        except OSError as exc:
            raise StorageError(f"Failed to write JSON file {path}: {exc}") from exc

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except OSError as exc:
            raise StorageError(f"Failed to read JSON file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise StorageError(f"JSON file did not contain an object: {path}")
        return data
