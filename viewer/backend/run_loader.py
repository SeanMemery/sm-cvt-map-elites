from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np


ACTIVE_RUN_GRACE_SECONDS = 45


def _safe_json_load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _safe_jsonl_load(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                # Active runs can be mid-write; ignore trailing partial lines.
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _numeric_metadata_keys(candidates: list[dict[str, Any]]) -> list[str]:
    keys: set[str] = set()
    for candidate in candidates:
        metadata = candidate.get("metadata") or {}
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                keys.add(key)
    return sorted(keys)


def _weights_include_curiosity(weights: Any) -> bool:
    if not isinstance(weights, list):
        return False
    for entry in weights:
        if isinstance(entry, dict) and entry.get("strategy") == "curiosity_weighted":
            return True
    return False


def _schedule_includes_curiosity(schedule: Any) -> bool:
    if not isinstance(schedule, list):
        return False
    for phase in schedule:
        if isinstance(phase, dict) and _weights_include_curiosity(phase.get("weights")):
            return True
    return False


def _config_uses_curiosity(config: dict[str, Any]) -> bool:
    curiosity_strategies = {"curiosity_weighted", "emitter_curiosity_weighted"}
    return (
        config.get("grid_selection_strategy") in curiosity_strategies
        or config.get("elite_selection_strategy") in curiosity_strategies
        or config.get("generation_method") in {"llm_emitters_emitter_curiosity"}
        or _weights_include_curiosity(config.get("grid_sampling_weights"))
        or _weights_include_curiosity(config.get("elite_sampling_weights"))
        or _schedule_includes_curiosity(config.get("grid_sampling_schedule"))
        or _schedule_includes_curiosity(config.get("elite_sampling_schedule"))
    )


@dataclass
class CandidateIndex:
    latest_by_id: dict[str, dict[str, Any]]
    history_by_id: dict[str, list[dict[str, Any]]]
    children_by_parent_id: dict[str, list[str]]
    llm_call_to_candidate_ids: dict[str, list[str]]


@dataclass
class ImportedRun:
    run_id: str
    path: Path
    metadata: dict[str, Any]
    config: dict[str, Any]
    candidates: CandidateIndex
    events: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    llm_call_by_id: dict[str, dict[str, Any]]
    stats: list[dict[str, Any]]
    snapshot_steps: list[int]
    centroids: np.ndarray | None
    descriptor_labels: list[str]
    primary_metric_label: str
    primary_validation_metric_label: str
    secondary_metric_label: str
    secondary_validation_metric_label: str
    numeric_metadata_fields: list[str]
    island_id: str | None = None
    job_id: str | None = None
    latest_snapshot_cache: dict[int, dict[str, Any]] = field(default_factory=dict)

    @property
    def candidate_count(self) -> int:
        return len(self.candidates.latest_by_id)

    def latest_snapshot(self) -> dict[str, Any] | None:
        if not self.snapshot_steps:
            return None
        return self.load_snapshot(self.snapshot_steps[-1])

    def load_snapshot(self, step: int) -> dict[str, Any]:
        if step in self.latest_snapshot_cache:
            return self.latest_snapshot_cache[step]
        snapshot_path = self.path / "archive_snapshots" / f"step_{step:06d}.json"
        snapshot = _safe_json_load(snapshot_path)
        if self.centroids is not None:
            snapshot["centroids"] = self.centroids.tolist()
        snapshot.setdefault("descriptor_labels", self.descriptor_labels)
        self.latest_snapshot_cache[step] = snapshot
        return snapshot

    def summary(self) -> dict[str, Any]:
        latest_snapshot = self.latest_snapshot()
        active_candidates = [candidate for candidate in self.candidates.latest_by_id.values() if candidate.get("is_active", True)]
        primary_values = [
            float(candidate["primary_fitness"])
            for candidate in self.candidates.latest_by_id.values()
            if candidate.get("primary_fitness") is not None
        ]
        secondary_values = [
            float(candidate["secondary_fitness"])
            for candidate in self.candidates.latest_by_id.values()
            if candidate.get("secondary_fitness") is not None
        ]
        primary_validation_values = [
            float(candidate["primary_validation_fitness"])
            for candidate in self.candidates.latest_by_id.values()
            if candidate.get("primary_validation_fitness") is not None
        ]
        secondary_validation_values = [
            float(candidate["secondary_validation_fitness"])
            for candidate in self.candidates.latest_by_id.values()
            if candidate.get("secondary_validation_fitness") is not None
        ]
        latest_stats = self.stats[-1] if self.stats else {}
        last_updated_at = self.last_updated_at()
        status = self.derived_status(last_updated_at)
        return {
            "run_id": self.run_id,
            "run_name": self.metadata.get("run_name", self.run_id),
            "project_id": self.config.get("project_id") or self.metadata.get("project_id") or "default",
            "job_id": self.config.get("job_id") or self.metadata.get("job_id"),
            "path": str(self.path),
            "created_at": self.metadata.get("created_at"),
            "status": status,
            "raw_status": self.metadata.get("status"),
            "current_step": self.metadata.get("current_step", 0),
            "candidate_count": self.candidate_count,
            "archive_occupancy": len(latest_snapshot.get("occupied_cells", [])) if latest_snapshot else 0,
            "active_candidate_count": len(active_candidates),
            "best_primary_fitness": max(primary_values) if primary_values else None,
            "best_primary_validation_fitness": max(primary_validation_values) if primary_validation_values else None,
            "best_secondary_fitness": max(secondary_values) if secondary_values else None,
            "best_secondary_validation_fitness": max(secondary_validation_values) if secondary_validation_values else None,
            "latest_stats": latest_stats,
            "last_updated_at": last_updated_at,
            "descriptor_labels": self.descriptor_labels,
            "primary_metric_label": self.primary_metric_label,
            "primary_validation_metric_label": self.primary_validation_metric_label,
            "secondary_metric_label": self.secondary_metric_label if secondary_values else None,
            "secondary_validation_metric_label": self.secondary_validation_metric_label if secondary_validation_values else None,
            "curiosity_enabled": _config_uses_curiosity(self.config),
        }

    def last_updated_at(self) -> str | None:
        candidates_path = self.path / "candidates.jsonl"
        events_path = self.path / "events.jsonl"
        stats_path = self.path / "stats.jsonl"
        llm_calls_path = self.path / "llm_calls.jsonl"
        mtimes = [
            path.stat().st_mtime
            for path in (candidates_path, events_path, stats_path, llm_calls_path)
            if path.exists()
        ]
        if not mtimes:
            return None
        return datetime.fromtimestamp(max(mtimes), tz=timezone.utc).isoformat()

    def derived_status(self, last_updated_at: str | None) -> str:
        # Paused marker written by the pause endpoint
        if (self.path / "paused").exists():
            return "paused"
        raw_status = str(self.metadata.get("status") or "unknown").lower()
        max_steps = int(self.config.get("max_steps") or 0)
        current_step = int(self.metadata.get("current_step") or 0)
        if max_steps > 0 and current_step >= max_steps:
            return "completed"
        if raw_status in {"completed", "aborted", "failed"}:
            return raw_status
        if raw_status == "running":
            if last_updated_at is None:
                return "stalled"
            try:
                last_updated_epoch = datetime.fromisoformat(last_updated_at).timestamp()
            except ValueError:
                return "stalled"
            return "running" if (time.time() - last_updated_epoch) <= ACTIVE_RUN_GRACE_SECONDS else "stalled"
        return raw_status


class RunLoader:
    def __init__(
        self,
        registry_path: Path,
        discovery_roots: list[Path] | None = None,
        run_name_prefixes: tuple[str, ...] = (),
    ):
        self.registry_path = registry_path
        self.discovery_roots = discovery_roots or []
        self.run_name_prefixes = run_name_prefixes
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_path.exists():
            self.registry_path.write_text("[]\n", encoding="utf-8")
        self.runs: dict[str, ImportedRun] = {}
        self._load_cache: dict[str, tuple[tuple[int, ...], ImportedRun]] = {}
        self._load_registry()

    def import_path(self, path: str) -> list[dict[str, Any]]:
        target = Path(path).expanduser().resolve()
        # Detect job dir so we can inject island metadata
        job_metadata: dict[str, Any] = {}
        is_job = self._is_job_dir(target)
        if is_job:
            try:
                job_metadata = _safe_json_load(target / "job_metadata.json")
            except Exception:
                job_metadata = {}
        run_paths = self._discover_run_paths(target)
        summaries: list[dict[str, Any]] = []
        for run_path in run_paths:
            imported = self._load_run_cached(run_path)
            if is_job:
                job_id = str(job_metadata.get("job_id", target.name)) if job_metadata else target.name
                # island_id: prefer from run's own metadata, then from job_metadata islands list, then use dir name
                island_id = (
                    imported.metadata.get("island_id")
                    or str(job_metadata.get("island_id", run_path.name))
                )
                imported.island_id = island_id
                imported.job_id = job_id
            self.runs[imported.run_id] = imported
            summaries.append(imported.summary())
        self._persist_registry()
        return summaries

    def list_runs(self) -> list[dict[str, Any]]:
        return self._summaries_with_running_disambiguation()

    def get_run(self, run_id: str) -> ImportedRun:
        return self.runs[run_id]

    def refresh_run(self, run_id: str) -> dict[str, Any]:
        run_path = self.runs[run_id].path
        imported = self._load_run_cached(run_path)
        self.runs[run_id] = imported
        self._persist_registry()
        for summary in self._summaries_with_running_disambiguation():
            if summary.get("run_id") == run_id:
                return summary
        return imported.summary()

    def refresh_project_runs(self, project_id: str) -> list[dict[str, Any]]:
        source_paths = self.discovery_roots or {self._infer_refresh_root(imported.path) for imported in self.runs.values()}
        discovered: dict[str, ImportedRun] = {}
        for source_path in source_paths:
            for run_path in self._discover_run_paths(source_path):
                reloaded = self._load_run_cached(run_path)
                reloaded_project_id = reloaded.config.get("project_id") or reloaded.metadata.get("project_id") or "default"
                if reloaded_project_id != project_id:
                    continue
                discovered[reloaded.run_id] = reloaded
        self.runs.update(discovered)
        self._persist_registry()
        return [
            summary
            for summary in self._summaries_with_running_disambiguation()
            if (summary.get("project_id") or "default") == project_id
        ]

    def refresh_all_runs(self) -> list[dict[str, Any]]:
        refreshed: dict[str, ImportedRun] = {}
        source_paths = self.discovery_roots or {self._infer_refresh_root(imported.path) for imported in self.runs.values()}
        for source_path in source_paths:
            for run_path in self._discover_run_paths(source_path):
                reloaded = self._load_run_cached(run_path)
                refreshed[reloaded.run_id] = reloaded
        self.runs = refreshed
        self._persist_registry()
        return self._summaries_with_running_disambiguation()

    def remove_run(self, run_id: str) -> None:
        del self.runs[run_id]
        self._persist_registry()

    def _load_registry(self) -> None:
        records = _safe_json_load(self.registry_path)
        if not isinstance(records, list):
            return
        for item in records:
            if not isinstance(item, dict):
                continue
            path = item.get("path")
            if not path:
                continue
            run_path = Path(path)
            if not run_path.exists():
                continue
            imported = self._load_run_cached(run_path)
            if item.get("island_id") is not None:
                imported.island_id = item["island_id"]
            if item.get("job_id") is not None:
                imported.job_id = item["job_id"]
            self.runs[imported.run_id] = imported

    def _persist_registry(self) -> None:
        records = []
        for run_id, imported in sorted(self.runs.items()):
            record: dict[str, Any] = {"run_id": run_id, "path": str(imported.path)}
            if imported.island_id is not None:
                record["island_id"] = imported.island_id
            if imported.job_id is not None:
                record["job_id"] = imported.job_id
            records.append(record)
        self.registry_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def _summaries_with_running_disambiguation(self) -> list[dict[str, Any]]:
        summaries = [self.runs[run_id].summary() for run_id in sorted(self.runs.keys())]
        latest_running_by_name: dict[tuple[str, str], tuple[float, str]] = {}
        for summary in summaries:
            if summary.get("status") != "running":
                continue
            key = (str(summary.get("project_id") or ""), str(summary.get("run_name") or ""))
            last_updated_at = summary.get("last_updated_at")
            try:
                timestamp = datetime.fromisoformat(last_updated_at).timestamp() if last_updated_at else 0.0
            except ValueError:
                timestamp = 0.0
            current = latest_running_by_name.get(key)
            if current is None or timestamp > current[0]:
                latest_running_by_name[key] = (timestamp, str(summary.get("run_id")))
        for summary in summaries:
            if summary.get("status") != "running":
                continue
            key = (str(summary.get("project_id") or ""), str(summary.get("run_name") or ""))
            if str(summary.get("run_id")) != latest_running_by_name.get(key, (0.0, ""))[1]:
                summary["status"] = "stalled"
        return summaries

    def _discover_run_paths(self, path: Path) -> list[Path]:
        if self._is_run_dir(path):
            return [path] if self._matches_run_name(path) else []
        if self._is_job_dir(path):
            return self._discover_island_run_paths(path)
        if not path.exists() or not path.is_dir():
            return []  # path deleted — skip silently during refresh
        run_dirs = [
            child
            for child in sorted(path.rglob("*"))
            if child.is_dir() and self._is_run_dir(child) and self._matches_run_name(child)
        ]
        if not run_dirs:
            raise FileNotFoundError(f"No run directories found at {path}")
        return run_dirs

    def _discover_island_run_paths(self, job_path: Path) -> list[Path]:
        islands_dir = job_path / "islands"
        if not islands_dir.exists() or not islands_dir.is_dir():
            return []
        return [
            child
            for child in sorted(islands_dir.iterdir())
            if child.is_dir() and self._is_run_dir(child)
        ]

    def _matches_run_name(self, path: Path) -> bool:
        if not self.run_name_prefixes:
            return True
        return path.name.startswith(self.run_name_prefixes)

    @staticmethod
    def _infer_refresh_root(run_path: Path) -> Path:
        if run_path.parent.name == "runs":
            return run_path.parent
        return run_path.parent

    @staticmethod
    def _is_job_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "job_metadata.json").exists()
            and (path / "islands").is_dir()
        )

    @staticmethod
    def _is_run_dir(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "metadata.json").exists()
            and (path / "config.json").exists()
            and (path / "archive_snapshots").exists()
        )

    @staticmethod
    def _run_signature(path: Path) -> tuple[int, ...]:
        names = ("metadata.json", "config.json", "candidates.jsonl", "events.jsonl", "llm_calls.jsonl", "stats.jsonl")
        sig: list[int] = []
        for name in names:
            try:
                sig.append(int((path / name).stat().st_mtime_ns))
            except FileNotFoundError:
                sig.append(-1)
        try:
            sig.append(int((path / "archive_snapshots").stat().st_mtime_ns))
        except FileNotFoundError:
            sig.append(-1)
        return tuple(sig)

    def _load_run_cached(self, path: Path) -> ImportedRun:
        """Like _load_run, but skips reparsing JSONL files when nothing on disk has changed since the last load."""
        path_key = str(path)
        signature = self._run_signature(path)
        cached = self._load_cache.get(path_key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        imported = self._load_run(path)
        self._load_cache[path_key] = (signature, imported)
        return imported

    def _load_run(self, path: Path) -> ImportedRun:
        metadata = _safe_json_load(path / "metadata.json")
        config = _safe_json_load(path / "config.json")
        candidate_records = _safe_jsonl_load(path / "candidates.jsonl")
        latest_by_id: dict[str, dict[str, Any]] = {}
        history_by_id: dict[str, list[dict[str, Any]]] = {}
        children_by_parent_id: dict[str, list[str]] = {}
        llm_call_to_candidate_ids: dict[str, list[str]] = {}
        for record in candidate_records:
            candidate_id = record.get("id")
            if not candidate_id:
                continue
            history_by_id.setdefault(candidate_id, []).append(record)
        for candidate_id, records in history_by_id.items():
            ordered = sorted(records, key=lambda item: item.get("record_version", 0))
            latest_by_id[candidate_id] = ordered[-1]
            for record in ordered:
                for parent_id in record.get("parent_ids", []):
                    children_by_parent_id.setdefault(parent_id, [])
                    if candidate_id not in children_by_parent_id[parent_id]:
                        children_by_parent_id[parent_id].append(candidate_id)
                llm_call_id = record.get("llm_call_id")
                if llm_call_id:
                    llm_call_to_candidate_ids.setdefault(llm_call_id, [])
                    if candidate_id not in llm_call_to_candidate_ids[llm_call_id]:
                        llm_call_to_candidate_ids[llm_call_id].append(candidate_id)
        events = _safe_jsonl_load(path / "events.jsonl")
        llm_calls = _safe_jsonl_load(path / "llm_calls.jsonl")
        llm_call_by_id = {record["llm_call_id"]: record for record in llm_calls if record.get("llm_call_id")}
        stats = _safe_jsonl_load(path / "stats.jsonl")
        archive_dir = path / "archive_snapshots"
        snapshot_steps = sorted(
            int(file.stem.split("_")[-1])
            for file in archive_dir.glob("step_*.json")
        )
        centroids = None
        centroids_path = path / "centroids.npy"
        if centroids_path.exists():
            centroids = np.load(centroids_path)
        descriptor_labels = (
            config.get("descriptor_labels")
            or metadata.get("descriptor_labels")
            or [f"descriptor_{index}" for index in range(int(metadata.get("descriptor_dim", 0) or config.get("descriptor_dim", 0)))]
        )
        primary_metric_label = config.get("primary_metric_label") or metadata.get("primary_metric_label") or "Primary"
        primary_validation_metric_label = (
            config.get("primary_validation_metric_label")
            or metadata.get("primary_validation_metric_label")
            or f"{primary_metric_label} Validation"
        )
        secondary_metric_label = config.get("secondary_metric_label") or metadata.get("secondary_metric_label") or "Secondary"
        secondary_validation_metric_label = (
            config.get("secondary_validation_metric_label")
            or metadata.get("secondary_validation_metric_label")
            or f"{secondary_metric_label} Validation"
        )
        return ImportedRun(
            run_id=str(metadata.get("run_id", path.name)),
            path=path,
            metadata=metadata,
            config=config,
            island_id=metadata.get("island_id") or config.get("island_id"),
            job_id=metadata.get("job_id") or config.get("job_id"),
            candidates=CandidateIndex(
                latest_by_id=latest_by_id,
                history_by_id={candidate_id: sorted(records, key=lambda item: item.get("record_version", 0)) for candidate_id, records in history_by_id.items()},
                children_by_parent_id=children_by_parent_id,
                llm_call_to_candidate_ids=llm_call_to_candidate_ids,
            ),
            events=events,
            llm_calls=llm_calls,
            llm_call_by_id=llm_call_by_id,
            stats=stats,
            snapshot_steps=snapshot_steps,
            centroids=centroids,
            descriptor_labels=descriptor_labels,
            primary_metric_label=str(primary_metric_label),
            primary_validation_metric_label=str(primary_validation_metric_label),
            secondary_metric_label=str(secondary_metric_label),
            secondary_validation_metric_label=str(secondary_validation_metric_label),
            numeric_metadata_fields=_numeric_metadata_keys(list(latest_by_id.values())),
        )
