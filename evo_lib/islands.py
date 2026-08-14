from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random
from types import ModuleType
from typing import Any, Callable
from uuid import uuid4

from evo_lib.candidate import Candidate
from evo_lib.config import GlobalJobConfig, IslandConfig
from evo_lib.engine import CandidateCommitResult, EvolutionEngine, ExecutedStep
from evo_lib.errors import ConfigurationError
from evo_lib.llm import LLMClient, build_llm_client


def _load_module(module_path: str, *, relative_to: str | None) -> ModuleType:
    path = Path(module_path)
    if not path.is_absolute():
        base_dir = Path(relative_to).resolve().parent if relative_to is not None else Path.cwd()
        path = (base_dir / path).resolve()
    if not path.exists():
        raise ConfigurationError(f"Island hooks module does not exist: {path}")
    module_name = f"evo_island_hooks_{path.stem}_{uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigurationError(f"Failed to load hooks module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class LoadedIslandHooks:
    initial_candidates_factory: Callable[[IslandConfig], list[Candidate]]
    descriptor_fn: Callable[[Candidate], list[float]]
    primary_fitness_factory: Callable[[IslandConfig], Callable[[Candidate], float]]
    primary_validation_fitness_factory: Callable[[IslandConfig], Callable[[Candidate], float] | None] | None
    secondary_fitness_factory: Callable[[IslandConfig], Callable[[Candidate], float] | None] | None
    secondary_validation_fitness_factory: Callable[[IslandConfig], Callable[[Candidate], float] | None] | None
    candidate_validator_fn: Callable[[Candidate], bool] | None
    candidate_parser_fn: Callable[[str, Any], list[Candidate]] | None
    llm_client_factory: Callable[[IslandConfig], LLMClient] | None


def load_island_hooks(config: IslandConfig) -> LoadedIslandHooks:
    module = _load_module(config.hooks.module_path, relative_to=config.source_path)

    def _resolve(name: str, *, required: bool) -> Any:
        if not hasattr(module, name):
            if required:
                raise ConfigurationError(
                    f"Hooks module {config.hooks.module_path} is missing required attribute {name!r}"
                )
            return None
        return getattr(module, name)

    return LoadedIslandHooks(
        initial_candidates_factory=_resolve(config.hooks.initial_candidates_factory, required=True),
        descriptor_fn=_resolve(config.hooks.descriptor_fn, required=True),
        primary_fitness_factory=_resolve(config.hooks.primary_fitness_factory, required=True),
        primary_validation_fitness_factory=_resolve(config.hooks.primary_validation_fitness_factory, required=False)
        if config.hooks.primary_validation_fitness_factory
        else None,
        secondary_fitness_factory=_resolve(config.hooks.secondary_fitness_factory, required=False)
        if config.hooks.secondary_fitness_factory
        else None,
        secondary_validation_fitness_factory=_resolve(config.hooks.secondary_validation_fitness_factory, required=False)
        if config.hooks.secondary_validation_fitness_factory
        else None,
        candidate_validator_fn=_resolve(config.hooks.candidate_validator_fn, required=False)
        if config.hooks.candidate_validator_fn
        else None,
        candidate_parser_fn=_resolve(config.hooks.candidate_parser_fn, required=False)
        if config.hooks.candidate_parser_fn
        else None,
        llm_client_factory=_resolve(config.hooks.llm_client_factory, required=False)
        if config.hooks.llm_client_factory
        else None,
    )


@dataclass
class IslandRuntime:
    config: IslandConfig
    hooks: LoadedIslandHooks
    engine: EvolutionEngine


@dataclass
class IslandJobResult:
    job_id: str
    job_dir: Path
    island_run_dirs: dict[str, Path]


class IslandJobRunner:
    def __init__(self, global_config: GlobalJobConfig, island_configs: list[IslandConfig]):
        if not island_configs:
            raise ConfigurationError("IslandJobRunner requires at least one island config")
        self.global_config = global_config
        self.island_configs = island_configs
        self._rng = random.Random(global_config.random_seed)
        self.job_id = self._generate_job_id(global_config.job_name)
        self.job_dir = Path(global_config.output_dir) / self.job_id
        self.islands_dir = self.job_dir / "islands"
        self._submission_counters = {config.island_id: 0 for config in island_configs}
        self.runtimes = self._build_runtimes()

    @classmethod
    def from_paths(cls, global_config_path: str, island_config_paths: list[str]) -> "IslandJobRunner":
        global_config = GlobalJobConfig.from_json_file(global_config_path)
        island_configs = [IslandConfig.from_json_file(path) for path in island_config_paths]
        return cls(global_config=global_config, island_configs=island_configs)

    def run(self) -> IslandJobResult:
        self._initialize_job_dir()
        for runtime in self.runtimes:
            seeds = runtime.hooks.initial_candidates_factory(runtime.config)
            runtime.engine.initialize(seeds)

        total_workers = self.global_config.parallel_workers
        active_islands = [runtime for runtime in self.runtimes if runtime.engine.step_count < runtime.engine.config.max_steps]
        in_flight: dict[Future[ExecutedStep], tuple[str, int]] = {}
        inflight_counts = {runtime.config.island_id: 0 for runtime in self.runtimes}
        per_island_limit = self._per_island_limits(total_workers)
        round_robin_index = 0

        with ThreadPoolExecutor(max_workers=total_workers) as executor:
            while active_islands or in_flight:
                while len(in_flight) < total_workers:
                    runtime = self._next_schedulable_runtime(
                        active_islands=active_islands,
                        inflight_counts=inflight_counts,
                        per_island_limit=per_island_limit,
                        round_robin_index=round_robin_index,
                    )
                    if runtime is None:
                        break
                    round_robin_index = (active_islands.index(runtime) + 1) if active_islands else 0
                    prepared = runtime.engine.prepare_step()
                    submission_index = self._next_submission_index(runtime.config.island_id)
                    runtime.engine.run_store.log_event(
                        "worker_task_submitted",
                        {
                            "submission_index": submission_index,
                            "submitted_at": datetime.now(timezone.utc).isoformat(),
                            "in_random_init": prepared.in_random_init,
                            "parent_ids": [parent.id for parent in prepared.parents],
                            "inflight_for_island": inflight_counts[runtime.config.island_id] + 1,
                            "total_parallel_workers": total_workers,
                        },
                    )
                    future = executor.submit(runtime.engine.execute_prepared_step, prepared)
                    in_flight[future] = (runtime.config.island_id, submission_index)
                    inflight_counts[runtime.config.island_id] += 1
                if not in_flight:
                    break
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                ordered_done = [future for future in list(in_flight.keys()) if future in done]
                for future in ordered_done:
                    island_id, submission_index = in_flight.pop(future)
                    inflight_counts[island_id] -= 1
                    runtime = self._runtime_by_id(island_id)
                    runtime.engine.run_store.log_event(
                        "worker_task_completed",
                        {
                            "submission_index": submission_index,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "remaining_inflight_for_island": inflight_counts[island_id],
                        },
                    )
                    committed = runtime.engine.commit_executed_step(future.result())
                    self._maybe_migrate(runtime, committed.inserted_candidates)
                    active_islands = [
                        candidate_runtime
                        for candidate_runtime in self.runtimes
                        if candidate_runtime.engine.step_count < candidate_runtime.engine.config.max_steps
                    ]

        for runtime in self.runtimes:
            runtime.engine.run_store.update_metadata(current_step=runtime.engine.step_count, status="completed")

        self._write_job_metadata()
        return IslandJobResult(
            job_id=self.job_id,
            job_dir=self.job_dir,
            island_run_dirs={runtime.config.island_id: runtime.engine.run_store.run_dir for runtime in self.runtimes},
        )

    def _build_runtimes(self) -> list[IslandRuntime]:
        island_ids = set()
        runtimes: list[IslandRuntime] = []
        islands_root = self.islands_dir
        for island_config in self.island_configs:
            if island_config.island_id in island_ids:
                raise ConfigurationError(f"Duplicate island id: {island_config.island_id}")
            island_ids.add(island_config.island_id)
            hooks = load_island_hooks(island_config)
            engine_config = replace(
                island_config.evolution,
                job_id=self.job_id,
                island_id=island_config.island_id,
                output_dir=str(islands_root),
                parallel_workers=1,
                llm=self._resolve_llm_config(island_config),
            )
            llm_client = (
                hooks.llm_client_factory(island_config)
                if hooks.llm_client_factory is not None
                else build_llm_client(engine_config, random_seed=engine_config.random_seed)
            )
            engine = EvolutionEngine(
                config=engine_config,
                llm_client=llm_client,
                primary_fitness=hooks.primary_fitness_factory(island_config),
                primary_validation_fitness=hooks.primary_validation_fitness_factory(island_config)
                if hooks.primary_validation_fitness_factory is not None
                else None,
                secondary_fitness=hooks.secondary_fitness_factory(island_config)
                if hooks.secondary_fitness_factory is not None
                else None,
                secondary_validation_fitness=hooks.secondary_validation_fitness_factory(island_config)
                if hooks.secondary_validation_fitness_factory is not None
                else None,
                descriptor_fn=hooks.descriptor_fn,
                candidate_validator=hooks.candidate_validator_fn,
                cross_island_elite_provider=self._foreign_elite_candidates,
            )
            runtimes.append(IslandRuntime(config=island_config, hooks=hooks, engine=engine))
        return runtimes

    def _foreign_elite_candidates(self, source_island_id: str | None) -> list[Candidate]:
        candidates: list[Candidate] = []
        for runtime in self.runtimes:
            if source_island_id is not None and runtime.config.island_id == source_island_id:
                continue
            if runtime.engine.archive is None:
                continue
            candidates.extend(runtime.engine.archive.all_elites())
        return candidates

    def _resolve_llm_config(self, island_config: IslandConfig):
        llm_config = island_config.evolution.llm
        if llm_config.llm_pool_path is None:
            return llm_config
        pool_path = Path(llm_config.llm_pool_path)
        if not pool_path.is_absolute():
            base_dir = Path(island_config.source_path).resolve().parent if island_config.source_path is not None else Path.cwd()
            pool_path = (base_dir / pool_path).resolve()
        return replace(llm_config, llm_pool_path=str(pool_path))

    def _initialize_job_dir(self) -> None:
        self.islands_dir.mkdir(parents=True, exist_ok=False)
        configs_dir = self.job_dir / "configs"
        configs_dir.mkdir(parents=False, exist_ok=False)
        with (self.job_dir / "global_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.global_config.to_dict(), handle, indent=2, sort_keys=True)
        with (self.job_dir / "islands.json").open("w", encoding="utf-8") as handle:
            json.dump(
                [
                    {
                        "island_id": runtime.config.island_id,
                        "config_path": runtime.config.source_path,
                        "run_name": runtime.engine.config.run_name,
                    }
                    for runtime in self.runtimes
                ],
                handle,
                indent=2,
                sort_keys=True,
            )
        for runtime in self.runtimes:
            with (configs_dir / f"{runtime.config.island_id}.json").open("w", encoding="utf-8") as handle:
                json.dump(runtime.config.to_dict(), handle, indent=2, sort_keys=True)

    def _write_job_metadata(self) -> None:
        payload = {
            "job_id": self.job_id,
            "job_name": self.global_config.job_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parallel_workers": self.global_config.parallel_workers,
            "migration_probability": self.global_config.migration_probability,
            "islands": [
                {
                    "island_id": runtime.config.island_id,
                    "run_dir": str(runtime.engine.run_store.run_dir),
                    "step_count": runtime.engine.step_count,
                    "occupied_cells": len(runtime.engine.archive.occupied_cells()) if runtime.engine.archive is not None else 0,
                }
                for runtime in self.runtimes
            ],
        }
        with (self.job_dir / "job_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _maybe_migrate(self, source_runtime: IslandRuntime, inserted_candidates: list[Candidate]) -> None:
        if self.global_config.migration_probability <= 0 or len(self.runtimes) <= 1:
            return
        for candidate in inserted_candidates:
            if self._rng.random() > self.global_config.migration_probability:
                continue
            destinations = [runtime for runtime in self.runtimes if runtime.config.island_id != source_runtime.config.island_id]
            target_runtime = self._rng.choice(destinations)
            result = target_runtime.engine.integrate_migrant(
                candidate,
                source_island_id=source_runtime.config.island_id,
                source_candidate_id=candidate.id,
            )
            source_runtime.engine.run_store.log_event(
                "candidate_migrated_out",
                {
                    "step": source_runtime.engine.step_count,
                    "candidate_id": candidate.id,
                    "target_island_id": target_runtime.config.island_id,
                    "target_candidate_id": result.candidate_id,
                    "inserted": result.inserted,
                },
            )

    def _runtime_by_id(self, island_id: str) -> IslandRuntime:
        for runtime in self.runtimes:
            if runtime.config.island_id == island_id:
                return runtime
        raise ConfigurationError(f"Unknown island id: {island_id}")

    def _next_schedulable_runtime(
        self,
        *,
        active_islands: list[IslandRuntime],
        inflight_counts: dict[str, int],
        per_island_limit: dict[str, int],
        round_robin_index: int,
    ) -> IslandRuntime | None:
        if not active_islands:
            return None
        for offset in range(len(active_islands)):
            runtime = active_islands[(round_robin_index + offset) % len(active_islands)]
            island_id = runtime.config.island_id
            if inflight_counts[island_id] < per_island_limit[island_id]:
                return runtime
        return None

    def _per_island_limits(self, total_workers: int) -> dict[str, int]:
        island_ids = [runtime.config.island_id for runtime in self.runtimes]
        if total_workers >= len(island_ids):
            base = total_workers // len(island_ids)
            extras = total_workers % len(island_ids)
            return {
                island_id: base + (1 if index < extras else 0)
                for index, island_id in enumerate(island_ids)
            }
        return {
            island_id: 1
            for island_id in island_ids
        }

    def _next_submission_index(self, island_id: str) -> int:
        self._submission_counters[island_id] += 1
        return self._submission_counters[island_id]

    @staticmethod
    def _generate_job_id(job_name: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        return f"{job_name}_{timestamp}_{uuid4().hex[:6]}"


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an island-based CVT-MAP-Elites job from JSON configs.")
    parser.add_argument("--global-config", required=True, help="Path to the global job config JSON file.")
    parser.add_argument(
        "--island-config",
        action="append",
        required=True,
        dest="island_configs",
        help="Path to an island config JSON file. Pass once per island.",
    )
    return parser


def main() -> None:
    args = _build_cli().parse_args()
    runner = IslandJobRunner.from_paths(args.global_config, args.island_configs)
    result = runner.run()
    print(f"Job directory: {result.job_dir}")
    for island_id, run_dir in sorted(result.island_run_dirs.items()):
        print(f"{island_id}: {run_dir}")


if __name__ == "__main__":
    main()
