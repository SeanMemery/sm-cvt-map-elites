from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import random
from collections import deque
from dataclasses import replace
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.config import EmitterConfig, EvolutionConfig, ResumeMode, ThinkingMode
from evo_lib.errors import DescriptorError, FitnessError, LLMError, ParsingError, StatsValidationError, StorageError, ValidationError, validate_stats
from evo_lib.error_feedback import ErrorFeedbackCollector
from evo_lib.checkpoint import load_checkpoint as load_checkpoint_file
from evo_lib.llm import LLMClient, build_prompt, random_init_instructions
from evo_lib.normalizer import DescriptorNormalizer
from evo_lib.parsing import parse_edit_candidates, parse_full_candidates
from evo_lib.secondary import run_stage_end_evaluation
from evo_lib.sampling import (
    compute_reward,
    select_cell,
    select_emitter,
    update_cell_and_global_stats,
)
from evo_lib.selection import sample_parents
from evo_lib.stats import StatsTracker
from evo_lib.storage import RunStore


def _unwrap_fitness(result: object) -> tuple[float, dict | None]:
    """Unwrap a fitness function return value.

    Accepts:
        float                    → (float, None)
        (float, dict)            → (float, validated_dict)
        (float, dict) where dict is malformed → raises StatsValidationError
    """
    if isinstance(result, tuple):
        if len(result) != 2:
            raise StatsValidationError(
                f"Fitness function returned a tuple of length {len(result)}; "
                "expected (score: float, stats: dict)."
            )
        score, stats = result
        return float(score), validate_stats(stats)
    return float(result), None


@dataclass
class PreparedStep:
    in_random_init: bool
    parents: list[Candidate]
    ancestors: list[Candidate]
    inspirations: list[Candidate]
    prompt: str
    emitter_name: str | None = None
    thinking_mode: ThinkingMode = "default"
    is_local: bool = False  # True when a local emitter handles this step instead of the LLM


@dataclass
class CandidateEvaluation:
    candidate: Candidate
    status: str
    error: str | None = None


@dataclass
class CandidateCommitResult:
    candidate_id: str
    status: str
    inserted: bool
    cell_id: int | None = None
    replaced_candidate_id: str | None = None


@dataclass
class ExecutedStep:
    prepared: PreparedStep
    llm_result: Any | None = None
    children: list[CandidateEvaluation] | None = None
    error: Exception | None = None


@dataclass
class CommittedStep:
    step: int
    llm_call_id: str | None
    child_ids: list[str]
    inserted_candidates: list[Candidate]


class EvolutionEngine:
    def __init__(
        self,
        config: EvolutionConfig,
        llm_client: LLMClient,
        primary_fitness: Callable[[Candidate], float],
        descriptor_fn: Callable[[Candidate], list[float]],
        primary_validation_fitness: Callable[[Candidate], float] | None = None,
        secondary_fitness: Callable[[Candidate], float] | None = None,
        secondary_validation_fitness: Callable[[Candidate], float] | None = None,
        candidate_validator: Callable[[Candidate], bool] | None = None,
        batch_primary_fitness: Callable[[list[Candidate]], list] | None = None,
        archive: CVTArchive | None = None,
        normalizer: DescriptorNormalizer | None = None,
        run_store: RunStore | None = None,
        cross_island_elite_provider: Callable[[str | None], list[Candidate]] | None = None,
        local_emitters: dict[str, Callable[[Candidate], list[Candidate]]] | None = None,
    ):
        self.config = config
        self.llm_client = llm_client
        self.local_emitters: dict[str, Callable[[Candidate], list[Candidate]]] = local_emitters or {}
        self.error_feedback = ErrorFeedbackCollector(
            top_n=self.config.error_feedback_count,
            enabled=self.config.error_feedback_enabled,
        )
        self.primary_fitness_fn = primary_fitness
        self.primary_validation_fitness_fn = primary_validation_fitness
        self.batch_primary_fitness_fn = batch_primary_fitness
        self.descriptor_fn = descriptor_fn
        self.secondary_fitness_fn = secondary_fitness
        self.secondary_validation_fitness_fn = secondary_validation_fitness
        # fitness_batch_size > 1 with no batch_primary_fitness_fn is allowed —
        # external batching (e.g. CrossIslandBatcher) can handle batching via the
        # per-candidate primary_fitness_fn while setting this field for metadata only.
        self.candidate_validator = candidate_validator
        self.candidate_registry: dict[str, Candidate] = {}
        self._record_versions: dict[str, int] = {}
        self._save_candidate_keys: dict[str, tuple] = {}  # tracks last saved state to suppress no-op writes
        self._recent_elite_ids: deque[str] = deque(maxlen=1000)
        self._rng = np.random.default_rng(self.config.random_seed)
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        self.archive = archive
        self.normalizer = normalizer or DescriptorNormalizer(self.config.normalizer)
        self.normalizer.descriptor_labels = list(self.config.descriptor_labels or [])
        self.run_store = run_store or RunStore(
            config=self.config.storage,
            run_name=self.config.run_name,
            output_dir=self.config.output_dir,
        )
        self.cross_island_elite_provider = cross_island_elite_provider
        self.stats = StatsTracker()
        self.step_count = 0
        self.stage = 0
        self.generation = 0
        self._next_candidate_index = 0
        self._consecutive_failures = 0
        self._best_primary_seen: float | None = None
        self._best_primary_improvement_step = 0
        self._initialized = False
        # Sampling bandit state (used when config.sampling is set)
        self._sampling_total_trials: int = 0
        self._sampling_global_logit_exploit: float = 0.0
        self._sampling_pending_elite_id: str | None = None
        self._sampling_global_state: dict = {
            "global_logit_exploit": 0.0,
            "global_emitter_stats": {
                "exploit": {"successes": 1.0, "failures": 1.0},
                "explore": {"successes": 1.0, "failures": 1.0},
            },
        }

    def initialize(self, candidates: list[Candidate]) -> None:
        if self._initialized:
            raise RuntimeError("EvolutionEngine is already initialized")
        if not candidates and self.config.initial_random_steps <= 0:
            raise ValueError("EvolutionEngine.initialize requires at least one initial candidate")
        if self.archive is None:
            self.archive = CVTArchive.create(
                descriptor_dim=self.config.descriptor_dim,
                num_centroids=self.config.num_centroids,
                samples=self.config.cvt_samples,
                seed=self.config.random_seed,
                candidate_registry=self.candidate_registry,
                rng=self._rng,
                elites_per_cell=self.config.elites_per_cell,
            )
        self.run_store.initialize_run(self.config, self.archive.centroids)
        for candidate in candidates:
            self._process_candidate(
                candidate=candidate,
                parent_ids=[],
                generation=0,
                created_at_step=0,
            )
        self.run_store.save_archive_snapshot(step=0, archive=self.archive, normalizer=self.normalizer, stage=self.stage, sampling_global_state=self._sampling_global_state)
        self.run_store.save_stats(self.stats.to_record(step=0, archive=self.archive))
        self._initialized = True

    def step(self) -> None:
        if not self._initialized:
            raise RuntimeError("EvolutionEngine must be initialized before step()")
        if self.archive is None:
            raise RuntimeError("EvolutionEngine archive is not initialized")
        prepared = self.prepare_step()
        executed = self.execute_prepared_step(prepared)
        self.commit_executed_step(executed)

    def _is_random_init_step(self, current_step: int) -> bool:
        return current_step <= self.config.initial_random_steps

    def prepare_step(self) -> PreparedStep:
        return self._prepare_step()

    def execute_prepared_step(self, prepared: PreparedStep) -> ExecutedStep:
        return self._execute_prepared_step(prepared)

    def commit_executed_step(self, executed: ExecutedStep) -> CommittedStep:
        return self._commit_executed_step(executed)

    def run(self, steps: int, *, stage_end_hook: Callable[[], None] | None = None) -> None:
        stages = self.config.secondary_eval.stages if self.secondary_fitness_fn is not None else 1
        for stage_idx in range(stages):
            self.stage = stage_idx
            self.run_store.update_metadata(current_step=self.step_count, current_stage=stage_idx)
            self._run_stage(steps)
            if self.secondary_fitness_fn is not None and stage_idx < stages - 1:
                seeds = self._run_stage_end_evaluation()
                if stage_end_hook is not None:
                    stage_end_hook()
                self._reset_for_next_stage(seeds, stage_idx + 1)
        self.run_store.update_metadata(current_step=self.step_count, current_stage=self.stage, status="completed")

    def _run_stage(self, steps: int) -> None:
        stage_start = self.step_count
        target = min(stage_start + steps, stage_start + self.config.max_steps)
        if self.config.fitness_batch_size > 1 and self.batch_primary_fitness_fn is not None:
            self._run_batched(target)
        elif self.config.parallel_workers == 1:
            while self.step_count < target:
                self.step()
        else:
            self._run_parallel(target)

    def _run_parallel(self, target: int) -> None:
        if target <= self.step_count:
            return
        in_flight: dict[Future[ExecutedStep], PreparedStep] = {}
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            while self.step_count < target:
                while len(in_flight) < self.config.parallel_workers and (self.step_count + len(in_flight)) < target:
                    prepared = self._prepare_step()
                    future = executor.submit(self._execute_prepared_step, prepared)
                    in_flight[future] = prepared
                if not in_flight:
                    break
                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                ordered_done = [future for future in list(in_flight.keys()) if future in done]
                for future in ordered_done:
                    executed = future.result()
                    del in_flight[future]
                    self.commit_executed_step(executed)
                    if self.step_count >= target:
                        break

    def _run_batched(self, target: int) -> None:
        """Run evolution with per-island batch fitness evaluation.

        Accumulates fitness_batch_size candidates, then calls batch_primary_fitness_fn
        with all of them at once before committing any to the archive.
        """
        assert self.batch_primary_fitness_fn is not None  # validated in __init__
        batch_size = self.config.fitness_batch_size

        while self.step_count < target:
            n = min(batch_size, target - self.step_count)
            batch_prepared = [self._prepare_step() for _ in range(n)]

            # Run LLM calls in parallel — fitness deferred (_skip_fitness=True)
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(self._execute_prepared_step, p, True) for p in batch_prepared]
                executed_steps = [f.result() for f in futures]

            # Gather all children waiting for fitness evaluation
            pending: list[tuple[int, int, Candidate]] = []
            for step_idx, executed in enumerate(executed_steps):
                if executed.error or not executed.children:
                    continue
                for child_idx, eval_ in enumerate(executed.children):
                    if eval_.status == "pending_fitness":
                        pending.append((step_idx, child_idx, eval_.candidate))

            if pending:
                candidates_to_eval = [c for _, _, c in pending]
                try:
                    batch_results = self.batch_primary_fitness_fn(candidates_to_eval)
                    for i, (step_idx, child_idx, candidate) in enumerate(pending):
                        try:
                            fitness, stats = _unwrap_fitness(batch_results[i])
                            candidate.primary_fitness = fitness
                            candidate.stats = stats
                            descriptor_raw = self.descriptor_fn(candidate)
                            candidate.descriptor_raw = [float(v) for v in descriptor_raw]
                            executed_steps[step_idx].children[child_idx] = CandidateEvaluation(
                                candidate=candidate, status="ready"
                            )
                        except Exception as exc:
                            executed_steps[step_idx].children[child_idx] = CandidateEvaluation(
                                candidate=candidate, status="primary_fitness_failure", error=str(exc)
                            )
                except Exception as exc:
                    self.error_feedback.record(exc)
                    for step_idx, child_idx, candidate in pending:
                        executed_steps[step_idx].children[child_idx] = CandidateEvaluation(
                            candidate=candidate, status="primary_fitness_failure", error=str(exc)
                        )

            for executed in executed_steps:
                self.commit_executed_step(executed)

    def _prepare_step(self) -> PreparedStep:
        current_step = self.step_count + 1
        in_random_init = self._is_random_init_step(current_step)
        emitter = self._choose_emitter()
        if in_random_init:
            parents = []
            ancestors = []
            inspirations = []
            target = Candidate(id="initialization_target", code="")
            prompt_extra_instructions = (
                f"{self._edit_output_instructions(initialization=True)}\n\n"
                f"{random_init_instructions(self._compose_extra_instructions(emitter=emitter))}"
            )
            prompt = build_prompt(
                template=self.config.user_prompt,
                target=target,
                ancestors=ancestors,
                inspirations=inspirations,
                extra_instructions=prompt_extra_instructions,
                primary_metric_label=self.config.primary_metric_label or "Primary",
                primary_validation_metric_label=self.config.primary_validation_metric_label or "Primary Validation",
                secondary_metric_label=self.config.secondary_metric_label or "Secondary",
                secondary_validation_metric_label=self.config.secondary_validation_metric_label or "Secondary Validation",
                code_language=self.config.code_language,
            )
        else:
            if self.archive is None:
                raise RuntimeError("EvolutionEngine archive is not initialized")
            sc = self.config.sampling
            _use_new_sampling = (
                sc is not None
                and self.config.generation_method == "llm_emitters_emitter_curiosity"
                and (sc.cell_method != "uniform" or sc.emitter_method != "fixed")
            )
            if _use_new_sampling:
                assert sc is not None  # mypy
                occupied_cell_ids = self.archive.occupied_cells()
                if not occupied_cell_ids:
                    raise ValueError("Archive is empty")
                cell_id = select_cell(
                    occupied_cell_ids,
                    self.archive,
                    sc,
                    self._rng,
                    self._sampling_total_trials,
                )
                cell_elite = self.archive.get_cell_elite(cell_id)
                if cell_elite is None:
                    raise RuntimeError(f"Cell {cell_id} has no elite after select_cell")
                # Build the full emitter name list: LLM emitters + local emitters
                llm_emitter_names = [e.name for e in (self.config.emitters or [])]
                all_emitter_names = llm_emitter_names + [
                    name for name in self.local_emitters if name not in llm_emitter_names
                ]
                emitter_name_override = select_emitter(
                    cell_elite,
                    self._sampling_global_state.get("global_logit_exploit", 0.0),
                    sc,
                    self._rng,
                    emitter_names=all_emitter_names if all_emitter_names else None,
                    global_state=self._sampling_global_state,
                )
                # Override the emitter chosen earlier with the bandit selection
                if self.config.emitters:
                    emitter_map = {e.name: e for e in self.config.emitters}
                    if emitter_name_override in emitter_map:
                        emitter = emitter_map[emitter_name_override]
                parents = [cell_elite]
                # Store the cell elite id so _commit can update its stats
                self._sampling_pending_elite_id = cell_elite.id
            else:
                elite_strategy = self.config.elite_selection_strategy
                if self.config.generation_method == "llm_emitters_emitter_curiosity":
                    elite_strategy = "emitter_curiosity_weighted"
                parents = sample_parents(
                    archive=self.archive,
                    n=1,
                    grid_strategy=self.config.grid_selection_strategy,
                    elite_strategy=elite_strategy,
                    grid_distance_threshold=self.config.grid_distance_threshold,
                    current_step=current_step,
                    grid_sampling_weights=self.config.grid_sampling_weights,
                    elite_sampling_weights=self.config.elite_sampling_weights,
                    grid_sampling_schedule=self.config.grid_sampling_schedule,
                    elite_sampling_schedule=self.config.elite_sampling_schedule,
                    emitter_name=emitter.name if emitter is not None else None,
                    curiosity_temperature=self.config.curiosity_temperature,
                )
                # Record parent for EMA stats update in _commit_executed_step
                if sc is not None and parents:
                    self._sampling_pending_elite_id = parents[0].id
            ancestors = self._collect_ancestors(parents[0], self._resolve_ancestor_count(emitter))
            inspirations = self._sample_inspirations(
                target=parents[0],
                ancestors=ancestors,
                count=self._resolve_inspiration_elite_count(emitter),
            )
            prompt_extra_instructions = (
                f"{self._edit_output_instructions(initialization=False)}\n\n"
                f"{self._compose_extra_instructions(emitter=emitter)}"
            ).strip()
            prompt = build_prompt(
                template=self.config.user_prompt,
                target=parents[0],
                ancestors=ancestors,
                inspirations=inspirations,
                extra_instructions=prompt_extra_instructions,
                primary_metric_label=self.config.primary_metric_label or "Primary",
                primary_validation_metric_label=self.config.primary_validation_metric_label or "Primary Validation",
                secondary_metric_label=self.config.secondary_metric_label or "Secondary",
                secondary_validation_metric_label=self.config.secondary_validation_metric_label or "Secondary Validation",
                code_language=self.config.code_language,
            )
        emitter_name = emitter.name if emitter is not None else None
        is_local = (not in_random_init) and (emitter_name is not None) and (emitter_name in self.local_emitters)
        return PreparedStep(
            in_random_init=in_random_init,
            parents=parents,
            ancestors=ancestors,
            inspirations=inspirations,
            prompt=prompt,
            emitter_name=emitter_name,
            thinking_mode=self._resolve_thinking_mode(emitter),
            is_local=is_local,
        )

    def _execute_prepared_step(self, prepared: PreparedStep, _skip_fitness: bool = False) -> ExecutedStep:
        # Route local emitters — skip LLM entirely
        if prepared.is_local and prepared.emitter_name in self.local_emitters:
            return self._execute_local_emitter(prepared)

        llm_result = None
        try:
            llm_result = self.llm_client.generate(
                prepared.prompt,
                emitter_name=prepared.emitter_name,
                thinking_mode=prepared.thinking_mode,
            )
            base_code = "" if prepared.in_random_init or not prepared.parents else prepared.parents[0].code
            parsed_children, llm_result = self._parse_children_with_repair(
                prepared=prepared,
                base_code=base_code,
                llm_result=llm_result,
            )
            evaluated_children: list[CandidateEvaluation] = []
            for child in parsed_children:
                child.parent_ids = [parent.id for parent in prepared.parents]
                child.metadata.update(
                    {
                        "target_candidate_id": prepared.parents[0].id if prepared.parents else None,
                        "ancestor_ids": [candidate.id for candidate in prepared.ancestors],
                        "inspiration_ids": [candidate.id for candidate in prepared.inspirations],
                        "emitter_name": prepared.emitter_name,
                    }
                )
                try:
                    if self.candidate_validator is not None and not self.candidate_validator(child):
                        evaluated_children.append(
                            CandidateEvaluation(candidate=child, status="candidate_rejected", error="validator_failed")
                        )
                        continue
                    if _skip_fitness:
                        # Batch mode: defer fitness evaluation to _run_batched
                        evaluated_children.append(CandidateEvaluation(candidate=child, status="pending_fitness"))
                        continue
                    child.primary_fitness, child.stats = _unwrap_fitness(self.primary_fitness_fn(child))
                    if self.primary_validation_fitness_fn is not None:
                        child.primary_validation_fitness, _ = _unwrap_fitness(self.primary_validation_fitness_fn(child))
                except Exception as exc:
                    self.error_feedback.record(exc)
                    evaluated_children.append(
                        CandidateEvaluation(candidate=child, status="primary_fitness_failure", error=str(exc))
                    )
                    continue
                try:
                    descriptor_raw = self.descriptor_fn(child)
                    child.descriptor_raw = [float(value) for value in descriptor_raw]
                except Exception as exc:
                    evaluated_children.append(
                        CandidateEvaluation(candidate=child, status="descriptor_failure", error=str(exc))
                    )
                    continue
                evaluated_children.append(CandidateEvaluation(candidate=child, status="ready"))
            return ExecutedStep(prepared=prepared, llm_result=llm_result, children=evaluated_children)
        except (LLMError, ParsingError, ValidationError, DescriptorError, FitnessError, ValueError) as exc:
            llm_result = getattr(exc, "llm_result", llm_result)
            return ExecutedStep(prepared=prepared, llm_result=llm_result, error=exc)

    def _execute_local_emitter(self, prepared: PreparedStep) -> ExecutedStep:
        """Execute a local (non-LLM) emitter: call the registered function, evaluate all results."""
        local_fn = self.local_emitters[prepared.emitter_name]  # type: ignore[index]
        parent = prepared.parents[0] if prepared.parents else None
        try:
            raw_candidates = local_fn(parent) if parent is not None else []
        except Exception as exc:
            return ExecutedStep(prepared=prepared, llm_result=None, children=None, error=exc)

        evaluated: list[CandidateEvaluation] = []
        for child in raw_candidates:
            child.parent_ids = [parent.id] if parent is not None else []
            child.metadata.update({
                "target_candidate_id": parent.id if parent is not None else None,
                "emitter_name": prepared.emitter_name,
                "local_emitter": True,
            })
            try:
                if self.candidate_validator is not None and not self.candidate_validator(child):
                    evaluated.append(CandidateEvaluation(candidate=child, status="candidate_rejected", error="validator_failed"))
                    continue
                child.primary_fitness, child.stats = _unwrap_fitness(self.primary_fitness_fn(child))
                if self.primary_validation_fitness_fn is not None:
                    child.primary_validation_fitness, _ = _unwrap_fitness(self.primary_validation_fitness_fn(child))
                child.descriptor_raw = [float(v) for v in self.descriptor_fn(child)]
                evaluated.append(CandidateEvaluation(candidate=child, status="ready"))
            except Exception as exc:
                evaluated.append(CandidateEvaluation(candidate=child, status="primary_fitness_failure", error=str(exc)))

        return ExecutedStep(prepared=prepared, llm_result=None, children=evaluated, error=None)

    def _parse_children_with_repair(
        self,
        *,
        prepared: PreparedStep,
        base_code: str,
        llm_result: Any,
    ) -> tuple[list[Candidate], Any]:
        if self.config.output_format == "full" or prepared.in_random_init:
            return parse_full_candidates(llm_result.response_text, self.config.parsing), llm_result

        parsing = self.config.parsing
        max_attempts = parsing.parse_repair_max_attempts if parsing.parse_repair_enabled else 0

        current_result = llm_result
        last_exc: ParsingError | None = None
        initial_exc: ParsingError | None = None

        for attempt in range(max_attempts + 1):
            try:
                children = parse_edit_candidates(current_result.response_text, base_code, self.config.parsing)
                if attempt > 0:
                    for candidate in children:
                        candidate.metadata.update({
                            "parse_repair_attempted": True,
                            "parse_repair_attempts": attempt,
                            "initial_parse_error": str(initial_exc),
                        })
                return children, current_result
            except ParsingError as exc:
                last_exc = exc
                if initial_exc is None:
                    initial_exc = exc
                if attempt >= max_attempts:
                    break
                repair_prompt = self._build_parse_repair_prompt(
                    original_prompt=prepared.prompt,
                    broken_response=current_result.response_text,
                    parse_error=exc,
                    base_code=base_code,
                )
                current_result = self.llm_client.generate(
                    repair_prompt,
                    emitter_name=prepared.emitter_name,
                    thinking_mode=prepared.thinking_mode,
                )

        assert last_exc is not None
        setattr(last_exc, "llm_result", current_result)
        raise last_exc

    def _commit_executed_step(self, executed: ExecutedStep) -> CommittedStep:
        current_step = self.step_count + 1
        if executed.error is not None:
            self._log_failed_llm_call(current_step=current_step, executed=executed)
            self._handle_step_failure(current_step=current_step, exc=executed.error)
            return CommittedStep(step=current_step, llm_call_id=None, child_ids=[], inserted_candidates=[])
        is_local = executed.prepared.is_local
        if self.archive is None or executed.children is None:
            raise RuntimeError("Executed step is missing required state for commit")
        if not is_local and executed.llm_result is None:
            raise RuntimeError("Executed step is missing LLM result for non-local step")
        self.generation += 1
        llm_call_id = f"local_{current_step:06d}_{uuid4().hex[:8]}" if is_local else f"llm_{current_step:06d}_{uuid4().hex[:8]}"
        child_ids: list[str] = []
        inserted_candidates: list[Candidate] = []
        any_inserted = False
        for evaluation in executed.children:
            commit_result = self._commit_candidate_evaluation(
                candidate=evaluation.candidate,
                evaluation=evaluation,
                parent_ids=[parent.id for parent in executed.prepared.parents],
                generation=self.generation,
                created_at_step=current_step,
                llm_call_id=llm_call_id,
            )
            child_ids.append(commit_result.candidate_id)
            if commit_result.inserted:
                inserted_candidates.append(self.candidate_registry[commit_result.candidate_id])
                any_inserted = True
            self._update_parent_curiosity(
                parent_ids=[parent.id for parent in executed.prepared.parents],
                child_id=commit_result.candidate_id,
                current_step=current_step,
                inserted=commit_result.inserted,
                emitter_name=executed.prepared.emitter_name,
            )
        # Update bandit sampling stats whenever a parent elite was sampled
        sc = self.config.sampling
        if (
            sc is not None
            and self._sampling_pending_elite_id is not None
            and executed.prepared.emitter_name is not None
        ):
            parent_elite = self.candidate_registry.get(self._sampling_pending_elite_id)
            if parent_elite is not None:
                reward = compute_reward(any_inserted)
                update_cell_and_global_stats(
                    parent_elite,
                    executed.prepared.emitter_name,
                    reward,
                    self._sampling_global_state,
                    sc,
                )
                self._sampling_total_trials += 1
                self._sampling_global_logit_exploit = float(
                    self._sampling_global_state.get("global_logit_exploit", 0.0)
                )
                self._save_candidate(parent_elite)
        self._sampling_pending_elite_id = None
        if is_local:
            # Local emitters don't produce LLM calls — log a lightweight event instead
            self.run_store.log_event("local_emitter_step", {
                "step": current_step,
                "emitter_name": executed.prepared.emitter_name,
                "candidates_generated": len(child_ids),
                "inserted": any_inserted,
            })
            if current_step % self.config.remap_interval == 0:
                remap_result = self.archive.remap(self.archive.all_candidates(), self.normalizer)
                self.run_store.log_event("remap", {"step": current_step, "occupied_cells": len(self.archive.occupied_cells()), **vars(remap_result)})
            if current_step % self.config.storage.archive_snapshot_interval == 0 or current_step % self.config.remap_interval == 0:
                self.run_store.save_archive_snapshot(step=current_step, archive=self.archive, normalizer=self.normalizer, stage=self.stage, sampling_global_state=self._sampling_global_state)
            self.step_count = current_step
            self._consecutive_failures = 0
            self.run_store.update_metadata(current_step=self.step_count, status="running")
            self.run_store.save_stats(self.stats.to_record(step=self.step_count, archive=self.archive, stage=self.stage))
            return CommittedStep(step=current_step, llm_call_id=None, child_ids=child_ids, inserted_candidates=inserted_candidates)
        self.run_store.log_llm_call(
            step=current_step,
            parent_ids=[parent.id for parent in executed.prepared.parents],
            llm_call_id=llm_call_id,
            prompt=executed.llm_result.prompt,  # type: ignore[union-attr]
            response=executed.llm_result.response_text,  # type: ignore[union-attr]
            reasoning=getattr(executed.llm_result, "reasoning_text", None),
            metadata={
                "child_ids": child_ids,
                "model": executed.llm_result.model,  # type: ignore[union-attr]
                "status": executed.llm_result.status,  # type: ignore[union-attr]
                "latency_seconds": executed.llm_result.latency_seconds,  # type: ignore[union-attr]
                "started_at": executed.llm_result.started_at,  # type: ignore[union-attr]
                "finished_at": executed.llm_result.finished_at,  # type: ignore[union-attr]
                "phase": "random_init" if executed.prepared.in_random_init else "evolution",
                "emitter_name": executed.prepared.emitter_name,
                "thinking_mode": executed.prepared.thinking_mode,
                "request_payload": executed.llm_result.request_payload,
                "response_payload": executed.llm_result.response_payload,
                "stage": self.stage,
            },
        )
        if current_step % self.config.remap_interval == 0:
            remap_result = self.archive.remap(self.archive.all_candidates(), self.normalizer)
            self.run_store.log_event(
                "remap",
                {
                    "step": current_step,
                    "occupied_cells": len(self.archive.occupied_cells()),
                    "before_occupied_cells": remap_result.before_occupied_cells,
                    "after_occupied_cells": remap_result.after_occupied_cells,
                    "moved_count": remap_result.moved_count,
                    "collision_count": remap_result.collision_count,
                    "lost_candidate_ids": remap_result.lost_candidate_ids,
                    "cell_changes": remap_result.cell_changes,
                },
            )
        if current_step % self.config.storage.archive_snapshot_interval == 0 or current_step % self.config.remap_interval == 0:
            self.run_store.save_archive_snapshot(step=current_step, archive=self.archive, normalizer=self.normalizer, stage=self.stage, sampling_global_state=self._sampling_global_state)
        self.step_count = current_step
        self._consecutive_failures = 0
        self.run_store.update_metadata(current_step=self.step_count, status="running")
        if current_step % self.config.storage.checkpoint_interval == 0:
            self.save_checkpoint(current_step)
        self.run_store.save_stats(self.stats.to_record(step=self.step_count, archive=self.archive, stage=self.stage))
        return CommittedStep(
            step=self.step_count,
            llm_call_id=llm_call_id,
            child_ids=child_ids,
            inserted_candidates=inserted_candidates,
        )

    def _choose_emitter(self) -> EmitterConfig | None:
        if self.config.generation_method not in {"llm_emitters", "llm_emitters_emitter_curiosity"} or not self.config.emitters:
            return None
        if self.config.emitter_selection_strategy == "average_curiosity":
            return self._choose_emitter_by_curiosity()
        if self.config.emitter_selection_strategy == "plateau_scheduler":
            return self._choose_emitter_by_plateau_scheduler()
        weights = np.asarray([emitter.selection_weight for emitter in self.config.emitters], dtype=float)
        total_weight = float(weights.sum())
        if total_weight <= 0:
            raise ValueError("Emitter selection weights must have positive total weight")
        probabilities = weights / total_weight
        index = int(self._rng.choice(len(self.config.emitters), p=probabilities))
        return self.config.emitters[index]

    def _choose_emitter_by_curiosity(self) -> EmitterConfig:
        if not self.config.emitters:
            raise ValueError("Emitter curiosity selection requires configured emitters")
        if self.archive is None:
            index = int(self._rng.integers(0, len(self.config.emitters)))
            return self.config.emitters[index]
        elites = self.archive.all_elites()
        if not elites:
            index = int(self._rng.integers(0, len(self.config.emitters)))
            return self.config.emitters[index]
        averages = []
        for emitter in self.config.emitters:
            values = [float(candidate.emitter_curiosity_scores.get(emitter.name, self.config.curiosity_default)) for candidate in elites]
            averages.append(sum(values) / len(values) if values else float(self.config.curiosity_default))
        probabilities = np.asarray(averages, dtype=float)
        probabilities = probabilities / probabilities.sum() if float(probabilities.sum()) > 0 else np.ones(len(averages)) / len(averages)
        index = int(self._rng.choice(len(self.config.emitters), p=probabilities))
        return self.config.emitters[index]

    def _choose_emitter_by_plateau_scheduler(self) -> EmitterConfig:
        if not self.config.emitters or self.config.emitter_plateau_scheduler is None:
            raise ValueError("Plateau emitter scheduler requires configured emitters and emitter_plateau_scheduler")
        scheduler = self.config.emitter_plateau_scheduler
        plateau_steps = max(0, self.step_count - self._best_primary_improvement_step)
        plateau_stage = plateau_steps // scheduler.patience_steps
        weights = []
        for emitter in self.config.emitters:
            weight = float(emitter.selection_weight)
            if emitter.name == scheduler.target_emitter_name:
                scale = scheduler.target_weight_growth ** plateau_stage
                if scheduler.max_target_weight_scale is not None:
                    scale = min(scale, scheduler.max_target_weight_scale)
            else:
                scale = scheduler.non_target_weight_decay ** plateau_stage
                scale = max(scale, scheduler.min_non_target_weight_scale)
            weights.append(weight * float(scale))
        probabilities = np.asarray(weights, dtype=float)
        if float(probabilities.sum()) <= 0:
            probabilities = np.ones(len(weights), dtype=float)
        probabilities = probabilities / probabilities.sum()
        index = int(self._rng.choice(len(self.config.emitters), p=probabilities))
        return self.config.emitters[index]

    def _compose_extra_instructions(self, *, emitter: EmitterConfig | None) -> str:
        sections = [self.config.extra_instructions.strip()]
        if emitter is not None and emitter.extra_instructions.strip():
            sections.append(emitter.extra_instructions.strip())
        # Inject runtime error feedback from recent evaluation failures
        error_block = self.error_feedback.format_for_prompt()
        if error_block:
            sections.append(error_block)
        return "\n\n".join(section for section in sections if section)

    def _resolve_thinking_mode(self, emitter: EmitterConfig | None) -> ThinkingMode:
        if emitter is None:
            return self.config.llm.thinking_mode
        if emitter.thinking_mode != "default":
            return emitter.thinking_mode
        if emitter.llm is not None:
            return emitter.llm.thinking_mode
        return self.config.llm.thinking_mode

    def _resolve_ancestor_count(self, emitter: EmitterConfig | None) -> int:
        if emitter is not None and emitter.ancestor_count is not None:
            return emitter.ancestor_count
        return self.config.ancestor_count

    def _resolve_inspiration_elite_count(self, emitter: EmitterConfig | None) -> int:
        if emitter is not None and emitter.inspiration_elite_count is not None:
            return emitter.inspiration_elite_count
        return self.config.inspiration_elite_count

    def save_checkpoint(self, step: int | None = None) -> None:
        if self.archive is None:
            raise RuntimeError("Cannot save checkpoint without an archive")
        checkpoint_step = self.step_count if step is None else step
        state = {
            "config": self.config.to_dict(),
            "archive": self.archive.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "candidates": {candidate_id: candidate.to_dict() for candidate_id, candidate in self.candidate_registry.items()},
            "record_versions": dict(self._record_versions),
            "recent_elite_ids": list(self._recent_elite_ids),
            "step_count": self.step_count,
            "stage": self.stage,
            "generation": self.generation,
            "next_candidate_index": self._next_candidate_index,
            "best_primary_seen": self._best_primary_seen,
            "best_primary_improvement_step": self._best_primary_improvement_step,
            "stats": self.stats.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "engine_rng_state": self._rng.bit_generator.state,
            "run_dir": str(self.run_store.run_dir),
            "run_id": self.run_store.run_id,
            "sampling_total_trials": self._sampling_total_trials,
            "sampling_global_logit_exploit": self._sampling_global_logit_exploit,
            "sampling_global_state": dict(self._sampling_global_state),
        }
        self.run_store.save_checkpoint(state, checkpoint_step)
        self.run_store.log_event("checkpoint_saved", {"step": checkpoint_step})

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        primary_fitness: Callable[[Candidate], float],
        descriptor_fn: Callable[[Candidate], list[float]],
        llm_client: LLMClient,
        secondary_fitness: Callable[[Candidate], float] | None = None,
        primary_validation_fitness: Callable[[Candidate], float] | None = None,
        secondary_validation_fitness: Callable[[Candidate], float] | None = None,
        candidate_validator: Callable[[Candidate], bool] | None = None,
        resume_mode: ResumeMode | None = None,
    ) -> "EvolutionEngine":
        checkpoint_file = Path(checkpoint_path)
        raw_state = load_checkpoint_file(checkpoint_file)
        config = EvolutionConfig.from_dict(raw_state["config"])
        chosen_resume_mode = resume_mode or config.resume_mode
        candidate_registry = {
            candidate_id: Candidate.from_dict(candidate_data)
            for candidate_id, candidate_data in raw_state["candidates"].items()
        }
        rng = np.random.default_rng()
        rng.bit_generator.state = raw_state["engine_rng_state"]
        archive = CVTArchive.from_state_dict(
            raw_state["archive"],
            candidate_registry=candidate_registry,
            rng=rng,
        )
        normalizer = DescriptorNormalizer(config.normalizer)
        normalizer.load_state_dict(raw_state["normalizer"])
        if chosen_resume_mode == "continue":
            run_store = RunStore(
                config=config.storage,
                run_name=config.run_name,
                output_dir=config.output_dir,
                run_id=raw_state["run_id"],
                existing_run_dir=raw_state["run_dir"],
            )
            run_store.attach_existing_run()
        else:
            fork_storage = replace(config.storage, run_id=None, overwrite=False)
            run_store = RunStore(
                config=fork_storage,
                run_name=config.run_name,
                output_dir=config.output_dir,
            )
            run_store.initialize_run(config, archive.centroids)
            run_store.save_archive_snapshot(step=raw_state["step_count"], archive=archive, normalizer=normalizer, stage=int(raw_state.get("stage", 0)))
        engine = cls(
            config=config,
            llm_client=llm_client,
            primary_fitness=primary_fitness,
            primary_validation_fitness=primary_validation_fitness,
            descriptor_fn=descriptor_fn,
            secondary_fitness=secondary_fitness,
            secondary_validation_fitness=secondary_validation_fitness,
            candidate_validator=candidate_validator,
            archive=archive,
            normalizer=normalizer,
            run_store=run_store,
        )
        engine.candidate_registry = candidate_registry
        engine.archive._candidate_registry = engine.candidate_registry
        engine._record_versions = {candidate_id: int(version) for candidate_id, version in raw_state["record_versions"].items()}
        engine._recent_elite_ids = deque(raw_state["recent_elite_ids"], maxlen=1000)
        engine.step_count = int(raw_state["step_count"])
        engine.stage = int(raw_state.get("stage", 0))
        engine.generation = int(raw_state["generation"])
        engine._next_candidate_index = int(raw_state["next_candidate_index"])
        engine._best_primary_seen = raw_state.get("best_primary_seen")
        engine._best_primary_improvement_step = int(raw_state.get("best_primary_improvement_step", 0))
        engine.stats = engine.stats.from_state_dict(raw_state["stats"])
        engine._sampling_total_trials = int(raw_state.get("sampling_total_trials", 0))
        engine._sampling_global_logit_exploit = float(raw_state.get("sampling_global_logit_exploit", 0.0))
        engine._sampling_global_state = dict(raw_state.get("sampling_global_state", {
            "global_logit_exploit": 0.0,
            "global_emitter_stats": {
                "exploit": {"successes": 1.0, "failures": 1.0},
                "explore": {"successes": 1.0, "failures": 1.0},
            },
        }))
        random.setstate(raw_state["python_random_state"])
        np.random.set_state(raw_state["numpy_random_state"])
        engine._rng = rng
        engine.archive._rng = engine._rng
        engine._initialized = True
        engine.run_store.update_metadata(current_step=engine.step_count, status="running")
        engine.run_store.log_event("checkpoint_loaded", {"step": engine.step_count, "mode": chosen_resume_mode})
        return engine

    def _process_candidate(
        self,
        *,
        candidate: Candidate,
        parent_ids: list[str],
        generation: int,
        created_at_step: int,
        llm_call_id: str | None = None,
    ) -> None:
        if not candidate.id:
            candidate.id = self._new_candidate_id()
        elif candidate.id in self.candidate_registry:
            raise ValidationError(f"Duplicate candidate id: {candidate.id}")
        candidate.parent_ids = list(parent_ids)
        self._initialize_candidate_curiosity(candidate)
        candidate.llm_call_id = llm_call_id
        candidate.generation = generation
        candidate.created_at_step = created_at_step
        candidate.stage = self.stage
        self.candidate_registry[candidate.id] = candidate
        self.stats.generated_count += 1
        try:
            if self.candidate_validator is not None and not self.candidate_validator(candidate):
                self.stats.validation_failure_count += 1
                self._save_candidate(candidate, force=True)
                self.run_store.log_event(
                    "candidate_rejected",
                    {"step": created_at_step, "candidate_id": candidate.id, "reason": "validator_failed"},
                )
                return
            try:
                candidate.primary_fitness, candidate.stats = _unwrap_fitness(self.primary_fitness_fn(candidate))
                if self.primary_validation_fitness_fn is not None:
                    candidate.primary_validation_fitness, _ = _unwrap_fitness(self.primary_validation_fitness_fn(candidate))
            except Exception as exc:
                self.stats.fitness_failure_count += 1
                self.error_feedback.record(exc)
                self._save_candidate(candidate, force=True)
                self.run_store.log_event(
                    "primary_fitness_failure",
                    {"step": created_at_step, "candidate_id": candidate.id, "error": str(exc)},
                )
                return
            try:
                descriptor_raw = self.descriptor_fn(candidate)
                candidate.descriptor_raw = [float(value) for value in descriptor_raw]
                self.normalizer.update(candidate.descriptor_raw)
                candidate.descriptor_norm = self.normalizer.normalize(candidate.descriptor_raw)
            except Exception as exc:
                self.stats.descriptor_failure_count += 1
                self._save_candidate(candidate, force=True)
                self.run_store.log_event(
                    "descriptor_failure",
                    {"step": created_at_step, "candidate_id": candidate.id, "error": str(exc)},
                )
                return
            insert_result = self.archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
            self._save_candidate(candidate, force=True)
            self.run_store.log_event(
                "candidate_inserted" if insert_result.inserted else "candidate_not_inserted",
                {
                    "step": created_at_step,
                    "candidate_id": candidate.id,
                    "cell_id": insert_result.cell_id,
                    "inserted": insert_result.inserted,
                    "replaced_candidate_id": insert_result.replaced_candidate_id,
                    "llm_call_id": llm_call_id,
                },
            )
            if insert_result.inserted:
                self.stats.accepted_count += 1
                self._recent_elite_ids.append(candidate.id)
                self._register_primary_improvement(candidate.primary_fitness, created_at_step)
        except ValidationError:
            raise

    def _commit_candidate_evaluation(
        self,
        *,
        candidate: Candidate,
        evaluation: CandidateEvaluation,
        parent_ids: list[str],
        generation: int,
        created_at_step: int,
        llm_call_id: str | None = None,
    ) -> CandidateCommitResult:
        if not candidate.id:
            candidate.id = self._new_candidate_id()
        elif candidate.id in self.candidate_registry:
            raise ValidationError(f"Duplicate candidate id: {candidate.id}")
        candidate.parent_ids = list(parent_ids)
        self._initialize_candidate_curiosity(candidate)
        candidate.llm_call_id = llm_call_id
        candidate.generation = generation
        candidate.created_at_step = created_at_step
        candidate.stage = self.stage
        self.candidate_registry[candidate.id] = candidate
        self.stats.generated_count += 1
        if evaluation.status == "candidate_rejected":
            self.stats.validation_failure_count += 1
            self._save_candidate(candidate, force=True)
            self.run_store.log_event(
                "candidate_rejected",
                {"step": created_at_step, "candidate_id": candidate.id, "reason": evaluation.error or "validator_failed"},
            )
            return CandidateCommitResult(candidate_id=candidate.id, status=evaluation.status, inserted=False)
        if evaluation.status == "descriptor_failure":
            self.stats.descriptor_failure_count += 1
            self._save_candidate(candidate, force=True)
            self.run_store.log_event(
                "descriptor_failure",
                {"step": created_at_step, "candidate_id": candidate.id, "error": evaluation.error or "descriptor failure"},
            )
            return CandidateCommitResult(candidate_id=candidate.id, status=evaluation.status, inserted=False)
        if evaluation.status == "primary_fitness_failure":
            self.stats.fitness_failure_count += 1
            self._save_candidate(candidate, force=True)
            self.run_store.log_event(
                "primary_fitness_failure",
                {"step": created_at_step, "candidate_id": candidate.id, "error": evaluation.error or "fitness failure"},
            )
            return CandidateCommitResult(candidate_id=candidate.id, status=evaluation.status, inserted=False)
        if evaluation.status != "ready":
            raise ValidationError(f"Unknown candidate evaluation status: {evaluation.status}")
        if candidate.descriptor_raw is None:
            raise DescriptorError(f"Candidate {candidate.id} has no descriptor_raw during commit")
        if candidate.primary_fitness is None:
            raise FitnessError(f"Candidate {candidate.id} has no primary_fitness during commit")
        self.normalizer.update(candidate.descriptor_raw)
        candidate.descriptor_norm = self.normalizer.normalize(candidate.descriptor_raw)
        insert_result = self.archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
        self._save_candidate(candidate, force=True)
        self.run_store.log_event(
            "candidate_inserted" if insert_result.inserted else "candidate_not_inserted",
            {
                "step": created_at_step,
                "candidate_id": candidate.id,
                "cell_id": insert_result.cell_id,
                "inserted": insert_result.inserted,
                "replaced_candidate_id": insert_result.replaced_candidate_id,
                "llm_call_id": llm_call_id,
            },
        )
        if insert_result.inserted:
            self.stats.accepted_count += 1
            self._recent_elite_ids.append(candidate.id)
            self._register_primary_improvement(candidate.primary_fitness, created_at_step)
        return CandidateCommitResult(
            candidate_id=candidate.id,
            status=evaluation.status,
            inserted=insert_result.inserted,
            cell_id=insert_result.cell_id,
            replaced_candidate_id=insert_result.replaced_candidate_id,
        )

    def integrate_migrant(
        self,
        candidate: Candidate,
        *,
        source_island_id: str,
        source_candidate_id: str,
    ) -> CandidateCommitResult:
        from evo_lib.candidate import Candidate as _Candidate
        migrated = _Candidate(
            id="",
            code=candidate.code,
            metadata={
                **candidate.metadata,
                "migrated_from_island": source_island_id,
                "migrated_from_candidate_id": source_candidate_id,
            },
        )
        try:
            if self.candidate_validator is not None and not self.candidate_validator(migrated):
                evaluation = CandidateEvaluation(
                    candidate=migrated,
                    status="candidate_rejected",
                    error="validator_failed",
                )
            else:
                try:
                    migrated.primary_fitness, migrated.stats = _unwrap_fitness(self.primary_fitness_fn(migrated))
                    if self.primary_validation_fitness_fn is not None:
                        migrated.primary_validation_fitness, _ = _unwrap_fitness(self.primary_validation_fitness_fn(migrated))
                except Exception as exc:
                    evaluation = CandidateEvaluation(
                        candidate=migrated,
                        status="primary_fitness_failure",
                        error=str(exc),
                    )
                else:
                    try:
                        migrated.descriptor_raw = [float(v) for v in self.descriptor_fn(migrated)]
                    except Exception as exc:
                        evaluation = CandidateEvaluation(
                            candidate=migrated,
                            status="descriptor_failure",
                            error=str(exc),
                        )
                    else:
                        evaluation = CandidateEvaluation(candidate=migrated, status="ready")
        except Exception as exc:
            evaluation = CandidateEvaluation(candidate=migrated, status="descriptor_failure", error=str(exc))
        result = self._commit_candidate_evaluation(
            candidate=migrated,
            evaluation=evaluation,
            parent_ids=[source_candidate_id],
            created_at_step=self.step_count,
            generation=self.generation,
        )
        self.run_store.log_event(
            "candidate_migrated_in",
            {
                "step": self.step_count,
                "candidate_id": migrated.id,
                "source_island_id": source_island_id,
                "source_candidate_id": source_candidate_id,
                "inserted": result.inserted,
                "cell_id": result.cell_id,
                "replaced_candidate_id": result.replaced_candidate_id,
                "status": result.status,
            },
        )
        return result

    def _log_secondary_eval_progress(self, candidate: Candidate, index: int, total: int) -> None:
        self.run_store.log_event(
            "secondary_eval_progress",
            {
                "step": self.step_count,
                "stage": self.stage,
                "candidate_id": candidate.id,
                "index": index,
                "total": total,
                "secondary_fitness": candidate.secondary_fitness,
            },
        )
        self._save_candidate(candidate, force=True)

    def _run_stage_end_evaluation(self) -> list[Candidate]:
        """Evaluate secondary fitness on P1 of archive elites, return P2 seeds for next stage."""
        if self.archive is None or self.secondary_fitness_fn is None:
            return []
        try:
            evaluated, seeds = run_stage_end_evaluation(
                archive=self.archive,
                p1=self.config.secondary_eval.p1,
                p2=self.config.secondary_eval.p2,
                secondary_fitness=self.secondary_fitness_fn,
                secondary_validation_fitness=self.secondary_validation_fitness_fn,
                rng=self._rng,
                min_seeds=self.config.secondary_eval.min_seeds,
                p2_threshold=self.config.secondary_eval.p2_threshold,
                per_candidate_callback=self._log_secondary_eval_progress,
            )
            self.stats.secondary_eval_count += len(evaluated)
            elites = self.archive.all_elites() if self.archive else []
            primary_vals = [e.primary_fitness for e in elites if e.primary_fitness is not None]
            self.run_store.log_event(
                "stage_end",
                {
                    "step": self.step_count,
                    "stage": self.stage,
                    "archive_occupancy": len(elites),
                    "best_primary_fitness": max(primary_vals) if primary_vals else None,
                    "mean_primary_fitness": sum(primary_vals) / len(primary_vals) if primary_vals else None,
                    "secondary_evaluated": len(evaluated),
                    "seeds_selected": len(seeds),
                    "seed_ids": [s.id for s in seeds],
                },
            )
            # Candidates already saved individually via per_candidate_callback.
            return seeds
        except Exception as exc:
            self.stats.fitness_failure_count += 1
            self.run_store.log_event(
                "secondary_fitness_failure",
                {"step": self.step_count, "stage": self.stage, "error": str(exc)},
            )
            if self.config.errors.fail_fast:
                raise FitnessError(str(exc)) from exc
            return []

    def _reset_for_next_stage(self, seeds: list[Candidate], next_stage: int) -> None:
        """Clear the archive, reset normalizer, and re-insert seeds to start the next stage."""
        if self.archive is None:
            return
        self.archive.clear()
        self.normalizer = DescriptorNormalizer(self.config.normalizer)
        self.normalizer.descriptor_labels = list(self.config.descriptor_labels or [])
        for seed in seeds:
            if seed.descriptor_raw is None or seed.primary_fitness is None:
                continue
            self.normalizer.update(seed.descriptor_raw)
            seed.descriptor_norm = self.normalizer.normalize(seed.descriptor_raw)
            self.archive.insert(seed, seed.descriptor_norm, seed.primary_fitness)
            self._save_candidate(seed, force=True)
        self.run_store.save_archive_snapshot(step=self.step_count, archive=self.archive, normalizer=self.normalizer, stage=self.stage, sampling_global_state=self._sampling_global_state)
        self.run_store.log_event(
            "stage_start",
            {"step": self.step_count, "stage": next_stage, "seeds": len(seeds)},
        )
        self.save_checkpoint(self.step_count)

    def _save_candidate(self, candidate: Candidate, force: bool = False) -> None:
        """Save candidate to storage. Only writes a new record if meaningful fields changed.

        Suppresses writes that only update curiosity scores or sampling stats
        to prevent history bloat from re-evaluations and remap events.
        force=True bypasses the check (used on initial save and fitness updates).
        """
        key = (
            candidate.primary_fitness,
            candidate.secondary_fitness,
            candidate.cell_id,
            candidate.is_active,
            len(candidate.code),
        )
        prev_key = self._save_candidate_keys.get(candidate.id)
        if not force and prev_key is not None and prev_key == key:
            return  # nothing meaningful changed — skip to avoid history bloat
        self._save_candidate_keys[candidate.id] = key
        candidate.record_version = self._record_versions.get(candidate.id, 0) + 1
        self._record_versions[candidate.id] = candidate.record_version
        self.run_store.save_candidate(candidate)

    def _initialize_candidate_curiosity(self, candidate: Candidate) -> None:
        if candidate.curiosity_score is None:
            candidate.curiosity_score = float(self.config.curiosity_default)
        if candidate.curiosity_updates is None:
            candidate.curiosity_updates = 0
        if candidate.emitter_curiosity_scores is None:
            candidate.emitter_curiosity_scores = {}
        if candidate.emitter_curiosity_updates is None:
            candidate.emitter_curiosity_updates = {}
        for emitter in self.config.emitters or []:
            candidate.emitter_curiosity_scores.setdefault(emitter.name, float(self.config.curiosity_default))
            candidate.emitter_curiosity_updates.setdefault(emitter.name, 0)

    def _update_parent_curiosity(
        self,
        *,
        parent_ids: list[str],
        child_id: str,
        current_step: int,
        inserted: bool,
        emitter_name: str | None = None,
    ) -> None:
        if not parent_ids:
            return
        updated_parent_ids: list[str] = []
        seen_ids: set[str] = set()
        for parent_id in parent_ids:
            if parent_id in seen_ids:
                continue
            seen_ids.add(parent_id)
            parent = self.candidate_registry.get(parent_id)
            if parent is None:
                continue
            self._initialize_candidate_curiosity(parent)
            multiplier = self.config.curiosity_reward if inserted else self.config.curiosity_penalty
            parent.curiosity_score = float(parent.curiosity_score or self.config.curiosity_default) * float(multiplier)
            parent.curiosity_updates += 1
            if emitter_name:
                current_emitter_curiosity = float(parent.emitter_curiosity_scores.get(emitter_name, self.config.curiosity_default))
                parent.emitter_curiosity_scores[emitter_name] = current_emitter_curiosity * float(multiplier)
                parent.emitter_curiosity_updates[emitter_name] = int(parent.emitter_curiosity_updates.get(emitter_name, 0)) + 1
            self._save_candidate(parent)
            updated_parent_ids.append(parent_id)
        if updated_parent_ids:
            self.run_store.log_event(
                "parent_curiosity_updated",
                {
                    "step": current_step,
                    "child_id": child_id,
                    "parent_ids": updated_parent_ids,
                    "multiplier": float(self.config.curiosity_reward if inserted else self.config.curiosity_penalty),
                    "inserted": inserted,
                    "emitter_name": emitter_name,
                },
            )

    def _new_candidate_id(self) -> str:
        candidate_id = f"cand_{self._next_candidate_index:06d}"
        self._next_candidate_index += 1
        return candidate_id

    def _register_primary_improvement(self, fitness: float | None, step: int) -> None:
        if fitness is None:
            return
        if self._best_primary_seen is None or float(fitness) > float(self._best_primary_seen):
            self._best_primary_seen = float(fitness)
            self._best_primary_improvement_step = int(step)

    def _handle_step_failure(self, current_step: int, exc: Exception) -> None:
        self._consecutive_failures += 1
        if isinstance(exc, LLMError):
            self.stats.llm_error_count += 1
            event_type = "llm_error"
        elif isinstance(exc, ParsingError):
            self.stats.parse_failure_count += 1
            event_type = "parse_failure"
        else:
            event_type = "step_failure"
        self.run_store.log_event(
            event_type,
            {"step": current_step, "error": str(exc)},
        )
        if self.archive is not None:
            self.run_store.save_stats(self.stats.to_record(step=self.step_count, archive=self.archive, stage=self.stage))
        if self.config.errors.fail_fast or self._consecutive_failures >= self.config.errors.max_consecutive_failures:
            raise exc

    def _edit_output_instructions(self, *, initialization: bool) -> str:
        if self.config.output_format_instructions is not None:
            return self.config.output_format_instructions
        if self.config.output_format == "full":
            return (
                "Return each candidate as a complete fenced code block. "
                "Do not use SEARCH/REPLACE diffs. "
                "Each fenced block is one complete candidate program."
            )
        if initialization:
            return (
                "Return each candidate as a complete fenced code block. "
                "Do not use SEARCH/REPLACE diffs. "
                "Write the full program inside the fenced block."
            )
        return (
            "Return one fenced ```diff block containing SEARCH/REPLACE edits.\n"
            "Each SEARCH block must contain a short unique excerpt from the parent — just enough to locate the section to change. Never copy the entire file into SEARCH.\n"
            "You may target a single function, a few lines, or any subsection of the file.\n"
            "Format:\n"
            "```diff\n"
            "<<<<<<< SEARCH\n"
            "exact lines from the parent to find and replace\n"
            "=======\n"
            "new lines to replace them with\n"
            ">>>>>>> REPLACE\n"
            "```\n"
            "Use one edit block per changed section. The SEARCH block is a locator — keep it concise and unique."
        )

    def _build_parse_repair_prompt(self, *, original_prompt: str, broken_response: str, parse_error: Exception, base_code: str = "") -> str:
        extra_guidance = (
            "\n\nRepair guidance:\n"
            "- Start the response immediately with ```diff and end it with ```.\n"
            "- Return only a fenced SEARCH/REPLACE diff response, with no prose before or after it.\n"
            "- Do not use <details>, <summary>, HTML, XML, headings, or commentary.\n"
            "- One fenced candidate may contain multiple SEARCH/REPLACE blocks; use as many sequential edits as needed.\n"
            "- The SEARCH content must match the current parent code exactly.\n"
            "- If you are rewriting the whole file, copy the exact parent file into SEARCH and place the full new file in REPLACE.\n"
            "- Never place the desired new code in SEARCH.\n"
            "- Never use placeholder SEARCH text such as 'old code region to find in the parent program'.\n"
        )
        parse_error_text = str(parse_error)
        if "search text must match exactly once" in parse_error_text:
            extra_guidance += (
                "- The previous SEARCH block did not exactly match the parent program.\n"
                "- For a whole-file rewrite, use the parent program exactly as shown in the target candidate block.\n"
                "- If exact matching is too hard, rewrite the whole file using the full current parent file in SEARCH.\n"
            )
            if "initialization_target" in original_prompt:
                extra_guidance += (
                    "- This is an initialization step with an empty parent file.\n"
                    "- The SEARCH section must be completely empty: place SEARCH immediately above =======.\n"
                    "- Do not copy the signature example or any placeholder code into SEARCH during initialization.\n"
                )
        extra_guidance += (
            "\nExact valid repair examples:\n"
            "Initialization with empty parent:\n"
            "```diff\n"
            "<<<<<<< SEARCH\n"
            "=======\n"
            "import numpy as np\n\n"
            "def detect_bounces(flow_clip: np.ndarray) -> list[dict[str, float]]:\n"
            "    return []\n"
            ">>>>>>> REPLACE\n"
            "```\n\n"
            "Targeted exact-match edit:\n"
            "```diff\n"
            "<<<<<<< SEARCH\n"
            "return x + 1\n"
            "=======\n"
            "return x + 7\n"
            ">>>>>>> REPLACE\n"
            "```"
        )
        parent_block = (
            f"\n\nCurrent parent code (your SEARCH blocks must match this exactly):\n"
            f"```\n{base_code}\n```"
            if base_code else ""
        )
        return (
            f"{original_prompt}\n\n"
            "The previous response could not be parsed or applied. "
            "Fix it and return only a corrected response that uses the required SEARCH/REPLACE edit format.\n\n"
            f"Parser error:\n{parse_error}{extra_guidance}"
            f"{parent_block}\n\n"
            f"Broken response:\n{broken_response}"
        )

    def _log_failed_llm_call(self, *, current_step: int, executed: ExecutedStep) -> None:
        if executed.llm_result is None:
            return
        llm_call_id = f"llm_failed_{current_step:06d}_{uuid4().hex[:8]}"
        self.run_store.log_llm_call(
            step=current_step,
            parent_ids=[parent.id for parent in executed.prepared.parents],
            llm_call_id=llm_call_id,
            prompt=executed.llm_result.prompt,
            response=executed.llm_result.response_text,
            reasoning=getattr(executed.llm_result, "reasoning_text", None),
            metadata={
                "child_ids": [],
                "model": executed.llm_result.model,
                "status": executed.llm_result.status,
                "latency_seconds": executed.llm_result.latency_seconds,
                "started_at": executed.llm_result.started_at,
                "finished_at": executed.llm_result.finished_at,
                "phase": "random_init" if executed.prepared.in_random_init else "evolution",
                "emitter_name": executed.prepared.emitter_name,
                "thinking_mode": executed.prepared.thinking_mode,
                "request_payload": executed.llm_result.request_payload,
                "response_payload": executed.llm_result.response_payload,
                "failure_type": type(executed.error).__name__ if executed.error is not None else None,
                "failure_error": str(executed.error) if executed.error is not None else None,
            },
        )

    def _collect_ancestors(self, candidate: Candidate, count: int) -> list[Candidate]:
        ancestors: list[Candidate] = []
        seen_ids: set[str] = {candidate.id}
        current = candidate
        while len(ancestors) < count and current.parent_ids:
            parent_id = current.parent_ids[0]
            if parent_id in seen_ids:
                break
            parent = self.candidate_registry.get(parent_id)
            if parent is None:
                break
            ancestors.append(parent)
            seen_ids.add(parent.id)
            current = parent
        return ancestors

    def _sample_inspirations(self, *, target: Candidate, ancestors: list[Candidate], count: int) -> list[Candidate]:
        if self.archive is None or count <= 0:
            return []
        excluded_ids = {target.id, *[candidate.id for candidate in ancestors]}
        local_pool = [candidate for candidate in self.archive.all_elites() if candidate.id not in excluded_ids]
        foreign_pool = (
            [c for c in self.cross_island_elite_provider(self.config.island_id) if c.id not in excluded_ids]
            if self.cross_island_elite_provider is not None
            else []
        )
        inspirations: list[Candidate] = []
        seen_ids = set(excluded_ids)
        for _ in range(count):
            use_foreign = bool(foreign_pool) and self._rng.random() < self.config.cross_island_inspiration_probability
            pool = foreign_pool if use_foreign else local_pool
            if not pool:
                pool = foreign_pool if pool is local_pool else local_pool
            if not pool:
                break
            # Fitness-weighted sampling using top-20 candidates for the distribution.
            # Rank pool by fitness, build probs from top 20, then sample the full pool.
            TOP_N = 20
            indexed = sorted(enumerate(pool), key=lambda x: x[1].primary_fitness or 0.0, reverse=True)
            top_indices = [i for i, _ in indexed[:TOP_N]]
            top_fits = [max(0.0, pool[i].primary_fitness or 0.0) for i in top_indices]
            total = sum(top_fits)
            if total > 0:
                probs = [f / total for f in top_fits]
                chosen_pos = int(self._rng.choice(len(top_indices), p=probs))
                choice_index = top_indices[chosen_pos]
            else:
                choice_index = int(self._rng.integers(0, len(pool)))
            candidate = pool.pop(choice_index)
            if candidate.id in seen_ids:
                continue
            seen_ids.add(candidate.id)
            inspirations.append(candidate)
            local_pool = [entry for entry in local_pool if entry.id != candidate.id]
            foreign_pool = [entry for entry in foreign_pool if entry.id != candidate.id]
        return inspirations
