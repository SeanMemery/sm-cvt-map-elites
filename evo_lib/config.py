from __future__ import annotations

import dataclasses
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Literal

from evo_lib.errors import ConfigurationError


BaseSamplingStrategy = Literal["uniform", "fitness_weighted", "best", "curiosity_weighted", "emitter_curiosity_weighted"]
SamplingStrategy = Literal["uniform", "fitness_weighted", "best", "curiosity_weighted", "emitter_curiosity_weighted", "weighted"]
ThinkingMode = Literal["default", "enabled", "disabled"]
GenerationMethod = Literal["standard", "llm_emitters", "llm_emitters_emitter_curiosity"]
OutputFormat = Literal["diff", "full"]
EmitterSelectionStrategy = Literal["fixed", "average_curiosity", "plateau_scheduler"]
ResumeMode = Literal["continue", "fork"]


@dataclass
class LLMConfig:
    api_base_url: str
    api_key: str
    model: str | None = None
    llm_pool_path: str | None = None
    thinking_mode: ThinkingMode = "default"
    timeout_seconds: int = 60
    temperature: float = 0.8
    max_tokens: int = 4096
    n: int = 1
    max_retries: int = 15
    retry_backoff_seconds: float = 180.0  # Fixed delay between retry attempts
    connection_recovery_timeout_seconds: float = 21600.0  # 6 hours total recovery window

    def __post_init__(self) -> None:
        if not self.api_base_url.strip():
            raise ConfigurationError("LLMConfig.api_base_url must be non-empty")
        if not self.api_key.strip():
            raise ConfigurationError("LLMConfig.api_key must be non-empty")
        if self.llm_pool_path is not None and not self.llm_pool_path.strip():
            raise ConfigurationError("LLMConfig.llm_pool_path must be non-empty when provided")
        if self.thinking_mode not in {"default", "enabled", "disabled"}:
            raise ConfigurationError("LLMConfig.thinking_mode must be 'default', 'enabled', or 'disabled'")
        if self.timeout_seconds <= 0:
            raise ConfigurationError("LLMConfig.timeout_seconds must be > 0")
        if self.max_tokens <= 0:
            raise ConfigurationError("LLMConfig.max_tokens must be > 0")
        if self.n <= 0:
            raise ConfigurationError("LLMConfig.n must be > 0")
        if self.max_retries < 0:
            raise ConfigurationError("LLMConfig.max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ConfigurationError("LLMConfig.retry_backoff_seconds must be >= 0")
        if self.connection_recovery_timeout_seconds < 0:
            raise ConfigurationError("LLMConfig.connection_recovery_timeout_seconds must be >= 0")


@dataclass
class NormalizerConfig:
    dim: int
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    history_size: int = 10000
    clip: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ConfigurationError("NormalizerConfig.dim must be > 0")
        if not 0 <= self.lower_quantile < self.upper_quantile <= 1:
            raise ConfigurationError("NormalizerConfig quantiles must satisfy 0 <= lower < upper <= 1")
        if self.history_size <= 0:
            raise ConfigurationError("NormalizerConfig.history_size must be > 0")


@dataclass
class ParsingConfig:
    allow_plaintext_fallback: bool = False
    max_candidates_per_response: int = 1  # always 1 — not user-configurable
    parse_repair_enabled: bool = True
    parse_repair_max_attempts: int = 3
    raw_response: bool = False  # if True, use the entire LLM response as candidate code

    def __post_init__(self) -> None:
        self.max_candidates_per_response = 1  # enforce — always 1


@dataclass
class SecondaryEvalConfig:
    """Config for staged secondary evaluation.

    At the end of each stage the engine evaluates the secondary fitness function
    on a P1 fraction of archive elites (by primary fitness), then seeds the next
    stage from the top P2 fraction of those evaluated candidates (by secondary
    fitness).
    """

    stages: int = 5
    p1: float = 0.5
    p2: float = 0.2
    min_seeds: int = 4
    p2_threshold: int = 8

    def __post_init__(self) -> None:
        if self.stages < 1:
            raise ConfigurationError("SecondaryEvalConfig.stages must be >= 1")
        if not 0.0 <= self.p1 <= 1.0:
            raise ConfigurationError("SecondaryEvalConfig.p1 must be in [0, 1]")
        if not 0.0 <= self.p2 <= 1.0:
            raise ConfigurationError("SecondaryEvalConfig.p2 must be in [0, 1]")
        if self.min_seeds < 0:
            raise ConfigurationError("SecondaryEvalConfig.min_seeds must be >= 0")
        if self.p2_threshold < 0:
            raise ConfigurationError("SecondaryEvalConfig.p2_threshold must be >= 0")


@dataclass
class StorageConfig:
    save_enabled: bool = True
    run_id: str | None = None
    checkpoint_interval: int = 500
    archive_snapshot_interval: int = 500
    save_llm_calls: bool = True
    save_candidates: bool = True
    save_events: bool = True
    overwrite: bool = False

    def __post_init__(self) -> None:
        if self.checkpoint_interval <= 0:
            raise ConfigurationError("StorageConfig.checkpoint_interval must be > 0")
        if self.archive_snapshot_interval <= 0:
            raise ConfigurationError("StorageConfig.archive_snapshot_interval must be > 0")


@dataclass
class ErrorConfig:
    fail_fast: bool = False
    max_consecutive_failures: int = 20

    def __post_init__(self) -> None:
        if self.max_consecutive_failures <= 0:
            raise ConfigurationError("ErrorConfig.max_consecutive_failures must be > 0")


@dataclass
class LLMEndpointConfig:
    api_base_url: str
    models: list[str]
    api_key: str | None = None

    def __post_init__(self) -> None:
        if not self.api_base_url.strip():
            raise ConfigurationError("LLMEndpointConfig.api_base_url must be non-empty")
        if not self.models:
            raise ConfigurationError("LLMEndpointConfig.models must be non-empty")
        if any(not model.strip() for model in self.models):
            raise ConfigurationError("LLMEndpointConfig.models entries must be non-empty")
        if self.api_key is not None and not self.api_key.strip():
            raise ConfigurationError("LLMEndpointConfig.api_key must be non-empty when provided")


@dataclass
class LLMPoolConfig:
    endpoints: list[LLMEndpointConfig]

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ConfigurationError("LLMPoolConfig.endpoints must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LLMPoolConfig":
        endpoints = [LLMEndpointConfig(**entry) for entry in data.get("endpoints", [])]
        return cls(endpoints=endpoints)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "LLMPoolConfig":
        return cls.from_dict(_load_json_file(path))


@dataclass
class SamplingWeightConfig:
    strategy: BaseSamplingStrategy
    weight: float

    def __post_init__(self) -> None:
        if self.strategy not in {"uniform", "fitness_weighted", "best", "curiosity_weighted", "emitter_curiosity_weighted"}:
            raise ConfigurationError(
                "SamplingWeightConfig.strategy must be 'uniform', 'fitness_weighted', 'curiosity_weighted', or 'emitter_curiosity_weighted'"
            )
        if self.weight < 0:
            raise ConfigurationError("SamplingWeightConfig.weight must be >= 0")


@dataclass
class SamplingSchedulePhaseConfig:
    weights: list[SamplingWeightConfig]
    until_step: int | None = None

    def __post_init__(self) -> None:
        if not self.weights:
            raise ConfigurationError("SamplingSchedulePhaseConfig.weights must be non-empty")
        if self.until_step is not None and self.until_step <= 0:
            raise ConfigurationError("SamplingSchedulePhaseConfig.until_step must be > 0 when provided")


@dataclass
class EmitterConfig:
    name: str
    extra_instructions: str = ""
    llm: LLMConfig | None = None
    llm_pool: "list[LLMConfig] | None" = None
    llm_pool_weights: "list[float] | None" = None
    thinking_mode: ThinkingMode = "default"
    selection_weight: float = 1.0
    ancestor_count: int | None = None
    inspiration_elite_count: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ConfigurationError("EmitterConfig.name must be non-empty")
        if self.thinking_mode not in {"default", "enabled", "disabled"}:
            raise ConfigurationError("EmitterConfig.thinking_mode must be 'default', 'enabled', or 'disabled'")
        if self.selection_weight < 0:
            raise ConfigurationError("EmitterConfig.selection_weight must be >= 0")
        if self.ancestor_count is not None and self.ancestor_count < 0:
            raise ConfigurationError("EmitterConfig.ancestor_count must be >= 0 when provided")
        if self.inspiration_elite_count is not None and self.inspiration_elite_count < 0:
            raise ConfigurationError("EmitterConfig.inspiration_elite_count must be >= 0 when provided")
        if self.llm_pool is not None and len(self.llm_pool) == 0:
            raise ConfigurationError("EmitterConfig.llm_pool must be non-empty when provided")
        if self.llm_pool_weights is not None and self.llm_pool is not None:
            if len(self.llm_pool_weights) != len(self.llm_pool):
                raise ConfigurationError("EmitterConfig.llm_pool_weights must have the same length as llm_pool")


@dataclass
class PlateauEmitterSchedulerConfig:
    target_emitter_name: str
    patience_steps: int = 100
    target_weight_growth: float = 1.5
    non_target_weight_decay: float = 0.85
    max_target_weight_scale: float | None = None
    min_non_target_weight_scale: float = 0.1

    def __post_init__(self) -> None:
        if not self.target_emitter_name.strip():
            raise ConfigurationError("PlateauEmitterSchedulerConfig.target_emitter_name must be non-empty")
        if self.patience_steps <= 0:
            raise ConfigurationError("PlateauEmitterSchedulerConfig.patience_steps must be > 0")
        if self.target_weight_growth <= 0:
            raise ConfigurationError("PlateauEmitterSchedulerConfig.target_weight_growth must be > 0")
        if not 0 < self.non_target_weight_decay <= 1:
            raise ConfigurationError("PlateauEmitterSchedulerConfig.non_target_weight_decay must be within (0, 1]")
        if self.max_target_weight_scale is not None and self.max_target_weight_scale <= 0:
            raise ConfigurationError("PlateauEmitterSchedulerConfig.max_target_weight_scale must be > 0 when provided")
        if self.min_non_target_weight_scale <= 0:
            raise ConfigurationError("PlateauEmitterSchedulerConfig.min_non_target_weight_scale must be > 0")


_EXPLOIT_INSTRUCTIONS = (
    "Think deeply about the provided candidate programs. "
    "Understand what each one does, identify weaknesses or missed opportunities, "
    "and produce an improved variant that is more correct, efficient, or robust. "
    "Prefer careful refinement over radical changes."
)

_EXPLORE_INSTRUCTIONS = (
    "Change tactic entirely. The current candidates share a common approach — "
    "try a fundamentally different strategy, algorithm, or structure. "
    "Diversity matters more than incremental improvement here."
)


def _default_emitters(base_llm: LLMConfig) -> list[EmitterConfig]:
    """Build the default exploit/explore emitter pair from a base LLMConfig."""
    # Both emitters allow thinking — reasoning improves candidate quality for
    # exploit (refinement) and explore (novel-strategy) generation alike.
    exploit_llm = dataclasses.replace(base_llm, temperature=0.3, thinking_mode="enabled")
    explore_llm = dataclasses.replace(base_llm, temperature=0.9, thinking_mode="enabled")
    return [
        EmitterConfig(
            name="exploit",
            extra_instructions=_EXPLOIT_INSTRUCTIONS,
            llm=exploit_llm,
            thinking_mode="enabled",
            selection_weight=1.0,
        ),
        EmitterConfig(
            name="explore",
            extra_instructions=_EXPLORE_INSTRUCTIONS,
            llm=explore_llm,
            thinking_mode="enabled",
            selection_weight=1.0,
        ),
    ]


@dataclass
class SamplingConfig:
    """Configurable cell and emitter sampling policy for LLM MAP-Elites."""

    cell_method: str = "ucb"                   # "uniform" | "ucb" | "mean_diff"
    cell_epsilon_uniform: float = 0.10
    cell_ucb_c: float = 0.4
    cell_mean_diff_temperature: float = 0.4
    cell_reward_ema_beta: float = 0.20
    emitter_method: str = "single_curiosity"   # "fixed" | "single_curiosity" | "ucb" | "thompson" | "momentum"
    emitter_p_exploit: float = 0.50
    momentum_ema_beta: float = 0.05      # smoothing factor for insertion-rate EMA
    momentum_exploit_lo: float = 0.20   # p_exploit when insertion rate → 0 (stagnating)
    momentum_exploit_hi: float = 0.80   # p_exploit when insertion rate → 1 (hot archive)
    emitter_ucb_c: float = 1.0
    emitter_cell_warmup_trials: int = 4
    emitter_min_p_exploit: float = 0.15
    emitter_max_p_exploit: float = 0.85
    emitter_cell_lr: float = 0.25
    emitter_global_lr: float = 0.05
    emitter_logit_clip_lo: float = -2.0
    emitter_logit_clip_hi: float = 2.0

    def __post_init__(self) -> None:
        if self.cell_method not in {"uniform", "ucb", "mean_diff"}:
            raise ConfigurationError("SamplingConfig.cell_method must be 'uniform', 'ucb', or 'mean_diff'")
        if self.emitter_method not in {"fixed", "single_curiosity", "ucb", "thompson", "softmax", "momentum"}:
            raise ConfigurationError(
                "SamplingConfig.emitter_method must be 'fixed', 'single_curiosity', 'ucb', 'thompson', 'softmax', or 'momentum'"
            )


@dataclass
class EvolutionConfig:
    run_name: str
    llm: LLMConfig
    generation_method: GenerationMethod = "llm_emitters_emitter_curiosity"
    emitters: list[EmitterConfig] | None = None
    emitter_selection_strategy: EmitterSelectionStrategy | None = None
    emitter_plateau_scheduler: PlateauEmitterSchedulerConfig | None = None
    project_id: str = "default"
    job_id: str | None = None
    island_id: str | None = None
    cross_island_inspiration_probability: float = 0.05
    output_dir: str = "runs"
    descriptor_dim: int = 5
    num_centroids: int = 50
    cvt_samples: int = 5000
    elites_per_cell: int = 1
    remap_interval: int = 1
    initial_random_steps: int = 4
    grid_distance_threshold: float = 0.05
    parallel_workers: int = 4
    fitness_batch_size: int = 1  # if >1, batch_primary_fitness must be supplied; candidates are evaluated in groups
    parents_per_mutation: int = 2
    max_steps: int = 10000
    random_seed: int = 0
    user_prompt: str = ""
    extra_instructions: str = ""
    output_format: OutputFormat = "diff"
    output_format_instructions: str | None = None
    descriptor_labels: list[str] | None = None
    ancestor_count: int = 2
    inspiration_elite_count: int = 2
    primary_metric_label: str | None = None
    primary_validation_metric_label: str | None = None
    secondary_metric_label: str | None = None
    secondary_validation_metric_label: str | None = None
    task_config: dict[str, Any] = field(default_factory=dict)
    error_feedback_enabled: bool = True   # inject common runtime errors into prompt
    error_feedback_count: int = 5         # how many top errors to inject
    curiosity_default: float = 1.0
    curiosity_reward: float = 1.5
    curiosity_penalty: float = 0.5
    curiosity_temperature: float = 0.7  # softmax temperature for curiosity distributions (< 1 = sharper)
    grid_selection_strategy: SamplingStrategy = "emitter_curiosity_weighted"
    elite_selection_strategy: SamplingStrategy = "best"
    grid_sampling_weights: list[SamplingWeightConfig] | None = None
    elite_sampling_weights: list[SamplingWeightConfig] | None = None
    grid_sampling_schedule: list[SamplingSchedulePhaseConfig] | None = None
    elite_sampling_schedule: list[SamplingSchedulePhaseConfig] | None = None
    resume_mode: ResumeMode = "continue"
    normalizer: NormalizerConfig | None = None
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    secondary_eval: SecondaryEvalConfig = field(default_factory=SecondaryEvalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    code_language: str = "python"
    sampling: SamplingConfig | None = None

    def __post_init__(self) -> None:
        if not self.run_name.strip():
            raise ConfigurationError("EvolutionConfig.run_name must be non-empty")
        if not self.project_id.strip():
            raise ConfigurationError("EvolutionConfig.project_id must be non-empty")
        if self.generation_method not in {"standard", "llm_emitters", "llm_emitters_emitter_curiosity"}:
            raise ConfigurationError(
                "EvolutionConfig.generation_method must be 'standard', 'llm_emitters', or 'llm_emitters_emitter_curiosity'"
        )
        if self.job_id is not None and not self.job_id.strip():
            raise ConfigurationError("EvolutionConfig.job_id must be non-empty when provided")
        if self.island_id is not None and not self.island_id.strip():
            raise ConfigurationError("EvolutionConfig.island_id must be non-empty when provided")
        if not 0 <= self.cross_island_inspiration_probability <= 1:
            raise ConfigurationError("EvolutionConfig.cross_island_inspiration_probability must be within [0, 1]")
        if self.descriptor_dim <= 0:
            raise ConfigurationError("EvolutionConfig.descriptor_dim must be > 0")
        if self.num_centroids <= 0:
            raise ConfigurationError("EvolutionConfig.num_centroids must be > 0")
        if self.cvt_samples < self.num_centroids:
            raise ConfigurationError("EvolutionConfig.cvt_samples must be >= num_centroids")
        if self.elites_per_cell <= 0:
            raise ConfigurationError("EvolutionConfig.elites_per_cell must be > 0")
        if self.remap_interval <= 0:
            raise ConfigurationError("EvolutionConfig.remap_interval must be > 0")
        if self.initial_random_steps < 0:
            raise ConfigurationError("EvolutionConfig.initial_random_steps must be >= 0")
        if self.grid_distance_threshold <= 0:
            raise ConfigurationError("EvolutionConfig.grid_distance_threshold must be > 0")
        if self.parallel_workers <= 0:
            raise ConfigurationError("EvolutionConfig.parallel_workers must be > 0")
        if self.fitness_batch_size < 1:
            raise ConfigurationError("EvolutionConfig.fitness_batch_size must be >= 1")
        if self.parents_per_mutation <= 0:
            raise ConfigurationError("EvolutionConfig.parents_per_mutation must be > 0")
        if self.ancestor_count < 0:
            raise ConfigurationError("EvolutionConfig.ancestor_count must be >= 0")
        if self.inspiration_elite_count < 0:
            raise ConfigurationError("EvolutionConfig.inspiration_elite_count must be >= 0")
        if self.max_steps <= 0:
            raise ConfigurationError("EvolutionConfig.max_steps must be > 0")
        if not self.user_prompt.strip():
            raise ConfigurationError("EvolutionConfig.user_prompt must be non-empty")
        if self.output_format not in {"diff", "full"}:
            raise ConfigurationError("EvolutionConfig.output_format must be 'diff' or 'full'")
        if self.normalizer is None:
            self.normalizer = NormalizerConfig(dim=self.descriptor_dim)
        if self.normalizer.dim != self.descriptor_dim:
            raise ConfigurationError("NormalizerConfig.dim must equal descriptor_dim")
        if self.descriptor_labels is None:
            self.descriptor_labels = [f"descriptor_{index}" for index in range(self.descriptor_dim)]
        if len(self.descriptor_labels) != self.descriptor_dim:
            raise ConfigurationError("descriptor_labels length must equal descriptor_dim")
        if self.primary_metric_label is not None and not self.primary_metric_label.strip():
            raise ConfigurationError("primary_metric_label must be non-empty when provided")
        if self.primary_validation_metric_label is not None and not self.primary_validation_metric_label.strip():
            raise ConfigurationError("primary_validation_metric_label must be non-empty when provided")
        if self.secondary_metric_label is not None and not self.secondary_metric_label.strip():
            raise ConfigurationError("secondary_metric_label must be non-empty when provided")
        if self.secondary_validation_metric_label is not None and not self.secondary_validation_metric_label.strip():
            raise ConfigurationError("secondary_validation_metric_label must be non-empty when provided")
        if self.primary_metric_label is None:
            self.primary_metric_label = "Primary"
        if self.primary_validation_metric_label is None:
            self.primary_validation_metric_label = f"{self.primary_metric_label} Validation"
        if self.secondary_metric_label is None:
            self.secondary_metric_label = "Secondary"
        if self.secondary_validation_metric_label is None:
            self.secondary_validation_metric_label = f"{self.secondary_metric_label} Validation"
        if self.emitters is not None and not self.emitters:
            raise ConfigurationError("EvolutionConfig.emitters must be non-empty when provided")
        if self.emitters is None and self.generation_method in {"llm_emitters", "llm_emitters_emitter_curiosity"}:
            self.emitters = _default_emitters(self.llm)
        if self.emitter_selection_strategy is None:
            if self.generation_method == "llm_emitters_emitter_curiosity":
                self.emitter_selection_strategy = "average_curiosity"
            else:
                self.emitter_selection_strategy = "fixed"
        if self.emitter_selection_strategy not in {"fixed", "average_curiosity", "plateau_scheduler"}:
            raise ConfigurationError("emitter_selection_strategy must be 'fixed', 'average_curiosity', or 'plateau_scheduler'")
        if self.generation_method in {"llm_emitters", "llm_emitters_emitter_curiosity"}:
            if not self.emitters:
                raise ConfigurationError("EvolutionConfig.emitters must be provided when generation_method='llm_emitters'")
            if self.emitter_selection_strategy == "fixed" and sum(emitter.selection_weight for emitter in self.emitters) <= 0:
                raise ConfigurationError("EvolutionConfig.emitters must have positive total selection_weight")
            if self.emitter_selection_strategy == "plateau_scheduler":
                if self.emitter_plateau_scheduler is None:
                    raise ConfigurationError("emitter_plateau_scheduler must be provided when emitter_selection_strategy='plateau_scheduler'")
                emitter_names = {emitter.name for emitter in self.emitters}
                if self.emitter_plateau_scheduler.target_emitter_name not in emitter_names:
                    raise ConfigurationError("emitter_plateau_scheduler.target_emitter_name must match one of the configured emitters")
        elif self.emitter_plateau_scheduler is not None:
            raise ConfigurationError("emitter_plateau_scheduler requires an emitter-based generation_method")
        if self.grid_selection_strategy not in {"uniform", "fitness_weighted", "best", "curiosity_weighted", "emitter_curiosity_weighted", "weighted"}:
            raise ConfigurationError(
                "grid_selection_strategy must be 'uniform', 'fitness_weighted', 'best', 'curiosity_weighted', 'emitter_curiosity_weighted', or 'weighted'"
            )
        self.elite_selection_strategy = "best"  # always use best — not configurable
        self._validate_sampling_config(
            strategy=self.grid_selection_strategy,
            weights=self.grid_sampling_weights,
            schedule=self.grid_sampling_schedule,
            label="grid",
        )
        self._validate_sampling_config(
            strategy=self.elite_selection_strategy,
            weights=self.elite_sampling_weights,
            schedule=self.elite_sampling_schedule,
            label="elite",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionConfig":
        emitter_entries = data.get("emitters") or []
        return cls(
            run_name=data["run_name"],
            llm=LLMConfig(**data["llm"]),
            generation_method=data.get("generation_method", "llm_emitters_emitter_curiosity"),
            emitters=[EmitterConfig(
                name=entry["name"],
                extra_instructions=entry.get("extra_instructions", ""),
                llm=LLMConfig(**entry["llm"]) if entry.get("llm") is not None else None,
                thinking_mode=entry.get("thinking_mode", "default"),
                selection_weight=float(entry.get("selection_weight", 1.0)),
                ancestor_count=(
                    int(entry["ancestor_count"])
                    if entry.get("ancestor_count") is not None
                    else None
                ),
                inspiration_elite_count=(
                    int(entry["inspiration_elite_count"])
                    if entry.get("inspiration_elite_count") is not None
                    else None
                ),
            ) for entry in emitter_entries] or None,
            emitter_selection_strategy=data.get("emitter_selection_strategy"),
            emitter_plateau_scheduler=(
                PlateauEmitterSchedulerConfig(**data["emitter_plateau_scheduler"])
                if data.get("emitter_plateau_scheduler") is not None
                else None
            ),
            project_id=data.get("project_id", "default"),
            job_id=data.get("job_id"),
            island_id=data.get("island_id"),
            cross_island_inspiration_probability=float(data.get("cross_island_inspiration_probability", 0.05)),
            output_dir=data.get("output_dir", "runs"),
            descriptor_dim=data.get("descriptor_dim", 5),
            num_centroids=data.get("num_centroids", 50),
            cvt_samples=data.get("cvt_samples", 5000),
            elites_per_cell=data.get("elites_per_cell", 1),
            remap_interval=data.get("remap_interval", 1),
            initial_random_steps=int(data.get("initial_random_steps", 4)),
            grid_distance_threshold=data.get("grid_distance_threshold", 0.05),
            parallel_workers=data.get("parallel_workers", 4),
            parents_per_mutation=data.get("parents_per_mutation", 2),
            ancestor_count=int(data.get("ancestor_count", 2)),
            inspiration_elite_count=int(data.get("inspiration_elite_count", 2)),
            max_steps=data.get("max_steps", 10000),
            random_seed=data.get("random_seed", 0),
            user_prompt=data.get("user_prompt", ""),
            extra_instructions=data.get("extra_instructions", ""),
            output_format=data.get("output_format", "diff"),
            output_format_instructions=data.get("output_format_instructions"),
            descriptor_labels=list(data.get("descriptor_labels", [])) or None,
            primary_metric_label=data.get("primary_metric_label"),
            primary_validation_metric_label=data.get("primary_validation_metric_label"),
            secondary_metric_label=data.get("secondary_metric_label"),
            secondary_validation_metric_label=data.get("secondary_validation_metric_label"),
            task_config=dict(data.get("task_config", {})),
            curiosity_default=float(data.get("curiosity_default", 1.0)),
            curiosity_reward=float(data.get("curiosity_reward", 1.5)),
            curiosity_penalty=float(data.get("curiosity_penalty", 0.5)),
            grid_selection_strategy=data.get("grid_selection_strategy", "uniform"),
            elite_selection_strategy=data.get("elite_selection_strategy", "fitness_weighted"),
            grid_sampling_weights=[
                SamplingWeightConfig(**entry) for entry in data.get("grid_sampling_weights", []) or []
            ] or None,
            elite_sampling_weights=[
                SamplingWeightConfig(**entry) for entry in data.get("elite_sampling_weights", []) or []
            ] or None,
            grid_sampling_schedule=[
                SamplingSchedulePhaseConfig(
                    until_step=entry.get("until_step"),
                    weights=[SamplingWeightConfig(**weight) for weight in entry["weights"]],
                )
                for entry in data.get("grid_sampling_schedule", []) or []
            ] or None,
            elite_sampling_schedule=[
                SamplingSchedulePhaseConfig(
                    until_step=entry.get("until_step"),
                    weights=[SamplingWeightConfig(**weight) for weight in entry["weights"]],
                )
                for entry in data.get("elite_sampling_schedule", []) or []
            ] or None,
            resume_mode=data.get("resume_mode", "continue"),
            normalizer=NormalizerConfig(**data["normalizer"]),
            parsing=ParsingConfig(**data["parsing"]),
            secondary_eval=SecondaryEvalConfig(
                stages=int(data.get("secondary_eval", {}).get("stages", 5)),
                p1=float(data.get("secondary_eval", {}).get("p1", 0.5)),
                p2=float(data.get("secondary_eval", {}).get("p2", 0.2)),
                min_seeds=int(data.get("secondary_eval", {}).get("min_seeds", 4)),
                p2_threshold=int(data.get("secondary_eval", {}).get("p2_threshold", 8)),
            ),
            storage=StorageConfig(**data["storage"]),
            errors=ErrorConfig(**data["errors"]),
            sampling=(
                SamplingConfig(**data["sampling"])
                if data.get("sampling") is not None
                else None
            ),
        )

    @staticmethod
    def _validate_sampling_config(
        *,
        strategy: SamplingStrategy,
        weights: list[SamplingWeightConfig] | None,
        schedule: list[SamplingSchedulePhaseConfig] | None,
        label: str,
    ) -> None:
        if strategy != "weighted":
            if weights:
                raise ConfigurationError(f"{label}_sampling_weights requires {label}_selection_strategy='weighted'")
            if schedule:
                raise ConfigurationError(f"{label}_sampling_schedule requires {label}_selection_strategy='weighted'")
            return
        if not weights and not schedule:
            raise ConfigurationError(
                f"{label}_selection_strategy='weighted' requires {label}_sampling_weights or {label}_sampling_schedule"
            )
        if weights and sum(entry.weight for entry in weights) <= 0:
            raise ConfigurationError(f"{label}_sampling_weights must have positive total weight")
        if not schedule:
            return
        previous_until_step = 0
        saw_open_ended_phase = False
        for index, phase in enumerate(schedule):
            if sum(entry.weight for entry in phase.weights) <= 0:
                raise ConfigurationError(f"{label}_sampling_schedule phase {index} must have positive total weight")
            if saw_open_ended_phase:
                raise ConfigurationError(f"{label}_sampling_schedule cannot include phases after an open-ended phase")
            if phase.until_step is None:
                saw_open_ended_phase = True
                continue
            if phase.until_step <= previous_until_step:
                raise ConfigurationError(f"{label}_sampling_schedule until_step values must be strictly increasing")
            previous_until_step = phase.until_step


@dataclass
class GlobalJobConfig:
    job_name: str
    output_dir: str
    parallel_workers: int
    migration_probability: float = 0.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        if not self.job_name.strip():
            raise ConfigurationError("GlobalJobConfig.job_name must be non-empty")
        if not self.output_dir.strip():
            raise ConfigurationError("GlobalJobConfig.output_dir must be non-empty")
        if self.parallel_workers <= 0:
            raise ConfigurationError("GlobalJobConfig.parallel_workers must be > 0")
        if not 0 <= self.migration_probability <= 1:
            raise ConfigurationError("GlobalJobConfig.migration_probability must be within [0, 1]")

    def to_dict(self) -> dict:
        return {
            "job_name": self.job_name,
            "output_dir": self.output_dir,
            "parallel_workers": self.parallel_workers,
            "migration_probability": self.migration_probability,
            "random_seed": self.random_seed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GlobalJobConfig":
        return cls(
            job_name=data["job_name"],
            output_dir=data["output_dir"],
            parallel_workers=int(data["parallel_workers"]),
            migration_probability=float(data.get("migration_probability", 0.0)),
            random_seed=int(data.get("random_seed", 0)),
        )

    @classmethod
    def from_json_file(cls, path) -> "GlobalJobConfig":
        import json
        from pathlib import Path
        with open(Path(path)) as f:
            return cls.from_dict(json.load(f))


@dataclass
class IslandHooksConfig:
    module_path: str
    initial_candidates_factory: str
    descriptor_fn: str
    primary_fitness_factory: str
    primary_validation_fitness_factory: str | None = None
    secondary_fitness_factory: str | None = None
    secondary_validation_fitness_factory: str | None = None
    candidate_validator_fn: str | None = None
    candidate_parser_fn: str | None = None
    llm_client_factory: str | None = None

    def __post_init__(self) -> None:
        if not self.module_path.strip():
            raise ConfigurationError("IslandHooksConfig.module_path must be non-empty")
        if not self.initial_candidates_factory.strip():
            raise ConfigurationError("IslandHooksConfig.initial_candidates_factory must be non-empty")
        if not self.descriptor_fn.strip():
            raise ConfigurationError("IslandHooksConfig.descriptor_fn must be non-empty")
        if not self.primary_fitness_factory.strip():
            raise ConfigurationError("IslandHooksConfig.primary_fitness_factory must be non-empty")

    def to_dict(self) -> dict:
        return {
            "module_path": self.module_path,
            "initial_candidates_factory": self.initial_candidates_factory,
            "descriptor_fn": self.descriptor_fn,
            "primary_fitness_factory": self.primary_fitness_factory,
            "primary_validation_fitness_factory": self.primary_validation_fitness_factory,
            "secondary_fitness_factory": self.secondary_fitness_factory,
            "secondary_validation_fitness_factory": self.secondary_validation_fitness_factory,
            "candidate_validator_fn": self.candidate_validator_fn,
            "candidate_parser_fn": self.candidate_parser_fn,
            "llm_client_factory": self.llm_client_factory,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IslandHooksConfig":
        return cls(
            module_path=data["module_path"],
            initial_candidates_factory=data["initial_candidates_factory"],
            descriptor_fn=data["descriptor_fn"],
            primary_fitness_factory=data["primary_fitness_factory"],
            primary_validation_fitness_factory=data.get("primary_validation_fitness_factory"),
            secondary_fitness_factory=data.get("secondary_fitness_factory"),
            secondary_validation_fitness_factory=data.get("secondary_validation_fitness_factory"),
            candidate_validator_fn=data.get("candidate_validator_fn"),
            candidate_parser_fn=data.get("candidate_parser_fn"),
            llm_client_factory=data.get("llm_client_factory"),
        )


@dataclass
class IslandConfig:
    island_id: str
    evolution: "EvolutionConfig"
    hooks: IslandHooksConfig
    source_path: str | None = None

    def __post_init__(self) -> None:
        if not self.island_id.strip():
            raise ConfigurationError("IslandConfig.island_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "island_id": self.island_id,
            "evolution": self.evolution.to_dict(),
            "hooks": self.hooks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict, *, source_path: str | None = None) -> "IslandConfig":
        return cls(
            island_id=data["island_id"],
            evolution=EvolutionConfig.from_dict(data["evolution"]),
            hooks=IslandHooksConfig.from_dict(data["hooks"]),
            source_path=source_path,
        )

    @classmethod
    def from_json_file(cls, path) -> "IslandConfig":
        import json
        from pathlib import Path
        p = Path(path)
        with open(p) as f:
            return cls.from_dict(json.load(f), source_path=str(p))


def _load_json_file(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"Failed to read config file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Failed to parse JSON config file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Config file must contain a JSON object: {path}")
    return payload
