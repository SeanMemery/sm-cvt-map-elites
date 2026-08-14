# PROJECT.md — cvt-map-elites-llm

Implementation reference for LLM agents working on this codebase. Describes the current architecture, data model, sampling system, and storage format.

---

## Overview

A standalone Python library for evolving code using CVT-MAP-Elites archiving combined with LLM-driven mutation. The library is task-agnostic: users supply fitness functions, descriptors, prompts, and seed candidates; the library owns archive management, sampling, LLM calls, parsing, storage, and checkpointing.

**Key invariants:**
- Archive insertion is always based on primary fitness (higher is better, one elite per cell).
- Secondary fitness is evaluated periodically and used for pruning/seeding, never for insertion.
- Cell sampling and emitter sampling are independently configurable bandit policies.
- All run data is persisted append-only; nothing is deleted from disk.

---

## Repository layout

```
evo_lib/
  __init__.py
  engine.py        main EvolutionEngine orchestrator
  config.py        all dataclasses (EvolutionConfig, SamplingConfig, etc.)
  candidate.py     Candidate dataclass
  archive.py       CVTArchive wrapper
  normalizer.py    DescriptorNormalizer
  llm.py           LLMClient, OpenAICompatibleClient
  parsing.py       SEARCH/REPLACE diff parser + full-block parser
  selection.py     parent / elite selection helpers
  sampling.py      UCB + single_curiosity bandit sampling
  secondary.py     secondary evaluation and pruning
  storage.py       RunStore, file persistence
  stats.py         step-level statistics
  errors.py        typed exceptions

viewer/
  backend/main.py  FastAPI app
  backend/run_loader.py
  frontend/src/    React/Vite app

tests/
validation/        local experiments (git-ignored)
```

---

## Core data model

### Candidate

```python
@dataclass
class Candidate:
    id: str
    code: str
    metadata: dict = field(default_factory=dict)

    descriptor_raw: list[float] | None = None
    descriptor_norm: list[float] | None = None

    primary_fitness: float | None = None
    secondary_fitness: float | None = None

    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    created_at_step: int = 0
    cell_id: int | None = None
    is_active: bool = True

    # Sampling stats stored on the candidate at snapshot time
    cell_n_trials: int = 0
    cell_reward_ema: float = 0.0
    cell_logit_exploit: float = 0.0
    cell_emitter_n_trials: int = 0
    cell_emitter_reward_ema: float = 0.0
    cell_emitter_successes: float = 1.0
    cell_emitter_failures: float = 1.0
```

### EvolutionConfig (key fields)

```python
@dataclass
class EvolutionConfig:
    run_name: str
    output_dir: str = "runs"
    descriptor_dim: int = 2
    num_centroids: int = 100
    parents_per_mutation: int = 2
    initial_random_steps: int = 0       # LLM generates from scratch for this many steps
    max_steps: int = 100
    random_seed: int = 42
    user_prompt: str = ""
    output_format: OutputFormat = "diff"   # "diff" | "full"
    code_language: str = "python"
    parallel_workers: int = 1
    llm: LLMConfig = ...
    sampling: SamplingConfig | None = None
    secondary_eval: SecondaryEvalConfig | None = None
    storage: StorageConfig = ...
    normalizer: NormalizerConfig = ...
    parsing: ParsingConfig = ...
    emitters: list[EmitterConfig] = ...

    # Hardcoded — not user-configurable:
    elite_selection_strategy = "best"       # always argmax within cell
    max_candidates_per_response = 1
```

### SamplingConfig

```python
@dataclass
class SamplingConfig:
    # Cell sampling
    cell_method: str = "ucb"             # "ucb" | "uniform"
    cell_ucb_c: float = 1.4
    cell_reward_ema_beta: float = 0.2

    # Emitter sampling
    emitter_method: str = "single_curiosity"  # "single_curiosity" | "fixed" | "ucb" | "thompson"
    emitter_p_exploit: float = 0.5
    emitter_ucb_c: float = 1.0
    emitter_cell_warmup_trials: int = 4
    emitter_min_p_exploit: float = 0.15
    emitter_max_p_exploit: float = 0.85
    emitter_cell_lr: float = 0.25
    emitter_global_lr: float = 0.05
    emitter_logit_clip: tuple[float, float] = (-2.0, 2.0)
```

---

## Sampling system (`evo_lib/sampling.py`)

Two independent decisions per step:

```
1. Cell sampling   — which archive cell provides the parent?
2. Emitter sampling — exploit (refine) or explore (diversify)?
```

### Reward signal

```python
def compute_reward(was_inserted: bool, normalized_improvement: float = 0.0) -> float:
    if was_inserted:
        return 0.5 + 0.5 * normalized_improvement
    return 0.0
```

### Cell sampling

**UCB** (default):
```
UCB_i = reward_ema_i + c * sqrt(log(N+1) / (n_i+1))
```
- `reward_ema_i`: EMA of rewards from sampling cell i
- `n_i`: times cell i has been sampled
- `N`: total trials
- `c = cell_ucb_c` controls exploration

**Uniform**: random choice among occupied cells.

### Emitter sampling

**single_curiosity** (default):

Each cell has a `logit_exploit`. Emitter probability blends a global prior with the cell-local value:

```
w_i = min(1, n_i / warmup_trials)
p_exploit = (1 - w_i) * sigmoid(global_logit) + w_i * sigmoid(cell.logit_exploit)
p_exploit = clamp(p_exploit, min_p_exploit, max_p_exploit)
```

Update after outcome:
- exploit success → logit increases (more exploit)
- exploit failure → logit decreases (more explore)
- explore success → logit decreases (more explore)
- explore failure → logit increases (more exploit)

**fixed**: constant `p_exploit`.
**ucb**: UCB bandit over the two emitters per cell using per-emitter reward EMA.
**thompson**: Beta(successes, failures) sampling per emitter per cell.

### Stats update

```python
def update_cell_and_global_stats(cell_stats, global_stats, emitter, reward, config):
    # Update cell UCB reward EMA
    cell_stats.n_trials += 1
    cell_stats.reward_ema = ema(cell_stats.reward_ema, reward, beta)

    # Update emitter-level stats
    emitter_stats.n_trials += 1
    emitter_stats.reward_ema = ema(...)
    if reward > 0: emitter_stats.successes += reward
    else: emitter_stats.failures += 1.0

    # Update single_curiosity logit
    if method == "single_curiosity":
        delta = signed_delta(emitter, reward)
        cell_stats.logit_exploit += cell_lr * delta
        global_stats.logit_exploit += global_lr * delta
        # clip to logit_clip range
```

---

## Evolution engine flow (`evo_lib/engine.py`)

### Initialisation

```python
engine.initialize(candidates: list[Candidate])
```

- Pass `[]` with `initial_random_steps > 0` for fully random seed generation (LLM writes from scratch).
- Pass seed candidates to start from known-good code.
- Seeds go through the same evaluation/insertion pipeline as generated candidates.

### Step loop

```python
def step():
    # 1. Select cell and emitter
    cell = select_cell(archive, sampling_state, config.sampling)
    emitter = select_emitter(cell, sampling_state, config.sampling)

    # 2. Build prompt (target elite + ancestors + inspirations)
    parents = [cell.elite] + sample_ancestors(cell.elite)
    prompt = build_prompt(config.user_prompt, parents, emitter, ...)

    # 3. LLM call
    response = llm_client.generate(prompt, emitter_llm_config)
    run_store.log_llm_call(...)

    # 4. Parse candidate (SEARCH/REPLACE diff applied to parent code)
    child = parse_candidate(response, parent_code)

    # 5. Validate
    if not candidate_validator(child): log_rejected; continue

    # 6. Evaluate
    child.descriptor_raw = descriptor_fn(child)
    child.primary_fitness = primary_fitness(child)

    # 7. Insert into archive
    inserted = archive.insert(child)
    run_store.save_candidate(child)
    run_store.log_event("candidate_inserted" | "candidate_not_inserted", ...)

    # 8. Update sampling stats
    reward = compute_reward(inserted, normalized_improvement)
    update_cell_and_global_stats(cell.stats, global_stats, emitter, reward, config)

    # 9. Periodic: remap, snapshot, secondary eval, checkpoint
```

### Parallel workers

`parallel_workers > 1` runs multiple LLM+eval chains concurrently using a thread pool. Each worker independently samples a cell/emitter, generates, evaluates, and inserts. Archive insertions are serialised.

---

## Storage format

Each run writes to `{output_dir}/{run_name}_{timestamp}_{short_uuid}/`:

```
config.json                     full serialised EvolutionConfig
metadata.json                   run status, step count, best fitness
candidates.jsonl                append-only; one record per candidate
events.jsonl                    archive events, remaps, errors, checkpoints
llm_calls.jsonl                 prompt, response, latency, status per call
stats.jsonl                     per-step aggregate statistics
centroids.npy                   CVT centroids (fixed for the run lifetime)
archive_snapshots/stepNNNNNN.json
checkpoints/latest.pkl
```

### Archive snapshot schema

```json
{
  "step": 10,
  "centroids": [[0.1, 0.2], ...],
  "normalizer_bounds": {"lower": [...], "upper": [...]},
  "occupied_cells": [
    {
      "cell_id": 3,
      "candidate_id": "cand_000012",
      "primary_fitness": 0.87,
      "secondary_fitness": null,
      "descriptor_raw": [4.0, 12.0],
      "descriptor_norm": [0.4, 0.6],
      "cell_n_trials": 5,
      "cell_reward_ema": 0.4,
      "cell_logit_exploit": 0.8
    }
  ]
}
```

### Event types

| type | when |
|---|---|
| `candidate_inserted` | child improves its cell |
| `candidate_not_inserted` | child evaluated but didn't beat incumbent |
| `candidate_rejected` | validator returned False |
| `parse_failure` | LLM response couldn't be parsed |
| `llm_error` | LLM call failed |
| `remap` | archive remapped to updated normalizer bounds |
| `secondary_eval` | batch of elites re-evaluated with secondary fitness |
| `stage_start` / `stage_end` | secondary eval stage boundary |
| `checkpoint_saved` | checkpoint written to disk |
| `primary_fitness_failure` | fitness function raised an exception |

---

## Web viewer API

Backend: FastAPI at port 8000. Frontend: React/Vite served as static files.

Key endpoints:
- `POST /api/runs/import` — import a run directory (or parent dir for bulk)
- `GET /api/runs/{id}/summary` — config, descriptor labels, metric labels
- `GET /api/runs/{id}/timeseries` — stats, events, timeline
- `GET /api/runs/{id}/snapshot/{step}` — archive snapshot at a given step
- `GET /api/runs/{id}/snapshot/steps` — list of available snapshot steps
- `GET /api/runs/{id}/candidates` — paginated candidate list
- `GET /api/runs/{id}/timing` — per-step LLM + eval timing

The viewer builds timeline events from `events.jsonl` and enriches them with timestamps from `llm_calls.jsonl` using a `step → timestamp` lookup so all events display a time.

---

## Descriptor normalisation

Raw descriptors live in arbitrary user-defined space. CVT centroids live in `[0,1]^D` normalised space. A `DescriptorNormalizer` maintains sliding quantile bounds (default 1st–99th percentile) over the descriptor history and normalises each new descriptor before archive insertion.

Bounds are updated every step. When bounds shift significantly, a remap is triggered: all current elites are re-normalised and re-inserted into their new nearest centroids.

---

## Design rules

- Fitness functions always return higher-is-better floats (or `(float, stats_dict)`).
- Archive insertion criterion is always primary fitness — secondary fitness never affects insertion.
- `elite_selection_strategy` is hardcoded to `"best"` (argmax within cell).
- `max_candidates_per_response` is hardcoded to 1.
- SEARCH/REPLACE diff format is used for mutation; full-file output only during random init steps.
- The library is language-agnostic — `code_language` controls prompt hints only.
- User owns: descriptors, fitness, validation, prompt content, emitter definitions.
- Library owns: archive mechanics, normalisation, sampling bandits, LLM calls, parsing, storage, checkpointing.
