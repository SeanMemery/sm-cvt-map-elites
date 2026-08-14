"""Configurable cell and emitter sampling for LLM MAP-Elites.

Implements UCB cell sampling and four emitter selection methods:
  - fixed: constant exploit probability
  - single_curiosity: one logit per cell blended with a global logit
  - ucb: per-emitter UCB bandit within a cell
  - thompson: per-emitter Thompson sampling within a cell
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from evo_lib.archive import CVTArchive
    from evo_lib.candidate import Candidate
    from evo_lib.config import SamplingConfig


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_reward(was_inserted: bool, normalized_improvement: float = 0.0) -> float:
    """Return a reward in [0, 1] for a sampling trial.

    Args:
        was_inserted: True when the offspring entered the archive.
        normalized_improvement: Relative fitness gain in [0, 1] when the
            offspring improved an existing cell (unused for new cells).
    """
    if was_inserted:
        return 0.5 + 0.5 * float(normalized_improvement)
    return 0.0


# ---------------------------------------------------------------------------
# Cell selection
# ---------------------------------------------------------------------------

def select_cell_uniform(occupied_cell_ids: list[int], rng: np.random.Generator) -> int:
    """Choose uniformly among occupied archive cells."""
    idx = int(rng.integers(0, len(occupied_cell_ids)))
    return occupied_cell_ids[idx]


def select_cell_mean_diff(
    occupied_cell_ids: list[int],
    archive: "CVTArchive",
    config: "SamplingConfig",
    rng: np.random.Generator,
) -> int:
    """Sample cells proportional to how much their fitness exceeds the archive mean.

    Algorithm:
      1. Collect the best primary fitness for each occupied cell.
      2. Compute the mean across all cells.
      3. Compute diff_i = fitness_i - mean  (can be negative).
      4. Shift so the minimum diff becomes 0 by adding abs(min(diffs)).
      5. Apply temperature: weight_i = shifted_i ^ (1 / temperature).
         (When temperature → 0 this converges to argmax; temperature = 1 is linear.)
      6. Sample from the resulting probability distribution.

    Cells that have never been evaluated fall back to uniform.
    """
    fitnesses = []
    for cell_id in occupied_cell_ids:
        elite = archive.get_cell_elite(cell_id)
        f = elite.primary_fitness if elite is not None else None
        fitnesses.append(f)

    valid_mask = [f is not None for f in fitnesses]
    if not any(valid_mask):
        return select_cell_uniform(occupied_cell_ids, rng)

    valid_fits = [f for f in fitnesses if f is not None]
    mean_fit = sum(valid_fits) / len(valid_fits)

    diffs = [(f - mean_fit) if f is not None else 0.0 for f in fitnesses]
    min_diff = min(diffs)
    shifted = [d + abs(min_diff) for d in diffs]

    # Apply temperature scaling
    T = max(config.cell_mean_diff_temperature, 1e-6)
    weights = [s ** (1.0 / T) for s in shifted]

    total = sum(weights)
    if total <= 0.0:
        return select_cell_uniform(occupied_cell_ids, rng)

    probs = np.array([w / total for w in weights])
    idx = int(rng.choice(len(occupied_cell_ids), p=probs))
    return occupied_cell_ids[idx]


def select_cell_ucb(
    occupied_cell_ids: list[int],
    archive: "CVTArchive",
    total_trials: int,
    config: "SamplingConfig",
    rng: np.random.Generator,
) -> int:
    """Choose a cell using UCB with an epsilon-uniform fallback.

    UCB score:  reward_ema + c * sqrt(log(N+1) / (n+1))
    """
    if rng.random() < config.cell_epsilon_uniform:
        return select_cell_uniform(occupied_cell_ids, rng)

    best_cell = occupied_cell_ids[0]
    best_score = -math.inf
    log_n = math.log(total_trials + 1)

    for cell_id in occupied_cell_ids:
        elite = archive.get_cell_elite(cell_id)
        if elite is None:
            # Unvisited — always prefer it
            return cell_id
        n_trials = elite.cell_n_trials
        if n_trials == 0:
            return cell_id
        bonus = config.cell_ucb_c * math.sqrt(log_n / (n_trials + 1))
        score = elite.cell_reward_ema + bonus
        if score > best_score:
            best_score = score
            best_cell = cell_id

    return best_cell


# ---------------------------------------------------------------------------
# Emitter selection
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def select_emitter_fixed(config: "SamplingConfig", rng: np.random.Generator) -> str:
    """Choose 'exploit' or 'explore' with a fixed probability."""
    return "exploit" if rng.random() < config.emitter_p_exploit else "explore"


def select_emitter_momentum(
    config: "SamplingConfig",
    global_state: "dict[str, Any]",
    rng: np.random.Generator,
) -> str:
    """Interpolate p_exploit from momentum_exploit_hi to momentum_exploit_lo based on
    the island's per-step insertion-rate EMA.

    insertion_rate_ema ≈ 1.0  →  archive hot, still finding improvements → exploit more
    insertion_rate_ema ≈ 0.0  →  archive stagnating → explore more

    p_exploit = lerp(momentum_exploit_lo, momentum_exploit_hi, insertion_rate_ema)
    """
    rate = float(global_state.get("insertion_rate_ema", 0.5))
    lo = config.momentum_exploit_lo
    hi = config.momentum_exploit_hi
    p_exploit = lo + (hi - lo) * rate
    return "exploit" if rng.random() < p_exploit else "explore"


def select_emitter_single_curiosity(
    elite_candidate: "Candidate",
    global_logit_exploit: float,
    config: "SamplingConfig",
    rng: np.random.Generator,
) -> str:
    """Blend a global curiosity logit with the cell-local logit.

    New cells rely on the global value; frequently sampled cells rely more on
    their own local statistic.
    """
    warmup = config.emitter_cell_warmup_trials
    w = min(1.0, elite_candidate.cell_n_trials / warmup) if warmup > 0 else 1.0

    c_global = _sigmoid(global_logit_exploit)
    c_cell = _sigmoid(elite_candidate.cell_logit_exploit)

    p_exploit = (1.0 - w) * c_global + w * c_cell
    p_exploit = max(config.emitter_min_p_exploit, min(config.emitter_max_p_exploit, p_exploit))

    return "exploit" if rng.random() < p_exploit else "explore"


def select_emitter_ucb(
    elite_candidate: "Candidate",
    config: "SamplingConfig",
    rng: np.random.Generator,  # noqa: ARG001  (kept for uniform API)
) -> str:
    """UCB emitter selection using per-emitter stats stored on the elite."""
    emitters = ["exploit", "explore"]
    n_cell = elite_candidate.cell_n_trials
    log_n = math.log(n_cell + 1)

    best_emitter = emitters[0]
    best_score = -math.inf

    for emitter in emitters:
        n_e = elite_candidate.cell_emitter_n_trials.get(emitter, 0)
        if n_e == 0:
            return emitter  # always try untried arms first
        ema = elite_candidate.cell_emitter_reward_ema.get(emitter, 0.0)
        bonus = config.emitter_ucb_c * math.sqrt(log_n / (n_e + 1))
        score = ema + bonus
        if score > best_score:
            best_score = score
            best_emitter = emitter

    return best_emitter


def select_emitter_thompson(
    elite_candidate: "Candidate",
    config: "SamplingConfig",  # noqa: ARG001
    rng: np.random.Generator,
) -> str:
    """Thompson sampling emitter selection using Beta distributions."""
    emitters = ["exploit", "explore"]

    best_emitter = emitters[0]
    best_sample = -math.inf

    for emitter in emitters:
        alpha = elite_candidate.cell_emitter_successes.get(emitter, 1.0)
        beta = elite_candidate.cell_emitter_failures.get(emitter, 1.0)
        # numpy Generator doesn't have betavariate; use beta method
        sample = float(rng.beta(alpha, beta))
        if sample > best_sample:
            best_sample = sample
            best_emitter = emitter

    return best_emitter


def select_emitter_softmax(
    elite_candidate: "Candidate",
    emitter_names: list[str],
    global_logits: dict[str, float],
    config: "SamplingConfig",
    rng: np.random.Generator,
) -> str:
    """Softmax emitter selection over any number of emitters.

    Each emitter has a per-cell logit and a global prior logit.
    New cells rely on the global prior; warmed-up cells on their own logits.

    Args:
        emitter_names: All available emitter names (LLM + local).
        global_logits: Dict mapping emitter_name → global logit (prior).
    """
    warmup = config.emitter_cell_warmup_trials
    w = min(1.0, elite_candidate.cell_n_trials / warmup) if warmup > 0 else 1.0

    logits = np.array([
        (1.0 - w) * global_logits.get(name, 0.0) + w * elite_candidate.cell_emitter_logits.get(name, 0.0)
        for name in emitter_names
    ], dtype=float)

    # Softmax
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()

    idx = int(rng.choice(len(emitter_names), p=probs))
    return emitter_names[idx]


# ---------------------------------------------------------------------------
# Unified stat update
# ---------------------------------------------------------------------------

def update_cell_and_global_stats(
    elite_candidate: "Candidate",
    emitter: str,
    reward: float,
    global_state: dict[str, Any],
    config: "SamplingConfig",
) -> None:
    """Update cell-level stats on the Candidate and global state dict.

    Cell stats are stored directly on the Candidate object (the archive elite).
    Global state is a plain dict with keys:
        "total_trials"           int
        "global_logit_exploit"   float
        "global_emitter_stats"   dict[str, {"successes": float, "failures": float}]
    """
    beta = config.cell_reward_ema_beta

    # --- Momentum: per-island insertion-rate EMA ---
    # was_inserted = reward > 0 (any reward means the candidate entered the archive)
    was_inserted = float(reward > 0.0)
    ema_beta = config.momentum_ema_beta
    global_state["insertion_rate_ema"] = (
        (1.0 - ema_beta) * float(global_state.get("insertion_rate_ema", 0.5))
        + ema_beta * was_inserted
    )

    # --- Cell-level UCB stats ---
    elite_candidate.cell_n_trials += 1
    elite_candidate.cell_reward_ema = (
        (1.0 - beta) * elite_candidate.cell_reward_ema + beta * reward
    )

    # --- Per-emitter stats on the cell ---
    n_e = elite_candidate.cell_emitter_n_trials.get(emitter, 0)
    ema_e = elite_candidate.cell_emitter_reward_ema.get(emitter, 0.0)
    elite_candidate.cell_emitter_n_trials[emitter] = n_e + 1
    elite_candidate.cell_emitter_reward_ema[emitter] = (1.0 - beta) * ema_e + beta * reward
    if reward > 0.0:
        elite_candidate.cell_emitter_successes[emitter] = (
            elite_candidate.cell_emitter_successes.get(emitter, 1.0) + reward
        )
    else:
        elite_candidate.cell_emitter_failures[emitter] = (
            elite_candidate.cell_emitter_failures.get(emitter, 1.0) + 1.0
        )

    # --- Softmax per-emitter logit update ---
    if config.emitter_method == "softmax":
        lo = config.emitter_logit_clip_lo
        hi = config.emitter_logit_clip_hi
        # Reinforce the selected emitter; no penalty to others (policy gradient)
        current = elite_candidate.cell_emitter_logits.get(emitter, 0.0)
        elite_candidate.cell_emitter_logits[emitter] = max(lo, min(hi, current + config.emitter_cell_lr * reward))
        # Global prior update
        g_logits = global_state.setdefault("global_emitter_logits", {})
        g_logits[emitter] = max(lo, min(hi, g_logits.get(emitter, 0.0) + config.emitter_global_lr * reward))

    # --- Single-curiosity logit update ---
    if config.emitter_method == "single_curiosity":
        delta = 2.0 * reward - 1.0
        signed_delta = delta if emitter == "exploit" else -delta

        lo = config.emitter_logit_clip_lo
        hi = config.emitter_logit_clip_hi

        elite_candidate.cell_logit_exploit = max(
            lo, min(hi, elite_candidate.cell_logit_exploit + config.emitter_cell_lr * signed_delta)
        )
        global_state["global_logit_exploit"] = max(
            lo,
            min(
                hi,
                global_state.get("global_logit_exploit", 0.0)
                + config.emitter_global_lr * signed_delta,
            ),
        )

    # --- Global emitter Thompson stats ---
    g_stats = global_state.setdefault(
        "global_emitter_stats",
        {"exploit": {"successes": 1.0, "failures": 1.0}, "explore": {"successes": 1.0, "failures": 1.0}},
    )
    if emitter in g_stats:
        if reward > 0.0:
            g_stats[emitter]["successes"] = g_stats[emitter].get("successes", 1.0) + reward
        else:
            g_stats[emitter]["failures"] = g_stats[emitter].get("failures", 1.0) + 1.0


# ---------------------------------------------------------------------------
# Dispatcher helpers used by the engine
# ---------------------------------------------------------------------------

def select_cell(
    occupied_cell_ids: list[int],
    archive: "CVTArchive",
    config: "SamplingConfig",
    rng: np.random.Generator,
    total_trials: int,
) -> int:
    if config.cell_method == "uniform":
        return select_cell_uniform(occupied_cell_ids, rng)
    if config.cell_method == "ucb":
        return select_cell_ucb(occupied_cell_ids, archive, total_trials, config, rng)
    if config.cell_method == "mean_diff":
        return select_cell_mean_diff(occupied_cell_ids, archive, config, rng)
    raise ValueError(f"Unknown cell sampling method: {config.cell_method!r}")


def select_emitter(
    elite_candidate: "Candidate",
    global_logit_exploit: float,
    config: "SamplingConfig",
    rng: np.random.Generator,
    emitter_names: list[str] | None = None,
    global_state: "dict[str, Any] | None" = None,
) -> str:
    """Select an emitter for the next step.

    Args:
        emitter_names: Full list of available emitters (LLM + local). Required for 'softmax'.
        global_state:  Shared global sampling state dict. Required for 'softmax'.
    """
    method = config.emitter_method
    if method == "fixed":
        return select_emitter_fixed(config, rng)
    if method == "momentum":
        return select_emitter_momentum(config, global_state or {}, rng)
    if method == "single_curiosity":
        return select_emitter_single_curiosity(elite_candidate, global_logit_exploit, config, rng)
    if method == "ucb":
        return select_emitter_ucb(elite_candidate, config, rng)
    if method == "thompson":
        return select_emitter_thompson(elite_candidate, config, rng)
    if method == "softmax":
        names = emitter_names or ["exploit", "explore"]
        g_logits = (global_state or {}).get("global_emitter_logits", {})
        return select_emitter_softmax(elite_candidate, names, g_logits, config, rng)
    raise ValueError(f"Unknown emitter sampling method: {method!r}")
