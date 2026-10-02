from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any, Callable, Protocol

import httpx

from evo_lib.candidate import Candidate
from evo_lib.config import EmitterConfig, EvolutionConfig, LLMConfig, LLMPoolConfig, ThinkingMode
from evo_lib.errors import LLMError


@dataclass
class LLMCallResult:
    prompt: str
    response_text: str
    model: str | None
    status: str
    latency_seconds: float
    started_at: str
    finished_at: str
    request_payload: dict[str, Any]
    response_payload: dict[str, Any] | str
    reasoning_text: str | None = None


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        emitter_name: str | None = None,
        thinking_mode: ThinkingMode = "default",
    ) -> LLMCallResult:
        ...


@dataclass(frozen=True)
class LLMRoute:
    api_base_url: str
    api_key: str
    model: str


def _format_stats_key(stats: dict) -> str:
    """Format a stats legend block: 'name: explanation' for each stat."""
    lines = ["Stats key:"]
    for name, entry in stats.items():
        lines.append(f"  {name}: {entry['explanation']}")
    return "\n".join(lines)


def build_prompt(
    template: str,
    *,
    target: Candidate,
    ancestors: list[Candidate],
    inspirations: list[Candidate],
    extra_instructions: str,
    primary_metric_label: str,
    primary_validation_metric_label: str,
    secondary_metric_label: str,
    secondary_validation_metric_label: str,
    code_language: str = "python",
) -> str:
    # Build stats key from target's stats — shown once above target, then all blocks use compact values
    stats_key_block = _format_stats_key(target.stats) if target.stats else ""

    target_block = format_candidate_block(
        label="Target Elite",
        candidate=target,
        primary_metric_label=primary_metric_label,
        primary_validation_metric_label=primary_validation_metric_label,
        secondary_metric_label=secondary_metric_label,
        secondary_validation_metric_label=secondary_validation_metric_label,
        verbose_stats=False,
        code_language=code_language,
    )
    ancestor_block = format_candidate_list(
        title="Ancestors",
        candidates=ancestors,
        primary_metric_label=primary_metric_label,
        primary_validation_metric_label=primary_validation_metric_label,
        secondary_metric_label=secondary_metric_label,
        secondary_validation_metric_label=secondary_validation_metric_label,
        code_language=code_language,
    )
    inspiration_block = format_candidate_list(
        title="Inspirational Elites",
        candidates=inspirations,
        primary_metric_label=primary_metric_label,
        primary_validation_metric_label=primary_validation_metric_label,
        secondary_metric_label=secondary_metric_label,
        secondary_validation_metric_label=secondary_validation_metric_label,
        code_language=code_language,
    )
    replacements = {
        "{stats_key}": stats_key_block,
        "{target_candidate}": target_block,
        "{ancestor_candidates}": ancestor_block,
        "{inspiration_candidates}": inspiration_block,
        "{extra_instructions}": extra_instructions,
    }
    if any(token in template for token in replacements):
        prompt = template
        for placeholder, value in replacements.items():
            prompt = prompt.replace(placeholder, value)
        return prompt
    return build_standard_edit_prompt(
        task_instructions=template,
        stats_key_block=stats_key_block,
        target_block=target_block,
        ancestor_block=ancestor_block,
        inspiration_block=inspiration_block,
        extra_instructions=extra_instructions,
    )


def build_standard_edit_prompt(
    *,
    task_instructions: str,
    stats_key_block: str,
    target_block: str,
    ancestor_block: str,
    inspiration_block: str,
    extra_instructions: str,
) -> str:
    sections = [
        task_instructions.strip(),
        "Edit format requirements:\n"
        "- Return one fenced code block containing SEARCH/REPLACE diffs.\n"
        "- Within the block, you may use one or more SEARCH/REPLACE edit blocks.\n"
        "- Use this exact edit format for each block:\n"
        "  <<<<<<< SEARCH\n"
        "  old code region to find in the parent program\n"
        "  =======\n"
        "  new code region to replace it with\n"
        "  >>>>>>> REPLACE\n"
        "- Use the full parent file in SEARCH only when a full rewrite is necessary.",
        stats_key_block,
        target_block,
        ancestor_block,
        inspiration_block,
        "Produce one edited candidate as a SEARCH/REPLACE diff block.",
    ]
    if extra_instructions.strip():
        sections.append(f"Additional instructions:\n{extra_instructions}")
    return "\n\n".join(section for section in sections if section.strip())


def random_init_instructions(extra_instructions: str) -> str:
    seed_text = (
        "Initialization stage: generate diverse, independent random candidates to seed the archive. "
        "Do not refine, imitate, or interpolate from prior candidates. Prefer broad structural variety."
    )
    if extra_instructions.strip():
        return f"{seed_text}\n\n{extra_instructions}"
    return seed_text


def format_candidate_list(
    *,
    title: str,
    candidates: list[Candidate],
    primary_metric_label: str,
    primary_validation_metric_label: str,
    secondary_metric_label: str,
    secondary_validation_metric_label: str,
    code_language: str = "python",
) -> str:
    if not candidates:
        return f"{title}:\nNone"
    return "\n\n".join(
        [
            f"{title}:",
            *[
                format_candidate_block(
                    label=f"{title[:-1] if title.endswith('s') else title} {index + 1}",
                    candidate=candidate,
                    primary_metric_label=primary_metric_label,
                    primary_validation_metric_label=primary_validation_metric_label,
                    secondary_metric_label=secondary_metric_label,
                    secondary_validation_metric_label=secondary_validation_metric_label,
                    verbose_stats=False,
                    code_language=code_language,
                )
                for index, candidate in enumerate(candidates)
            ],
        ]
    )


def format_candidate_block(
    *,
    label: str,
    candidate: Candidate,
    primary_metric_label: str,
    primary_validation_metric_label: str,
    secondary_metric_label: str,
    secondary_validation_metric_label: str,
    verbose_stats: bool = True,
    code_language: str = "python",
) -> str:
    all_metrics = [
        (primary_metric_label, candidate.primary_fitness),
        (primary_validation_metric_label, candidate.primary_validation_fitness),
        (secondary_metric_label, candidate.secondary_fitness),
        (secondary_validation_metric_label, candidate.secondary_validation_fitness),
    ]
    metrics = [f"{lbl}: {_format_metric(val)}" for lbl, val in all_metrics if val is not None]
    parts = [
        f"{label} [{candidate.id}]",
        f"Fitness: {', '.join(metrics) if metrics else '-'}",
    ]
    if candidate.stats:
        parts.append(_format_stats(candidate.stats, verbose=verbose_stats))
    parts.append(f"```{code_language}\n{candidate.code}\n```")
    return "\n".join(parts)


def _format_stats(stats: dict, *, verbose: bool = True) -> str:
    lines = ["Stats:"]
    for name, entry in stats.items():
        value = entry["value"]
        value_str = f"{value:.4f}" if isinstance(value, float) else str(value)
        if verbose:
            explanation = entry["explanation"]
            lines.append(f"  {name}: {value_str} — {explanation}")
        else:
            lines.append(f"  {name}: {value_str}")
    return "\n".join(lines)


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig, http_client: httpx.Client | None = None, *, random_seed: int | None = None):
        self.config = config
        self._client = http_client or httpx.Client(
            timeout=httpx.Timeout(connect=30, read=config.timeout_seconds, write=30, pool=10)
        )
        self._rng = random.Random(random_seed)
        self._routes = self._load_routes(config)

    def generate(
        self,
        prompt: str,
        *,
        emitter_name: str | None = None,
        thinking_mode: ThinkingMode = "default",
    ) -> LLMCallResult:
        route = self._choose_route()
        return self._send_prompt(prompt, route=route, thinking_mode=thinking_mode)

    def _send_prompt(
        self,
        prompt: str,
        *,
        route: LLMRoute,
        thinking_mode: ThinkingMode,
    ) -> LLMCallResult:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": self.config.n,
        }
        resolved_thinking_mode = self.config.thinking_mode if thinking_mode == "default" else thinking_mode
        if resolved_thinking_mode == "enabled":
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        elif resolved_thinking_mode == "disabled":
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            payload["reasoning_format"] = "none"
        endpoint = route.api_base_url.rstrip("/") + "/chat/completions"
        if route.model:
            payload["model"] = route.model
        headers = {
            "Authorization": f"Bearer {route.api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        last_error: Exception | None = None
        recovery_deadline = time.monotonic() + self.config.connection_recovery_timeout_seconds
        max_retries_long = min(self.config.max_retries, 15)
        while True:
            started = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            try:
                response = self._client.post(endpoint, json=payload, headers=headers,
                                             timeout=httpx.Timeout(connect=30, read=self.config.timeout_seconds, write=30, pool=10))
                latency = time.monotonic() - started
                finished_at = datetime.now(timezone.utc).isoformat()
                response.raise_for_status()
                body = response.json()
                text = self._extract_text(body)
                return LLMCallResult(
                    prompt=prompt,
                    response_text=text,
                    model=body.get("model") or (route.model or None),
                    status="ok",
                    latency_seconds=latency,
                    started_at=started_at,
                    finished_at=finished_at,
                    request_payload=payload,
                    response_payload=body,
                    reasoning_text=self._extract_reasoning(body),
                )
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                retryable_outage = self._is_retryable_outage(exc)
                if retryable_outage:
                    can_retry = self.config.connection_recovery_timeout_seconds > 0 and time.monotonic() < recovery_deadline
                else:
                    can_retry = attempt < max_retries_long
                if not can_retry:
                    raise LLMError(f"LLM request failed after {attempt + 1} attempt(s): {exc}") from exc
                # Local inference endpoints recover quickly; honor the configured
                # cadence instead of sleeping for escalating multi-minute periods.
                time.sleep(self.config.retry_backoff_seconds)
                attempt += 1
        raise LLMError(f"LLM request failed: {last_error}")

    @staticmethod
    def _is_retryable_outage(exc: Exception) -> bool:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return status in {408, 429, 502, 503, 504}
        return isinstance(exc, httpx.RequestError)

    def _choose_route(self, model_override: str | None = None) -> LLMRoute:
        if not self._routes:
            raise LLMError("No LLM routes configured")
        selected = self._routes[0] if len(self._routes) == 1 else self._rng.choice(self._routes)
        if model_override is None:
            return selected
        return LLMRoute(api_base_url=selected.api_base_url, api_key=selected.api_key, model=model_override)

    @staticmethod
    def _load_routes(config: LLMConfig) -> list[LLMRoute]:
        if config.llm_pool_path is None:
            return [LLMRoute(api_base_url=config.api_base_url, api_key=config.api_key, model=config.model or "")]
        pool = LLMPoolConfig.from_json_file(Path(config.llm_pool_path))
        routes: list[LLMRoute] = []
        for endpoint in pool.endpoints:
            for model in endpoint.models:
                routes.append(
                    LLMRoute(
                        api_base_url=endpoint.api_base_url,
                        api_key=endpoint.api_key or config.api_key,
                        model=model,
                    )
                )
        if not routes:
            raise LLMError(f"No endpoint/model routes found in LLM pool config: {config.llm_pool_path}")
        return routes

    @staticmethod
    def _extract_text(body: dict[str, Any], ) -> str:
        choices = body["choices"]
        if not choices:
            raise LLMError("LLM response contained no choices")
        message = choices[0]["message"]
        content = message.get("content")
        if content is None and message.get("reasoning"):
            # vLLM qwen3 reasoning parser splits thinking into the `reasoning`
            # field; when the generation is cut short the final `content` may be
            # null while reasoning still holds the text. Fall back so callers can
            # still parse (and repair) a partial fenced module.
            content = message.get("reasoning")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
            text = "".join(parts).strip()
            if not text:
                raise LLMError("LLM response content parts contained no text")
            return text
        raise LLMError("Unsupported LLM response content format")

    @staticmethod
    def _extract_reasoning(body: dict[str, Any]) -> str | None:
        try:
            choices = body.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            for key in ("reasoning", "reasoning_content"):
                val = message.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            return None
        except (KeyError, TypeError, IndexError):
            return None

class MultiEmitterLLMClient:
    def __init__(
        self,
        default_config: LLMConfig,
        emitters: list[EmitterConfig],
        http_client_factory: Callable[[LLMConfig], httpx.Client] | None = None,
        *,
        random_seed: int | None = None,
    ) -> None:
        import random as _random
        self._rng = _random.Random(random_seed)
        self._default_client = OpenAICompatibleClient(
            default_config,
            http_client=http_client_factory(default_config) if http_client_factory is not None else None,
            random_seed=random_seed,
        )
        # Single-LLM override per emitter (existing behaviour)
        self._emitter_clients: dict[str, OpenAICompatibleClient] = {}
        # Pool of LLM clients per emitter with sampling weights
        self._emitter_pools: dict[str, tuple[list[OpenAICompatibleClient], list[float]]] = {}
        seed_base = random_seed or 0
        for index, emitter in enumerate(emitters):
            if emitter.llm_pool is not None:
                pool_clients = [
                    OpenAICompatibleClient(
                        cfg,
                        http_client=http_client_factory(cfg) if http_client_factory is not None else None,
                        random_seed=seed_base + index * 100 + pool_idx,
                    )
                    for pool_idx, cfg in enumerate(emitter.llm_pool)
                ]
                # Normalise weights (default uniform)
                raw_weights = emitter.llm_pool_weights or [1.0] * len(pool_clients)
                total = sum(raw_weights)
                norm_weights = [w / total for w in raw_weights]
                self._emitter_pools[emitter.name] = (pool_clients, norm_weights)
            elif emitter.llm is not None:
                self._emitter_clients[emitter.name] = OpenAICompatibleClient(
                    emitter.llm,
                    http_client=http_client_factory(emitter.llm) if http_client_factory is not None else None,
                    random_seed=seed_base + index + 1,
                )

    def generate(
        self,
        prompt: str,
        *,
        emitter_name: str | None = None,
        thinking_mode: ThinkingMode = "default",
    ) -> LLMCallResult:
        # Pool takes priority over single-LLM override
        if emitter_name is not None and emitter_name in self._emitter_pools:
            pool_clients, weights = self._emitter_pools[emitter_name]
            # Try each client in weighted-random order, falling back on failure
            order = self._rng.choices(range(len(pool_clients)), weights=weights, k=len(pool_clients))
            seen = set()
            ordered = [i for i in order if not (i in seen or seen.add(i))]  # deduplicate preserving order
            last_exc: Exception | None = None
            for idx in ordered:
                try:
                    return pool_clients[idx].generate(prompt, emitter_name=emitter_name, thinking_mode=thinking_mode)
                except Exception as exc:
                    last_exc = exc
                    continue
            raise last_exc or RuntimeError("All LLM pool clients failed")
        else:
            client = self._emitter_clients.get(emitter_name, self._default_client)
            return client.generate(prompt, emitter_name=emitter_name, thinking_mode=thinking_mode)


def build_llm_client(
    config: EvolutionConfig,
    http_client_factory: Callable[[LLMConfig], httpx.Client] | None = None,
    *,
    random_seed: int | None = None,
) -> LLMClient:
    if config.generation_method in {"llm_emitters", "llm_emitters_emitter_curiosity"} and config.emitters:
        return MultiEmitterLLMClient(
            default_config=config.llm,
            emitters=config.emitters,
            http_client_factory=http_client_factory,
            random_seed=random_seed,
        )
    return OpenAICompatibleClient(
        config=config.llm,
        http_client=http_client_factory(config.llm) if http_client_factory is not None else None,
        random_seed=random_seed,
    )
