from __future__ import annotations

from dataclasses import replace
import threading
import time
from datetime import datetime, timezone

import httpx
import pytest

from evo_lib.candidate import Candidate
from evo_lib.config import EvolutionConfig, LLMConfig, NormalizerConfig, StorageConfig, ThinkingMode
from evo_lib.llm import LLMCallResult


class FakeLLMClient:
    def __init__(self, responses: list[str], *, delay_seconds: float = 0.0):
        self._responses = list(responses)
        self._delay_seconds = delay_seconds
        self.calls = 0
        self.prompts: list[str] = []
        self.emitter_names: list[str | None] = []
        self.thinking_modes: list[ThinkingMode] = []
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        emitter_name: str | None = None,
        thinking_mode: ThinkingMode = "default",
    ) -> LLMCallResult:
        started_at = datetime.now(timezone.utc).isoformat()
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        with self._lock:
            self.prompts.append(prompt)
            self.emitter_names.append(emitter_name)
            self.thinking_modes.append(thinking_mode)
            if self.calls >= len(self._responses):
                response = self._responses[-1]
            else:
                response = self._responses[self.calls]
            self.calls += 1
        return LLMCallResult(
            prompt=prompt,
            response_text=response,
            model="fake-model",
            status="ok",
            latency_seconds=0.01,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            request_payload={"prompt": prompt, "emitter_name": emitter_name, "thinking_mode": thinking_mode},
            response_payload={"response": response},
        )


@pytest.fixture
def base_config(tmp_path) -> EvolutionConfig:
    return EvolutionConfig(
        run_name="test_run",
        project_id="test_project",
        output_dir=str(tmp_path / "runs"),
        descriptor_dim=2,
        num_centroids=4,
        cvt_samples=16,
        remap_interval=2,
        initial_random_steps=0,
        parents_per_mutation=2,
        ancestor_count=2,
        inspiration_elite_count=2,
        max_steps=10,
        random_seed=7,
        user_prompt=(
            "Edit the target elite using diffs.\n"
            "Preserve the function signature and keep the code executable."
        ),
        extra_instructions="Prefer short code.",
        primary_metric_label="Visible Score",
        secondary_metric_label="Hidden Score",
        llm=LLMConfig(
            api_base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
        ),
        normalizer=NormalizerConfig(dim=2, lower_quantile=0.0, upper_quantile=1.0, history_size=32),
        storage=StorageConfig(checkpoint_interval=2, archive_snapshot_interval=2),
    )


@pytest.fixture
def seed_candidates() -> list[Candidate]:
    return [
        Candidate(id="seed_a", code="def solve(x):\n    return x + 1\n"),
        Candidate(id="seed_b", code="def solve(x):\n    return x * 2\n"),
    ]


@pytest.fixture
def descriptor_fn():
    def _descriptor(candidate: Candidate) -> list[float]:
        return [float(len(candidate.code)), float(candidate.code.count("return"))]

    return _descriptor


@pytest.fixture
def primary_fitness():
    def _fitness(candidate: Candidate) -> float:
        return float(candidate.code.count("+") + candidate.code.count("*") * 2 + candidate.code.count("return"))

    return _fitness


@pytest.fixture
def primary_validation_fitness():
    def _validation(candidate: Candidate) -> float:
        return float(candidate.code.count("-") + candidate.code.count("return") * 0.5)

    return _validation


@pytest.fixture
def secondary_fitness():
    def _secondary(candidate: Candidate) -> float:
        return float(candidate.code.count("x"))

    return _secondary


@pytest.fixture
def secondary_validation_fitness():
    def _validation(candidate: Candidate) -> float:
        return float(candidate.code.count("solve") + candidate.code.count("x") * 0.25)

    return _validation
