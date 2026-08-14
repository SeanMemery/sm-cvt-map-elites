from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import httpx

from evo_lib.config import EmitterConfig, EvolutionConfig, LLMConfig, LLMPoolConfig, ParsingConfig, PlateauEmitterSchedulerConfig
from evo_lib.candidate import Candidate
from evo_lib.errors import ConfigurationError, LLMError, ParsingError
from evo_lib import llm as llm_module
from evo_lib.llm import MultiEmitterLLMClient, OpenAICompatibleClient, build_prompt
from evo_lib.parsing import apply_edit_script, parse_edit_candidates


def test_parse_edit_candidates_applies_search_replace_blocks():
    response = """
```diff
<<<<<<< SEARCH
return x + 1
=======
return x + 4
>>>>>>> REPLACE
```

```diff
<<<<<<< SEARCH
return x + 1
=======
return x * 3
>>>>>>> REPLACE
```
"""
    candidates = parse_edit_candidates(
        response,
        "def solve(x):\n    return x + 1\n",
        ParsingConfig(max_candidates_per_response=8),
    )

    assert len(candidates) == 2
    assert "return x + 4" in candidates[0].code
    assert "return x * 3" in candidates[1].code
    assert candidates[0].metadata["edit_format"] == "search_replace"


def test_parse_edit_candidates_accepts_indented_search_replace_blocks():
    response = """
```diff
  <<<<<<< SEARCH
  def solve(x):
      return x + 1
  =======
  def solve(x):
      return x + 4
  >>>>>>> REPLACE
```
"""
    candidates = parse_edit_candidates(
        response,
        "def solve(x):\n    return x + 1\n",
        ParsingConfig(max_candidates_per_response=8),
    )

    assert len(candidates) == 1
    assert candidates[0].code == "def solve(x):\n    return x + 4\n"


def test_parse_edit_candidates_accepts_crlf_search_replace_blocks():
    response = (
        "```diff\r\n"
        "<<<<<<< SEARCH\r\n"
        "def solve(x):\r\n"
        "    return x + 1\r\n"
        "=======\r\n"
        "def solve(x):\r\n"
        "    return x + 4\r\n"
        ">>>>>>> REPLACE\r\n"
        "```\r\n"
    )
    candidates = parse_edit_candidates(
        response,
        "def solve(x):\n    return x + 1\n",
        ParsingConfig(max_candidates_per_response=8),
    )

    assert len(candidates) == 1
    assert candidates[0].code == "def solve(x):\n    return x + 4\n"


def test_parse_edit_candidates_applies_multiple_edit_blocks_within_one_candidate():
    response = """
```diff
<<<<<<< SEARCH
helper = 1
=======
helper = 2
>>>>>>> REPLACE
<<<<<<< SEARCH
return helper + x
=======
return helper * x
>>>>>>> REPLACE
```
"""
    candidates = parse_edit_candidates(
        response,
        "def solve(x):\n    helper = 1\n    return helper + x\n",
        ParsingConfig(max_candidates_per_response=8),
    )

    assert len(candidates) == 1
    assert candidates[0].code == "def solve(x):\n    helper = 2\n    return helper * x\n"
    assert candidates[0].metadata["edit_count"] == 2


def test_apply_edit_script_uses_updated_code_for_later_edit_blocks():
    updated = apply_edit_script(
        "alpha\nbeta\n",
        [
            {"search": "alpha", "replace": "gamma"},
            {"search": "gamma", "replace": "delta"},
        ],
    )

    assert updated == "delta\nbeta\n"


def test_apply_edit_script_supports_full_file_insert_for_empty_base():
    updated = apply_edit_script(
        "",
        [{"search": "", "replace": "def solve(x):\n    return x + 9\n"}],
    )
    assert "return x + 9" in updated


def test_apply_edit_script_allows_implicit_full_rewrite_for_tiny_seed():
    base = (
        '#include <bits/stdc++.h>\n'
        "using namespace std;\n"
        "int main(){\n"
        '    std::cout << "Hello, World!" << std::endl;\n'
        "    return 0;\n"
        "}"
    )
    mistaken_search = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n\n"
        "int main(){\n"
        "    ios::sync_with_stdio(false);\n"
        "    cin.tie(nullptr);\n"
        "    return 0;\n"
        "}"
    )
    replacement = (
        "#include <bits/stdc++.h>\n"
        "using namespace std;\n"
        "int main(){\n"
        "    ios::sync_with_stdio(false);\n"
        "    cin.tie(nullptr);\n"
        "    return 0;\n"
        "}"
    )

    updated = apply_edit_script(base, [{"search": mistaken_search, "replace": replacement}])

    assert updated == replacement


def test_parse_edit_candidates_rejects_plain_code_block_without_search_replace():
    response = """
```python
import numpy as np

def solve(x):
    return x + 9
```
"""
    try:
        parse_edit_candidates(
            response,
            "def solve(x):\n    return x + 1\n",
            ParsingConfig(max_candidates_per_response=8),
        )
    except ParsingError as exc:
        assert "SEARCH/REPLACE" in str(exc)
    else:
        raise AssertionError("Expected plain code block to be rejected without SEARCH/REPLACE edits")


def test_build_prompt_preserves_literal_json_braces():
    prompt = build_prompt(
        (
            "Return SEARCH/REPLACE diff blocks.\n"
            "Preserve literal braces like {\"status\": \"ok\"} in the task instructions.\n"
        ),
        target=Candidate(id="cand_target", code="def solve(x):\n    return x\n", primary_fitness=1.0),
        ancestors=[],
        inspirations=[],
        extra_instructions="Stay minimal.",
        primary_metric_label="Primary",
        primary_validation_metric_label="Primary Validation",
        secondary_metric_label="Secondary",
        secondary_validation_metric_label="Secondary Validation",
    )
    assert "SEARCH/REPLACE" in prompt
    assert "one or more SEARCH/REPLACE edit blocks" in prompt
    assert "Target Elite" in prompt
    assert '{"status": "ok"}' in prompt


def test_openai_compatible_client_parses_chat_completion():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["messages"][0]["content"] == "hello"
        assert payload["model"] == "test-model"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef solve():\n    return 1\n```"
                        }
                    }
                ],
                "model": "served-model",
            },
        )

    client = OpenAICompatibleClient(
        config=LLMConfig(
            api_base_url="https://example.test/v1",
            api_key="secret",
            model="test-model",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.generate("hello")

    assert result.status == "ok"
    assert "def solve" in result.response_text
    assert result.model == "served-model"


def test_llm_pool_config_round_trips():
    payload = {
        "endpoints": [
            {"api_base_url": "https://a.test/v1", "models": ["m1", "m2"]},
            {"api_base_url": "https://b.test/v1", "api_key": "other", "models": ["m3"]},
        ]
    }
    loaded = LLMPoolConfig.from_dict(payload)
    assert len(loaded.endpoints) == 2
    assert loaded.endpoints[0].models == ["m1", "m2"]
    assert loaded.endpoints[1].api_key == "other"


def test_openai_compatible_client_samples_from_multiple_pool_routes(tmp_path):
    seen_routes: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_routes.append((str(request.url), payload.get("model")))
        assert payload["messages"][0]["content"] == "hello"
        assert payload["model"] in {"model-a", "model-b", "model-c"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef solve():\n    return 1\n```"
                        }
                    }
                ],
                "model": payload["model"],
            },
        )

    pool_path = tmp_path / "llm_pool.json"
    pool_path.write_text(
        json.dumps(
            {
                "endpoints": [
                    {"api_base_url": "https://a.test/v1", "models": ["model-a", "model-b"]},
                    {"api_base_url": "https://b.test/v1", "models": ["model-c"]},
                ]
            }
        ),
        encoding="utf-8",
    )

    client = OpenAICompatibleClient(
        config=LLMConfig(
            api_base_url="https://unused.test/v1",
            api_key="secret",
            llm_pool_path=str(pool_path),
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        random_seed=7,
    )

    results = [client.generate("hello") for _ in range(6)]

    assert all(result.status == "ok" for result in results)
    assert {result.model for result in results}.issubset({"model-a", "model-b", "model-c"})
    assert len(set(seen_routes)) >= 2
    assert any(url.startswith("https://a.test/v1/chat/completions") for url, _ in seen_routes)
    assert any(url.startswith("https://b.test/v1/chat/completions") for url, _ in seen_routes)


def test_openai_compatible_client_sends_thinking_flag_when_enabled():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef solve():\n    return 1\n```"
                        }
                    }
                ],
                "model": "served-model",
            },
        )

    client = OpenAICompatibleClient(
        config=LLMConfig(
            api_base_url="https://example.test/v1",
            api_key="secret",
            thinking_mode="enabled",
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.generate("hello")

    assert result.status == "ok"


def test_openai_compatible_client_waits_through_transient_connection_outage(monkeypatch):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(llm_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm_module.time, "sleep", clock.sleep)

    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise httpx.ConnectError("router offline", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "```python\ndef solve():\n    return 1\n```"
                        }
                    }
                ],
                "model": "served-model",
            },
        )

    client = OpenAICompatibleClient(
        config=LLMConfig(
            api_base_url="https://example.test/v1",
            api_key="secret",
            max_retries=0,
            retry_backoff_seconds=2.0,
            connection_recovery_timeout_seconds=180.0,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = client.generate("hello")

    assert result.status == "ok"
    assert attempts["count"] == 3
    assert clock.sleeps == [2.0, 2.0]


def test_openai_compatible_client_stops_after_connection_recovery_window(monkeypatch):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0
            self.sleeps: list[float] = []

        def monotonic(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)
            self.now += seconds

    clock = FakeClock()
    monkeypatch.setattr(llm_module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(llm_module.time, "sleep", clock.sleep)

    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        raise httpx.ConnectError("router offline", request=request)

    client = OpenAICompatibleClient(
        config=LLMConfig(
            api_base_url="https://example.test/v1",
            api_key="secret",
            max_retries=0,
            retry_backoff_seconds=2.0,
            connection_recovery_timeout_seconds=5.0,
        ),
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    try:
        client.generate("hello")
    except LLMError as exc:
        assert "router offline" in str(exc)
    else:
        raise AssertionError("Expected connection recovery window to expire")

    assert attempts["count"] == 3
    assert clock.sleeps == [2.0, 2.0]


def test_multi_emitter_client_routes_to_emitter_specific_llm_and_thinking():
    seen_requests: list[tuple[str, str | None, dict[str, bool] | None]] = []

    def build_client(config: LLMConfig) -> httpx.Client:
        def handler(request: httpx.Request) -> httpx.Response:
            payload = json.loads(request.content.decode("utf-8"))
            seen_requests.append((str(request.url), payload.get("model"), payload.get("chat_template_kwargs")))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "```python\ndef solve():\n    return 1\n```"
                            }
                        }
                    ],
                    "model": payload.get("model") or "served-model",
                },
            )

        return httpx.Client(transport=httpx.MockTransport(handler))

    client = MultiEmitterLLMClient(
        default_config=LLMConfig(
            api_base_url="https://default.test/v1",
            api_key="secret",
            model="default-model",
        ),
        emitters=[
            EmitterConfig(
                name="runtime",
                llm=LLMConfig(
                    api_base_url="https://runtime.test/v1",
                    api_key="runtime-secret",
                    model="runtime-model",
                ),
                thinking_mode="enabled",
            )
        ],
        http_client_factory=build_client,
        random_seed=7,
    )

    result = client.generate("hello", emitter_name="runtime", thinking_mode="enabled")

    assert result.status == "ok"
    assert seen_requests == [
        ("https://runtime.test/v1/chat/completions", "runtime-model", {"enable_thinking": True})
    ]


def test_evolution_config_round_trips_llm_emitters(base_config):
    config = replace(
        base_config,
        generation_method="llm_emitters",
        llm=replace(base_config.llm, thinking_mode="disabled"),
        emitters=[
            EmitterConfig(
                name="exploit",
                extra_instructions="Optimize runtime first.",
                thinking_mode="enabled",
                ancestor_count=3,
                inspiration_elite_count=0,
                llm=LLMConfig(
                    api_base_url="https://large.test/v1",
                    api_key="secret",
                    model="large-model",
                ),
            )
        ],
    )

    loaded = EvolutionConfig.from_dict(config.to_dict())

    assert loaded.generation_method == "llm_emitters"
    assert loaded.llm.thinking_mode == "disabled"
    assert loaded.emitters is not None
    assert loaded.emitters[0].name == "exploit"
    assert loaded.emitters[0].thinking_mode == "enabled"
    assert loaded.emitters[0].ancestor_count == 3
    assert loaded.emitters[0].inspiration_elite_count == 0
    assert loaded.emitters[0].llm is not None
    assert loaded.emitters[0].llm.api_base_url == "https://large.test/v1"


def test_evolution_config_round_trips_emitter_curiosity_generation_mode(base_config):
    config = replace(
        base_config,
        generation_method="llm_emitters_emitter_curiosity",
        emitters=[
            EmitterConfig(
                name="explore",
                selection_weight=3.0,
            )
        ],
    )

    loaded = EvolutionConfig.from_dict(config.to_dict())

    assert loaded.generation_method == "llm_emitters_emitter_curiosity"
    assert loaded.emitters is not None
    assert loaded.emitters[0].name == "explore"


def test_evolution_config_round_trips_plateau_emitter_scheduler(base_config):
    config = replace(
        base_config,
        generation_method="llm_emitters",
        emitter_selection_strategy="plateau_scheduler",
        emitters=[
            EmitterConfig(name="explore", selection_weight=2.0),
            EmitterConfig(name="exploit", selection_weight=1.0),
        ],
        emitter_plateau_scheduler=PlateauEmitterSchedulerConfig(
            target_emitter_name="exploit",
            patience_steps=25,
            target_weight_growth=2.0,
            non_target_weight_decay=0.8,
        ),
    )

    loaded = EvolutionConfig.from_dict(config.to_dict())

    assert loaded.emitter_selection_strategy == "plateau_scheduler"
    assert loaded.emitter_plateau_scheduler is not None
    assert loaded.emitter_plateau_scheduler.target_emitter_name == "exploit"
    assert loaded.emitter_plateau_scheduler.patience_steps == 25
