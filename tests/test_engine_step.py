from __future__ import annotations

import json
from dataclasses import replace

from evo_lib.candidate import Candidate
from evo_lib.config import EmitterConfig, NormalizerConfig, PlateauEmitterSchedulerConfig, SecondaryEvalConfig  # noqa: F401 (SecondaryEvalConfig still used below)
from evo_lib.engine import EvolutionEngine
from tests.conftest import FakeLLMClient


def _diff_block(search: str, replace: str) -> str:
    return (
        "```python\n"
        "<<<<<<< SEARCH\n"
        f"{search.rstrip()}\n"
        "=======\n"
        f"{replace.rstrip()}\n"
        ">>>>>>> REPLACE\n"
        "```"
    )


def _seed_parent_code() -> str:
    return "def solve(x):\n    return x * 2\n"


def _seed_parent_diff(replace: str) -> str:
    return _diff_block(_seed_parent_code(), replace)


def test_engine_initialize_and_step(
    base_config,
    seed_candidates,
    descriptor_fn,
    primary_fitness,
    primary_validation_fitness,
    secondary_fitness,
    secondary_validation_fitness,
):
    config = replace(
        base_config,
        secondary_eval=SecondaryEvalConfig(stages=1, p1=1.0, p2=0.5),
    )
    client = FakeLLMClient(
        [
            "\n\n".join(
                [
                    _seed_parent_diff("def solve(x):\n    return x + 3\n"),
                    _seed_parent_diff("def solve(x):\n    return x - 4\n"),
                ]
            ),
            "\n\n".join(
                [
                    _seed_parent_diff("def solve(x):\n    return x - 1\n"),
                    _seed_parent_diff("def solve(x):\n    return x + 0\n"),
                ]
            ),
        ]
    )
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        primary_validation_fitness=primary_validation_fitness,
        secondary_fitness=secondary_fitness,
        secondary_validation_fitness=secondary_validation_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])
    engine.step()
    engine.step()

    assert engine.step_count == 2
    assert engine.stats.generated_count >= 4
    assert len(engine.archive.all_elites()) >= 1

    snapshot_path = engine.run_store.run_dir / "archive_snapshots" / "step_000002.json"
    assert snapshot_path.exists()
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["step"] == 2
    assert snapshot["descriptor_labels"] == config.descriptor_labels
    assert (engine.run_store.run_dir / "checkpoints" / "latest.pkl").exists()
    with (engine.run_store.run_dir / "candidates.jsonl").open("r", encoding="utf-8") as handle:
        candidate_lines = [json.loads(line) for line in handle]
    generated = [line for line in candidate_lines if line["id"].startswith("cand_")]
    assert all(line["llm_call_id"] for line in generated)
    assert any(line["primary_validation_fitness"] is not None for line in generated)
    # secondary eval fires at stage end via run(), not during individual step() calls
    with (engine.run_store.run_dir / "stats.jsonl").open("r", encoding="utf-8") as handle:
        stats = [json.loads(line) for line in handle]
    assert any(entry.get("best_primary_validation_fitness") is not None for entry in stats)
    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert llm_calls[-1]["child_ids"]


def test_engine_random_init_stage_uses_no_parents_and_delays_secondary_eval(
    base_config,
    seed_candidates,
    descriptor_fn,
    primary_fitness,
    primary_validation_fitness,
    secondary_fitness,
    secondary_validation_fitness,
):
    config = replace(
        base_config,
        initial_random_steps=2,
        secondary_eval=SecondaryEvalConfig(stages=1, p1=1.0, p2=0.5),
    )
    client = FakeLLMClient(
        [
            _diff_block("", "def solve(x):\n    return x + 5\n"),
            _diff_block("", "def solve(x):\n    return x * 6\n"),
            _seed_parent_diff("def solve(x):\n    return x - 7\n"),
        ]
    )
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        primary_validation_fitness=primary_validation_fitness,
        secondary_fitness=secondary_fitness,
        secondary_validation_fitness=secondary_validation_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])
    engine.step()
    engine.step()
    engine.step()

    assert "Target Elite" in client.prompts[0]
    assert "Edit the target elite using diffs." in client.prompts[0]
    assert "Initialization stage" in client.prompts[0]
    assert "Target Elite" in client.prompts[1]
    assert "Edit the target elite using diffs." in client.prompts[1]
    assert "Initialization stage" in client.prompts[1]
    assert "Target Elite" in client.prompts[2]

    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert llm_calls[0]["phase"] == "random_init"
    assert llm_calls[1]["phase"] == "random_init"
    assert llm_calls[2]["phase"] == "evolution"

    # secondary eval now only fires at stage end, not during individual steps
    with (engine.run_store.run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    secondary_events = [event for event in events if event.get("type") == "secondary_eval"]
    assert secondary_events == []  # no stage end occurred; engine.run() was not called


def test_engine_repairs_empty_array_parse_failure_with_reprompt(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient(
        [
            "[]",
            _seed_parent_diff("def solve(x):\n    return x + 9\n"),
        ]
    )
    seed_code = _seed_parent_code()
    engine = EvolutionEngine(
        config=replace(base_config, generation_method="standard", initial_random_steps=0),
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_a", code=seed_code)])
    engine.step()

    assert engine.step_count == 1
    assert client.calls == 2
    assert "previous response could not be parsed or applied" in client.prompts[1].lower()
    assert "No valid SEARCH/REPLACE edit blocks were found" in client.prompts[1]
    generated = [candidate for candidate in engine.candidate_registry.values() if candidate.id.startswith("cand_")]
    assert generated
    assert generated[0].metadata["parse_repair_attempted"] is True


def test_engine_repairs_search_mismatch_parse_failure_with_reprompt(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient(
        [
            _diff_block("def solve(x):\n    return x + 999\n", "def solve(x):\n    return x + 7\n"),
            _diff_block("def solve(x):\n    return x + 1\n", "def solve(x):\n    return x + 7\n"),
        ]
    )
    engine = EvolutionEngine(
        config=replace(base_config),
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_a", code="def solve(x):\n    return x + 1\n")])
    engine.step()

    assert engine.step_count == 1
    assert client.calls == 2
    assert "found 0 matches" in client.prompts[1]
    assert "Broken response:" in client.prompts[1]
    generated = [candidate for candidate in engine.candidate_registry.values() if candidate.id.startswith("cand_")]
    assert generated
    assert "return x + 7" in generated[0].code


def test_engine_logs_failed_llm_call_for_unrepaired_parse_failure(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient(
        [
            "not a diff",
            "still not a diff",
        ]
    )
    engine = EvolutionEngine(
        config=replace(base_config, initial_random_steps=0),
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_a", code="def solve(x):\n    return x + 1\n")])
    engine.step()

    assert engine.step_count == 0
    assert engine.stats.parse_failure_count == 1
    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert len(llm_calls) == 1
    assert llm_calls[0]["failure_type"] == "ParsingError"
    assert llm_calls[0]["failure_error"] == "No valid SEARCH/REPLACE edit blocks were found in the LLM response"
    assert llm_calls[0]["response"] == "still not a diff"


def test_engine_logs_llm_call_metadata(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient(
        [_diff_block("def solve(x):\n    return x + 1\n", "def solve(x):\n    return x + 8\n")]
    )
    engine = EvolutionEngine(
        config=replace(base_config),
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )
    engine.initialize([Candidate(id="seed_a", code="def solve(x):\n    return x + 1\n")])
    engine.step()

    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert llm_calls[-1]["child_ids"]
    assert llm_calls[-1]["step"] == 1


def test_engine_parallel_run_commits_consistent_archive_and_logs(
    base_config,
    seed_candidates,
    descriptor_fn,
    primary_fitness,
    primary_validation_fitness,
    secondary_fitness,
    secondary_validation_fitness,
):
    config = replace(
        base_config,
        parallel_workers=2,
        initial_random_steps=4,
        storage=replace(base_config.storage, checkpoint_interval=4, archive_snapshot_interval=2),
        secondary_eval=SecondaryEvalConfig(stages=1, p1=1.0, p2=0.5),
    )
    client = FakeLLMClient(
        [
            _diff_block("", "def solve(x):\n    return x + 10\n"),
            _diff_block("", "def solve(x):\n    return x + 11\n"),
            _diff_block("", "def solve(x):\n    return x - 12\n"),
            _diff_block("", "def solve(x):\n    return x + 13\n"),
        ],
        delay_seconds=0.01,
    )
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        primary_validation_fitness=primary_validation_fitness,
        secondary_fitness=secondary_fitness,
        secondary_validation_fitness=secondary_validation_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])
    engine.run(4)

    assert engine.step_count == 4
    assert client.calls == 4
    candidate_ids = list(engine.candidate_registry.keys())
    assert len(candidate_ids) == len(set(candidate_ids))

    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert len(llm_calls) == 4
    assert {call["step"] for call in llm_calls} == {1, 2, 3, 4}

    with (engine.run_store.run_dir / "candidates.jsonl").open("r", encoding="utf-8") as handle:
        candidate_lines = [json.loads(line) for line in handle]
    generated = [line for line in candidate_lines if line["id"].startswith("cand_")]
    assert generated
    assert all(line["record_version"] >= 1 for line in generated)

    snapshot_path = engine.run_store.run_dir / "archive_snapshots" / "step_000004.json"
    assert snapshot_path.exists()
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["step"] == 4
    assert "cell_elites" in snapshot


def test_engine_updates_parent_curiosity_from_insertions_and_misses(
    base_config,
    descriptor_fn,
    primary_validation_fitness,
):
    config = replace(
        base_config,
        descriptor_dim=1,
        num_centroids=1,
        cvt_samples=4,
        elites_per_cell=1,
        descriptor_labels=["descriptor_0"],
        parents_per_mutation=1,
        
        normalizer=NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=32),
        curiosity_default=1.0,
        curiosity_reward=1.5,
        curiosity_penalty=0.5,
    )
    seed_candidates = [Candidate(id="seed_parent", code="def solve(x):\n    return x\n")]

    def constant_descriptor(candidate):
        del candidate
        return [0.0]

    def primary_fitness(candidate):
        if "+ 10" in candidate.code:
            return 10.0
        if "- 1" in candidate.code:
            return 1.0
        return 2.0

    client = FakeLLMClient(
        [
            "\n\n".join(
                [
                    _diff_block("def solve(x):\n    return x\n", "def solve(x):\n    return x + 10\n"),
                    _diff_block("def solve(x):\n    return x\n", "def solve(x):\n    return x - 1\n"),
                ]
            )
        ]
    )
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        primary_validation_fitness=primary_validation_fitness,
        descriptor_fn=constant_descriptor,
    )

    engine.initialize(seed_candidates)
    engine.step()

    parent = engine.candidate_registry["seed_parent"]
    assert parent.curiosity_score == 0.75
    assert parent.curiosity_updates == 2

    with (engine.run_store.run_dir / "events.jsonl").open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle]
    curiosity_events = [event for event in events if event.get("type") == "parent_curiosity_updated"]
    assert len(curiosity_events) == 2
    assert curiosity_events[0]["multiplier"] in {1.5, 0.5}


def test_engine_llm_emitters_mode_uses_emitter_prompt_and_thinking(
    base_config,
    seed_candidates,
    descriptor_fn,
    primary_fitness,
):
    config = replace(
        base_config,
        generation_method="llm_emitters",
        llm=replace(base_config.llm, thinking_mode="disabled"),
        
        emitters=[
            EmitterConfig(
                name="runtime",
                extra_instructions="Emitter focus: improve runtime without changing the interface.",
                thinking_mode="enabled",
            )
        ],
    )
    client = FakeLLMClient([_seed_parent_diff("def solve(x):\n    return x + 9\n")])
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )

    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])
    engine.step()

    assert client.emitter_names == ["runtime"]
    assert client.thinking_modes == ["enabled"]
    assert "Emitter focus: improve runtime" in client.prompts[0]

    with (engine.run_store.run_dir / "llm_calls.jsonl").open("r", encoding="utf-8") as handle:
        llm_calls = [json.loads(line) for line in handle]
    assert llm_calls[-1]["emitter_name"] == "runtime"
    assert llm_calls[-1]["thinking_mode"] == "enabled"


def test_engine_prepare_step_includes_target_ancestors_and_inspirations(base_config, descriptor_fn, primary_fitness):
    config = replace(
        base_config,
        generation_method="standard",
        descriptor_dim=1,
        num_centroids=1,
        cvt_samples=4,
        elites_per_cell=3,
        descriptor_labels=["descriptor_0"],
        ancestor_count=1,
        inspiration_elite_count=2,
        elite_selection_strategy="curiosity_weighted",
        normalizer=NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=32),
    )
    target = Candidate(id="target", code="def solve(x):\n    return x * 5\n")
    local_inspiration = Candidate(id="local", code="def solve(x):\n    return x + 1\n")
    second_inspiration = Candidate(id="second", code="def solve(x):\n    return x - 3\n", primary_fitness=7.0)
    ancestor = Candidate(id="ancestor", code="def solve(x):\n    return x\n", primary_fitness=2.0)
    engine = EvolutionEngine(
        config=config,
        llm_client=FakeLLMClient([_diff_block("def solve(x):\n    return x * 5\n", "def solve(x):\n    return x + 9\n")]),
        primary_fitness=primary_fitness,
        descriptor_fn=lambda candidate: [0.0],
    )
    engine.initialize([target, local_inspiration, second_inspiration])
    engine.candidate_registry["target"].curiosity_score = 100.0
    engine.candidate_registry["local"].curiosity_score = 0.0
    engine.candidate_registry["second"].curiosity_score = 0.0
    engine.candidate_registry["target"].parent_ids = ["ancestor"]
    engine.candidate_registry["ancestor"] = ancestor
    prepared = engine.prepare_step()

    assert len(prepared.parents) == 1
    assert "Target Elite" in prepared.prompt
    assert "Ancestors:" in prepared.prompt
    assert "Inspirational Elites:" in prepared.prompt
    assert "target" in prepared.prompt
    assert "ancestor" in prepared.prompt
    assert "second" in prepared.prompt
    assert config.primary_metric_label in prepared.prompt


def test_engine_prepare_step_uses_emitter_specific_ancestor_and_inspiration_counts(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    config = replace(
        base_config,
        generation_method="llm_emitters",
        descriptor_dim=1,
        num_centroids=1,
        cvt_samples=4,
        elites_per_cell=4,
        descriptor_labels=["descriptor_0"],
        ancestor_count=1,
        inspiration_elite_count=2,
        emitters=[
            EmitterConfig(
                name="exploit",
                thinking_mode="enabled",
                ancestor_count=3,
                inspiration_elite_count=0,
            )
        ],
        normalizer=NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=32),
    )
    ancestor_3 = Candidate(id="ancestor_3", code="def solve(x):\n    return x - 3\n", primary_fitness=1.0)
    ancestor_2 = Candidate(id="ancestor_2", code="def solve(x):\n    return x - 2\n", primary_fitness=2.0, parent_ids=["ancestor_3"])
    ancestor_1 = Candidate(id="ancestor_1", code="def solve(x):\n    return x - 1\n", primary_fitness=3.0, parent_ids=["ancestor_2"])
    target = Candidate(id="target", code="def solve(x):\n    return x\n", primary_fitness=10.0, parent_ids=["ancestor_1"])
    local_inspiration = Candidate(id="local", code="def solve(x):\n    return x + 1\n", primary_fitness=5.0)
    engine = EvolutionEngine(
        config=config,
        llm_client=FakeLLMClient([_diff_block("def solve(x):\n    return x\n", "def solve(x):\n    return x + 9\n")]),
        primary_fitness=primary_fitness,
        descriptor_fn=lambda candidate: [0.0],
    )
    engine.initialize([target, local_inspiration])
    engine.candidate_registry["ancestor_1"] = ancestor_1
    engine.candidate_registry["ancestor_2"] = ancestor_2
    engine.candidate_registry["ancestor_3"] = ancestor_3
    engine.candidate_registry["target"].parent_ids = ["ancestor_1"]
    engine.candidate_registry["ancestor_1"].parent_ids = ["ancestor_2"]
    engine.candidate_registry["ancestor_2"].parent_ids = ["ancestor_3"]
    engine.candidate_registry["target"].curiosity_score = 100.0
    engine.candidate_registry["local"].curiosity_score = 0.0

    prepared = engine.prepare_step()

    assert prepared.emitter_name == "exploit"
    assert prepared.thinking_mode == "enabled"
    assert [candidate.id for candidate in prepared.ancestors] == ["ancestor_1", "ancestor_2", "ancestor_3"]
    assert prepared.inspirations == []
    assert "ancestor_1" in prepared.prompt
    assert "ancestor_2" in prepared.prompt
    assert "ancestor_3" in prepared.prompt
    assert "Inspirational Elites:\nNone" in prepared.prompt


def test_engine_llm_emitters_emitter_curiosity_updates_per_emitter_scores(
    base_config,
    descriptor_fn,
    primary_validation_fitness,
):
    config = replace(
        base_config,
        generation_method="llm_emitters_emitter_curiosity",
        
        parents_per_mutation=1,
        descriptor_dim=1,
        num_centroids=1,
        cvt_samples=4,
        elites_per_cell=2,
        descriptor_labels=["descriptor_0"],
        normalizer=NormalizerConfig(dim=1, lower_quantile=0.0, upper_quantile=1.0, history_size=32),
        emitters=[
            EmitterConfig(name="runtime", extra_instructions="Focus on runtime."),
            EmitterConfig(name="algorithm", extra_instructions="Focus on algorithm."),
        ],
        curiosity_default=1.0,
        curiosity_reward=1.5,
        curiosity_penalty=0.5,
    )
    seed_candidates = [Candidate(id="seed_parent", code="def solve(x):\n    return x\n")]

    def constant_descriptor(candidate):
        del candidate
        return [0.0]

    def primary_fitness(candidate):
        return 10.0 if "+ 10" in candidate.code else 1.0

    client = FakeLLMClient([_diff_block("def solve(x):\n    return x\n", "def solve(x):\n    return x + 10\n")])
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=primary_fitness,
        primary_validation_fitness=primary_validation_fitness,
        descriptor_fn=constant_descriptor,
    )

    engine.initialize(seed_candidates)
    engine.step()

    parent = engine.candidate_registry["seed_parent"]
    chosen_emitter = client.emitter_names[-1]
    assert chosen_emitter in {"runtime", "algorithm"}
    assert parent.curiosity_score == 1.5
    assert parent.emitter_curiosity_scores[chosen_emitter] == 1.5
    other_emitter = "algorithm" if chosen_emitter == "runtime" else "runtime"
    assert parent.emitter_curiosity_scores[other_emitter] == 1.0


def test_plateau_emitter_scheduler_shifts_toward_exploit_emitter(base_config):
    config = replace(
        base_config,
        generation_method="llm_emitters",
        emitter_selection_strategy="plateau_scheduler",
        emitters=[
            EmitterConfig(name="explore", selection_weight=1.0),
            EmitterConfig(name="exploit", selection_weight=1.0),
        ],
        emitter_plateau_scheduler=PlateauEmitterSchedulerConfig(
            target_emitter_name="exploit",
            patience_steps=5,
            target_weight_growth=2.0,
            non_target_weight_decay=0.5,
            min_non_target_weight_scale=0.1,
        ),
    )
    client = FakeLLMClient([_diff_block("def solve(x):\n    return x + 1\n", "def solve(x):\n    return x + 1\n")])
    engine = EvolutionEngine(
        config=config,
        llm_client=client,
        primary_fitness=lambda candidate: float(candidate.code.count("+")),
        descriptor_fn=lambda candidate: [float(len(candidate.code)), 1.0],
    )

    early_counts = {"explore": 0, "exploit": 0}
    for _ in range(400):
        early_counts[engine._choose_emitter().name] += 1

    engine.step_count = 25
    engine._best_primary_seen = 1.0
    engine._best_primary_improvement_step = 0

    late_counts = {"explore": 0, "exploit": 0}
    for _ in range(400):
        late_counts[engine._choose_emitter().name] += 1

    assert late_counts["exploit"] > early_counts["exploit"]
    assert late_counts["exploit"] > late_counts["explore"]


def test_prepared_prompts_include_unique_diversity_tokens(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient([_seed_parent_diff(_seed_parent_code())])
    engine = EvolutionEngine(
        config=base_config,
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )
    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])

    first = engine.prepare_step()
    second = engine.prepare_step()

    assert "Mutation diversity token:" in first.prompt
    assert "Mutation diversity token:" in second.prompt
    assert first.prompt != second.prompt


def test_identical_child_is_rejected_before_fitness(
    base_config,
    descriptor_fn,
    primary_fitness,
):
    client = FakeLLMClient([_seed_parent_diff(_seed_parent_code())])
    engine = EvolutionEngine(
        config=base_config,
        llm_client=client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )
    engine.initialize([Candidate(id="seed_parent", code=_seed_parent_code())])
    engine.step()

    events = [json.loads(line) for line in (engine.run_store.run_dir / "events.jsonl").read_text().splitlines()]
    assert any(
        event.get("type") == "candidate_rejected" and event.get("reason") == "identical_to_parent"
        for event in events
    )
