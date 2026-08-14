from __future__ import annotations

from evo_lib.engine import EvolutionEngine
from tests.conftest import FakeLLMClient


def _diff(search: str, replace: str) -> str:
    return (
        "```python\n"
        "<<<<<<< SEARCH\n"
        f"{search.rstrip()}\n"
        "=======\n"
        f"{replace.rstrip()}\n"
        ">>>>>>> REPLACE\n"
        "```"
    )


def test_checkpoint_resume_continue_and_fork(base_config, seed_candidates, descriptor_fn, primary_fitness):
    initial_client = FakeLLMClient(
        [
            _diff("def solve(x):\n    return x * 2\n", "def solve(x):\n    return x + 3\n"),
            _diff("def solve(x):\n    return x * 2\n", "def solve(x):\n    return x + 4\n"),
            _diff("def solve(x):\n    return x * 2\n", "def solve(x):\n    return x + 5\n"),
        ]
    )
    engine = EvolutionEngine(
        config=base_config,
        llm_client=initial_client,
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
    )
    engine.initialize([seed_candidates[1]])
    engine.step()
    engine.step()

    checkpoint_path = engine.run_store.run_dir / "checkpoints" / "latest.pkl"

    continue_client = FakeLLMClient([_diff("def solve(x):\n    return x * 2\n", "def solve(x):\n    return x + 6\n")])
    resumed = EvolutionEngine.load_from_checkpoint(
        str(checkpoint_path),
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
        llm_client=continue_client,
        resume_mode="continue",
    )
    resumed.step()
    assert resumed.step_count == 3
    assert resumed.run_store.run_dir == engine.run_store.run_dir

    fork_client = FakeLLMClient([_diff("def solve(x):\n    return x * 2\n", "def solve(x):\n    return x + 7\n")])
    forked = EvolutionEngine.load_from_checkpoint(
        str(checkpoint_path),
        primary_fitness=primary_fitness,
        descriptor_fn=descriptor_fn,
        llm_client=fork_client,
        resume_mode="fork",
    )
    forked.step()

    assert forked.step_count == 3
    assert forked.run_store.run_dir != engine.run_store.run_dir
