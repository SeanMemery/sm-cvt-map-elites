from __future__ import annotations

import json

import numpy as np

from evo_lib.archive import CVTArchive
from evo_lib.candidate import Candidate
from evo_lib.config import NormalizerConfig
from evo_lib.normalizer import DescriptorNormalizer
from evo_lib.storage import RunStore


def test_run_store_writes_required_files(base_config, tmp_path):
    registry = {}
    archive = CVTArchive(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float), registry, np.random.default_rng(1))
    normalizer = DescriptorNormalizer(NormalizerConfig(dim=2, lower_quantile=0.0, upper_quantile=1.0, history_size=8))
    normalizer.update([0.0, 0.0])
    normalizer.update([1.0, 1.0])
    candidate = Candidate(
        id="cand_000001",
        code="def solve():\n    return 1\n",
        descriptor_raw=[1.0, 1.0],
        descriptor_norm=normalizer.normalize([1.0, 1.0]),
        primary_fitness=1.0,
        curiosity_score=2.5,
        curiosity_updates=3,
        cell_id=1,
    )
    registry[candidate.id] = candidate
    archive.insert(candidate, candidate.descriptor_norm, candidate.primary_fitness)
    normalizer.descriptor_labels = ["length", "returns"]

    store = RunStore(base_config.storage, base_config.run_name, base_config.output_dir)
    store.initialize_run(base_config, archive.centroids)
    store.save_candidate(candidate)
    store.log_event("candidate_inserted", {"step": 0, "candidate_id": candidate.id})
    store.log_llm_call(0, [], "llm_000001", "prompt", "response", {"model": "x", "child_ids": [candidate.id]})
    store.save_archive_snapshot(0, archive, normalizer)
    store.save_stats({"step": 0, "archive_occupancy": 1})
    store.save_checkpoint({"value": 3}, step=0)

    assert (store.run_dir / "config.json").exists()
    assert (store.run_dir / "metadata.json").exists()
    assert (store.run_dir / "candidates.jsonl").exists()
    assert (store.run_dir / "events.jsonl").exists()
    assert (store.run_dir / "llm_calls.jsonl").exists()
    assert (store.run_dir / "stats.jsonl").exists()
    assert (store.run_dir / "centroids.npy").exists()
    assert (store.run_dir / "archive_snapshots" / "step_000000.json").exists()
    assert (store.run_dir / "checkpoints" / "latest.pkl").exists()

    with (store.run_dir / "config.json").open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert config_payload["primary_metric_label"] == "Visible Score"
    assert config_payload["primary_validation_metric_label"] == "Visible Score Validation"
    assert config_payload["secondary_metric_label"] == "Hidden Score"
    assert config_payload["secondary_validation_metric_label"] == "Hidden Score Validation"

    with (store.run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["primary_metric_label"] == "Visible Score"
    assert metadata["primary_validation_metric_label"] == "Visible Score Validation"
    assert metadata["secondary_metric_label"] == "Hidden Score"
    assert metadata["secondary_validation_metric_label"] == "Hidden Score Validation"

    with (store.run_dir / "archive_snapshots" / "step_000000.json").open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["occupied_cells"][0]["candidate_id"] == "cand_000001"
    assert snapshot["occupied_cells"][0]["curiosity_score"] == 2.5
    assert snapshot["cell_candidates"][0]["candidates"][0]["candidate_id"] == "cand_000001"
    assert snapshot["cell_candidates"][0]["candidates"][0]["is_elite"] is True
    assert snapshot["descriptor_labels"] == ["length", "returns"]
