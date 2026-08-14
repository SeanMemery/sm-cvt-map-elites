from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = REPO_ROOT.parents[1]
OPTICAL_FLOW_ROOT = WORK_ROOT / "projects" / "optical-flow-learning"
VIEWER_DATA_ROOT = REPO_ROOT / "viewer_data" / "agent-hub"


def is_run_dir(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "metadata.json").exists()
        and (path / "config.json").exists()
        and (path / "candidates.jsonl").exists()
        and (path / "archive_snapshots").is_dir()
    )


def discover_runs(root: Path, run_name_glob: str | None = None) -> list[Path]:
    if not root.exists():
        return []
    runs = [path for path in root.rglob("*") if is_run_dir(path)]
    if run_name_glob is not None:
        runs = [path for path in runs if path.name.startswith(run_name_glob)]
    return sorted(runs)


def write_registry(worker_id: str, runs: list[Path]) -> Path:
    registry_path = VIEWER_DATA_ROOT / worker_id / "imported_runs.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{"run_id": run.name, "path": str(run)} for run in runs]
    registry_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return registry_path


def build_registry(worker_id: str) -> tuple[Path, int]:
    if worker_id == "ball-bouncing-experiment":
        runs = discover_runs(
            OPTICAL_FLOW_ROOT / "validation" / "island_jobs",
            run_name_glob="ball_bounce_emitter_curiosity_",
        )
    elif worker_id == "ball-bouncing-plus-coordinates-experiment":
        runs = discover_runs(
            OPTICAL_FLOW_ROOT / "validation" / "island_jobs",
            run_name_glob="ball_bounce_with_positions_emitter_curiosity_",
        )
    elif worker_id == "cvt-map-elites":
        runs = discover_runs(REPO_ROOT / "validation")
    else:
        raise SystemExit(f"Unsupported worker id: {worker_id}")
    registry_path = write_registry(worker_id, runs)
    return registry_path, len(runs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-worker viewer registries for Agent Hub runs.")
    parser.add_argument(
        "--worker",
        action="append",
        choices=[
            "ball-bouncing-experiment",
            "ball-bouncing-plus-coordinates-experiment",
            "cvt-map-elites",
        ],
        help="Worker registry to build. Defaults to all workers.",
    )
    args = parser.parse_args()
    workers = args.worker or [
        "ball-bouncing-experiment",
        "ball-bouncing-plus-coordinates-experiment",
        "cvt-map-elites",
    ]
    for worker_id in workers:
        registry_path, run_count = build_registry(worker_id)
        print(f"{worker_id}: {run_count} runs -> {registry_path}")


if __name__ == "__main__":
    main()
