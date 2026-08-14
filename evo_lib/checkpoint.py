from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from evo_lib.errors import CheckpointError


def save_checkpoint(path: Path, engine_state: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(engine_state, handle)
    except OSError as exc:
        raise CheckpointError(f"Failed to save checkpoint to {path}: {exc}") from exc


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            state = pickle.load(handle)
    except OSError as exc:
        raise CheckpointError(f"Failed to load checkpoint from {path}: {exc}") from exc
    except pickle.PickleError as exc:
        raise CheckpointError(f"Checkpoint at {path} is invalid: {exc}") from exc
    if not isinstance(state, dict):
        raise CheckpointError("Checkpoint payload must be a dict")
    return state
