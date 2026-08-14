from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from evo_lib.config import NormalizerConfig
from evo_lib.errors import ConfigurationError


class DescriptorNormalizer:
    def __init__(self, config: NormalizerConfig):
        self.config = config
        self.descriptor_labels: list[str] | None = None
        self._history = [deque(maxlen=config.history_size) for _ in range(config.dim)]

    def update(self, descriptor_raw: list[float]) -> None:
        self._validate_descriptor(descriptor_raw)
        for index, value in enumerate(descriptor_raw):
            self._history[index].append(float(value))

    def normalize(self, descriptor_raw: list[float]) -> list[float]:
        self._validate_descriptor(descriptor_raw)
        lower, upper = self._bounds_arrays()
        normalized: list[float] = []
        for value, low, high in zip(descriptor_raw, lower, upper):
            if high <= low:
                norm_value = 0.5
            else:
                norm_value = (float(value) - low) / (high - low)
            if self.config.clip:
                norm_value = min(1.0, max(0.0, norm_value))
            normalized.append(float(norm_value))
        return normalized

    def bounds(self) -> dict[str, list[float]]:
        lower, upper = self._bounds_arrays()
        return {"lower": lower.tolist(), "upper": upper.tolist()}

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "dim": self.config.dim,
                "lower_quantile": self.config.lower_quantile,
                "upper_quantile": self.config.upper_quantile,
                "history_size": self.config.history_size,
                "clip": self.config.clip,
            },
            "descriptor_labels": self.descriptor_labels,
            "history": [list(values) for values in self._history],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        config = NormalizerConfig(**state["config"])
        self.config = config
        self.descriptor_labels = state.get("descriptor_labels")
        self._history = [deque(maxlen=config.history_size) for _ in range(config.dim)]
        history = state["history"]
        if len(history) != config.dim:
            raise ConfigurationError("Normalizer history dimension does not match config")
        for index, values in enumerate(history):
            self._history[index].extend(float(v) for v in values)

    def _validate_descriptor(self, descriptor_raw: list[float]) -> None:
        if len(descriptor_raw) != self.config.dim:
            raise ConfigurationError(
                f"Descriptor has dimension {len(descriptor_raw)} but expected {self.config.dim}"
            )

    def _bounds_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for values in self._history:
            if not values:
                raise ConfigurationError("DescriptorNormalizer has no history yet")
            array = np.asarray(values, dtype=float)
            lower.append(float(np.quantile(array, self.config.lower_quantile)))
            upper.append(float(np.quantile(array, self.config.upper_quantile)))
        return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
