"""Runtime error feedback for LLM-guided evolution.

Collects errors from fitness evaluation, extracts the most common patterns,
and injects them into the mutation prompt as anti-examples so the LLM learns
what code mistakes to avoid.

Usage: automatic when EvolutionConfig.error_feedback_enabled=True (default).
The prompt template should contain {runtime_error_examples} where the examples
should appear. If absent, examples are appended to extra_instructions.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _normalise_error(error_str: str) -> str:
    """Strip numbers, file paths and variable names to produce a stable key."""
    s = str(error_str)
    # Drop file paths
    s = re.sub(r'/[^\s:]+\.py:\d+', '', s)
    # Drop hex addresses
    s = re.sub(r'0x[0-9a-fA-F]+', '0x...', s)
    # Drop quoted variable names that vary
    s = re.sub(r"'[a-z_][a-z0-9_]*'", "'<var>'", s)
    # Normalise numbers
    s = re.sub(r'\b\d+\b', 'N', s)
    return s.strip()[:120]


class ErrorFeedbackCollector:
    """Collects fitness-time errors and surfaces the most common patterns.

    Attributes
    ----------
    window : int
        Maximum number of recent errors to track.
    top_n : int
        Number of most-common errors to inject into the prompt.
    enabled : bool
        Whether collection and injection are active.
    """

    def __init__(self, window: int = 500, top_n: int = 5, enabled: bool = True):
        self.window = window
        self.top_n = top_n
        self.enabled = enabled
        self._errors: list[tuple[str, str]] = []   # (normalised_key, original_msg)
        self._key_to_example: dict[str, str] = {}

    def record(self, error: Any) -> None:
        if not self.enabled:
            return
        msg = str(error).strip()
        if not msg:
            return
        key = _normalise_error(msg)
        # Keep first-seen example for display
        if key not in self._key_to_example:
            self._key_to_example[key] = msg
        self._errors.append((key, msg))
        # Trim to window
        if len(self._errors) > self.window:
            oldest_key = self._errors[0][0]
            self._errors = self._errors[-self.window:]
            # Clean stale examples
            active = {k for k, _ in self._errors}
            self._key_to_example = {k: v for k, v in self._key_to_example.items() if k in active}

    def format_for_prompt(self) -> str:
        """Return a formatted block of the most common errors, ready for prompt injection."""
        if not self.enabled or not self._errors:
            return ""
        counts = Counter(k for k, _ in self._errors)
        top = counts.most_common(self.top_n)
        lines = [
            "COMMON RUNTIME ERRORS — the following errors have appeared frequently "
            "in recently evaluated candidates. Do NOT write code that causes these:\n"
        ]
        for i, (key, count) in enumerate(top, 1):
            example = self._key_to_example.get(key, key)
            # Truncate long messages
            display = example if len(example) <= 120 else example[:117] + "..."
            lines.append(f"  {i}. ({count}x) {display}")
        return "\n".join(lines)

    def inject_into_prompt(self, template: str, extra_instructions: str) -> tuple[str, str]:
        """Inject error examples into the prompt template or extra_instructions.

        If the template contains {runtime_error_examples}, replaces that placeholder.
        Otherwise prepends the examples to extra_instructions.

        Returns (modified_template, modified_extra_instructions).
        """
        if not self.enabled:
            return template, extra_instructions
        block = self.format_for_prompt()
        if not block:
            return template, extra_instructions
        if "{runtime_error_examples}" in template:
            return template.replace("{runtime_error_examples}", block), extra_instructions
        # Prepend to extra_instructions so it appears before format directives
        separator = "\n\n" if extra_instructions else ""
        return template, block + separator + extra_instructions
