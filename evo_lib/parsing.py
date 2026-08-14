from __future__ import annotations

import re

from evo_lib.candidate import Candidate
from evo_lib.config import ParsingConfig
from evo_lib.errors import ParsingError


# Greedy match so nested ``` in JSDoc comments don't truncate the outer block.
# Captures from the opening fence to the LAST closing ``` on its own line.
CODE_BLOCK_PATTERN = re.compile(r"```[^\n`]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
# Matches an opening fence with no closing fence — captures everything to end of string
OPEN_CODE_BLOCK_PATTERN = re.compile(r"```[^\n`]*\n(.*)", re.DOTALL)
SEARCH_REPLACE_PATTERN = re.compile(
    r"^(?P<indent>[ \t]*)<<<<<<< SEARCH[ \t]*\n"
    r"(?P<search>.*?)"
    r"^(?P=indent)=======[ \t]*\n"
    r"(?P<replace>.*?)"
    r"^(?P=indent)>>>>>>> REPLACE[ \t]*(?:\n|$)",
    re.MULTILINE | re.DOTALL,
)


def parse_full_candidates(response: str, config: ParsingConfig) -> list[Candidate]:
    """Parse complete candidate code from fenced blocks in an LLM response.

    Used with output_format='full' — the LLM returns complete programs, not diffs.
    Each fenced code block (any language marker) becomes one candidate.

    When config.raw_response is True, the entire response is used as-is without
    any block extraction. This is useful for multi-block formats (e.g. a file
    containing both a TypeScript block and a Python block).

    Tolerates a missing closing fence: if the response was truncated mid-generation,
    the content from the opening fence to end-of-string is still extracted.
    Falls back to the whole response when allow_plaintext_fallback is True.
    """
    normalized = _normalize_response_text(response)

    # Raw mode: store the entire response without block extraction
    if config.raw_response:
        text = normalized.strip()
        if text:
            return [Candidate(id="", code=text, metadata={"edit_format": "raw"})]
        raise ParsingError("Empty response in raw_response mode.")

    # Prefer complete blocks (opening + closing fence)
    blocks = [match.strip() for match in CODE_BLOCK_PATTERN.findall(normalized) if match.strip()]

    # If none found, try an unclosed opening fence (truncated response)
    if not blocks:
        open_match = OPEN_CODE_BLOCK_PATTERN.search(normalized)
        if open_match:
            content = open_match.group(1).strip()
            if content:
                blocks = [content]

    candidates: list[Candidate] = []
    for block in blocks[: config.max_candidates_per_response]:
        candidates.append(Candidate(id="", code=block, metadata={"edit_format": "full"}))
    if candidates:
        return candidates
    if config.allow_plaintext_fallback:
        text = normalized.strip()
        if text:
            return [Candidate(id="", code=text, metadata={"edit_format": "full_plaintext"})]
    raise ParsingError(
        "No fenced code blocks were found in the LLM response. "
        "The model must return each candidate inside a fenced code block (e.g. ```python ... ``` or ```typescript ... ```)."
    )


def parse_edit_candidates(response: str, base_code: str, config: ParsingConfig) -> list[Candidate]:
    diff_candidates = _extract_search_replace_candidates(_normalize_response_text(response), base_code, config)
    if diff_candidates:
        return diff_candidates
    raise ParsingError("No valid SEARCH/REPLACE edit blocks were found in the LLM response")


def apply_edit_script(base_code: str, edits: list[object]) -> str:
    updated = base_code
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise ParsingError(f"Edit {index} must be an object")
        search = edit.get("search")
        replace = edit.get("replace")
        if not isinstance(search, str) or not isinstance(replace, str):
            raise ParsingError(f"Edit {index} must contain string search and replace fields")
        if search == "__FULL_FILE__":
            updated = replace
            continue
        if search == "" and updated == "":
            updated = replace
            continue
        occurrence_count = updated.count(search)
        if occurrence_count != 1:
            if _should_treat_as_implicit_full_rewrite(
                base_code=updated,
                search=search,
                replace=replace,
                edit_index=index,
            ):
                updated = replace
                continue
            raise ParsingError(
                f"Edit {index} search text must match exactly once in the target code; found {occurrence_count} matches"
            )
        updated = updated.replace(search, replace, 1)
    return updated


def _extract_search_replace_candidates(response: str, base_code: str, config: ParsingConfig) -> list[Candidate]:
    blocks = [match.strip() for match in CODE_BLOCK_PATTERN.findall(response) if match.strip()]
    candidate_chunks = [block for block in blocks if SEARCH_REPLACE_PATTERN.search(block)]
    if not candidate_chunks and SEARCH_REPLACE_PATTERN.search(response):
        candidate_chunks = [response.strip()]
    candidates: list[Candidate] = []
    for chunk in candidate_chunks[: config.max_candidates_per_response]:
        edits = _extract_search_replace_edits(chunk)
        updated_code = apply_edit_script(base_code, edits)
        candidates.append(
            Candidate(
                id="",
                code=updated_code,
                metadata={
                    "edit_count": len(edits),
                    "edit_format": "search_replace",
                },
            )
        )
    return candidates


def _extract_search_replace_edits(text: str) -> list[dict[str, str]]:
    edits: list[dict[str, str]] = []
    for match in SEARCH_REPLACE_PATTERN.finditer(text):
        indent = match.group("indent")
        edits.append(
            {
                "search": _finalize_edit_text(_strip_marker_indent(match.group("search"), indent)),
                "replace": _finalize_edit_text(_strip_marker_indent(match.group("replace"), indent)),
            }
        )
    if not edits:
        raise ParsingError("No SEARCH/REPLACE edit blocks were found in the candidate response")
    return edits


def _normalize_response_text(response: str) -> str:
    return response.replace("\r\n", "\n").replace("\r", "\n")


def _strip_marker_indent(text: str, indent: str) -> str:
    if not indent:
        return text
    return "\n".join(_strip_line_prefix(line, indent) for line in text.split("\n"))


def _strip_line_prefix(line: str, prefix: str) -> str:
    if not line.strip():
        return ""
    if line.startswith(prefix):
        return line[len(prefix) :]
    return line


def _finalize_edit_text(text: str) -> str:
    return text.removesuffix("\n")


def _should_treat_as_implicit_full_rewrite(*, base_code: str, search: str, replace: str, edit_index: int) -> bool:
    if edit_index != 0:
        return False
    if not base_code.strip() or len(base_code) > 256:
        return False
    if len(replace.strip()) <= len(base_code.strip()):
        return False
    if len(search.strip()) <= len(base_code.strip()):
        return False
    if "\n" not in replace or "\n" not in search:
        return False
    return True
