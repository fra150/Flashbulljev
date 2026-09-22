"""Short prompt builder with plain-text multiple choice.

The state is prefilled once, questions run in parallel, letters A-Z only.
Pure functions, lists.
"""
from __future__ import annotations

from typing import List

LETTERS: List[str] = [chr(65 + i) for i in range(26)]

SYSTEM_SHORT = "You are a fast decision function. Answer with ONE letter only."


def letters_for(n: int) -> List[str]:
    """First n letters of A-Z."""
    return LETTERS[: max(0, min(n, 26))]


def build_mc_prompt(state: object, instructions: str, options: List[str]) -> str:
    """Build a plain-text multiple-choice prompt. Pure function."""
    opts: List[str] = list(options)
    lets: List[str] = letters_for(len(opts))
    lines: List[str] = [SYSTEM_SHORT, "", f"State: {state}", "", f"Question: {instructions}", ""]
    for letter, opt in zip(lets, opts):
        lines.append(f"{letter}) {opt}")
    lines.append("Answer with one letter:")
    return "\n".join(lines)


def build_boolean_prompt(state: object, instructions: str) -> str:
    """Boolean question as multiple choice A=Yes B=No."""
    return build_mc_prompt(state, instructions, ["Yes", "No"])


def build_score_prompt(state: object, instructions: str, levels: List[str]) -> str:
    """Score question as multiple choice over levels."""
    return build_mc_prompt(state, instructions, list(levels))


def build_numeric_prompt(state: object, instructions: str, anchors: List[float], descs: List[str]) -> str:
    """Numeric question as multiple choice over described anchors."""
    opts: List[str] = []
    descs_padded: List[str] = list(descs) + [""] * max(0, len(anchors) - len(descs))
    for anchor, desc in zip(list(anchors), descs_padded):
        opts.append(f"{anchor} ({desc})" if desc else str(anchor))
    return build_mc_prompt(state, instructions, opts)
