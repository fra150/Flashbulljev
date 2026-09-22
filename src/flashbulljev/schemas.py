"""Typed schemas in the System One style. Clean code, pure functions, lists."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


VALID_TYPES = ["noul", "boolean", "choice", "score", "numeric"]


@dataclass
class BooleanQuestion:
    """Yes/no question."""

    instructions: str
    true_desc: str = "Yes"
    false_desc: str = "No"
    allow_abstain: bool = True


@dataclass
class ChoiceQuestion:
    """Choice among named options. Max 26 slots (A-Z)."""

    instructions: str
    options: List[str] = field(default_factory=list)
    allow_abstain: bool = True

    def option_ids(self) -> List[str]:
        """Return a clean list of option ids."""
        return [str(o) for o in self.options]


@dataclass
class ScoreQuestion:
    """Score over ordered levels, low -> high."""

    instructions: str
    levels: List[str] = field(default_factory=list)
    allow_abstain: bool = True


@dataclass
class NumericQuestion:
    """Numeric answer over increasing anchors."""

    instructions: str
    anchors: List[float] = field(default_factory=list)
    descriptions: List[str] = field(default_factory=list)
    unit: str = ""
    allow_abstain: bool = True


def parse_questions(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a raw dict into typed objects. Pure function using lists.

    Raw example:
        {"is_urgent": {"type": "noul", "instructions": "..."}}
    """
    out: Dict[str, Any] = {}
    keys: List[str] = list(raw.keys())
    for k in keys:
        spec: Dict[str, Any] = dict(raw[k] or {})
        t: str = str(spec.get("type", "noul")).lower()
        instructions: str = str(spec.get("instructions", ""))
        allow_abstain: bool = bool(spec.get("allow_abstain", True))
        if t in ("noul", "boolean"):
            out[k] = BooleanQuestion(
                instructions=instructions,
                true_desc=str((spec.get("true") or "Yes")),
                false_desc=str((spec.get("false") or "No")),
                allow_abstain=allow_abstain,
            )
        elif t == "choice":
            crit = spec.get("criteria", {}) or spec.get("options", {})
            options: List[str] = build_option_list(crit)
            out[k] = ChoiceQuestion(
                instructions=instructions,
                options=options,
                allow_abstain=allow_abstain,
            )
        elif t == "score":
            levels: List[str] = build_level_list(spec.get("criteria", []))
            out[k] = ScoreQuestion(
                instructions=instructions,
                levels=levels,
                allow_abstain=allow_abstain,
            )
        elif t == "numeric":
            anchors: List[float] = []
            descs: List[str] = []
            for a in list(spec.get("anchors", []) or []):
                if isinstance(a, dict):
                    anchors.append(float(a.get("value", 0.0)))
                    descs.append(str(a.get("description", "")))
                else:
                    anchors.append(float(a))
                    descs.append("")
            out[k] = NumericQuestion(
                instructions=instructions,
                anchors=anchors,
                descriptions=descs,
                unit=str(spec.get("unit", "")),
                allow_abstain=allow_abstain,
            )
        else:
            out[k] = BooleanQuestion(instructions=instructions)
    return out


def build_option_list(criteria: Any) -> List[str]:
    """Build an option list from a dict | list. Max 26 items."""
    options: List[str] = []
    if isinstance(criteria, dict):
        keys: List[str] = list(criteria.keys())
        for k in keys:
            options.append(str(k))
    elif isinstance(criteria, list):
        for item in criteria:
            options.append(str(item))
    return options[:26]


def build_level_list(criteria: Any) -> List[str]:
    """Build an ordered level list."""
    levels: List[str] = []
    items: List[Any] = list(criteria) if isinstance(criteria, list) else []
    for item in items:
        levels.append(str(item))
    return levels[:10]


def list_question_keys(questions: Dict[str, Any]) -> List[str]:
    """Return the list of question keys."""
    return list(questions.keys())


def count_slots(question: Any) -> int:
    """Count the answer slots used (max 26 with abstention)."""
    base = 2
    if isinstance(question, ChoiceQuestion):
        base = len(question.option_ids())
    elif isinstance(question, ScoreQuestion):
        base = len(question.levels)
    elif isinstance(question, NumericQuestion):
        base = len(question.anchors)
    extra = 1 if getattr(question, "allow_abstain", False) else 0
    return base + extra
