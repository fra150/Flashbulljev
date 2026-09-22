"""Reaction battery tests: input/output contract + rapid-fire behavior."""
import pytest

from src.flashbulljev.backends import get_backend
from src.flashbulljev.engine import FlashBullJevEngine
from src.flashbulljev.pipeline import run_pipeline
from src.flashbulljev.reaction import (
    abstain_cases,
    case_label_index,
    check_case,
    check_contract,
    eval_metrics,
    is_abstained,
    parse_json_letter,
    reaction_cases,
    split_cases,
)


def test_battery_has_208_cases():
    cases = reaction_cases()
    assert len(cases) == 208
    ids = [c["id"] for c in cases]
    assert len(set(ids)) == 208
    for c in cases:
        assert "state" in c and "question" in c and "expect" in c
    groups = {}
    for c in cases:
        groups[c["id"][:1]] = groups.get(c["id"][:1], 0) + 1
    assert groups == {"u": 48, "s": 32, "d": 48, "f": 48, "n": 32}


def test_split_family_no_leakage():
    fit, ev = split_cases(reaction_cases())
    assert len(fit) + len(ev) == 208
    assert len(ev) == 16 + 8 + 16 + 16 + 8  # whole families, every 3rd
    fam = lambda c: c["id"][:3]
    assert not ({fam(c) for c in fit} & {fam(c) for c in ev})


def test_label_index_mapping():
    cases = {c["id"]: c for c in reaction_cases()}
    assert case_label_index(cases["u01"]) == 0
    assert case_label_index(cases["u07"]) == 1
    assert case_label_index(cases["d01"]) == 0  # billing first
    assert case_label_index(cases["d05"]) == 1  # technical second
    assert case_label_index(cases["d09"]) == 2  # sales third
    assert case_label_index(cases["f09"]) == 2
    assert case_label_index(cases["n01"]) is None


def test_eval_metrics_shape():
    m = eval_metrics([[2.0, 0.5], [0.2, 1.5]], [0, 1], 1.0)
    assert m["accuracy"] == 1.0
    assert m["n"] == 2
    assert 0.0 <= m["ece"] <= 1.0


def test_check_case_unit_logic():
    assert check_case({"noul": 0.9}, {"kind": "bool_yes"}) is True
    assert check_case({"noul": 0.9}, {"kind": "bool_no"}) is False
    assert check_case({"noul": None}, {"kind": "bool_yes"}) is False
    assert check_case({"choice": "billing"}, {"kind": "choice", "value": "billing"}) is True
    assert check_case({"choice": "sales"}, {"kind": "choice", "value": "billing"}) is False
    assert check_case({"probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}}, {"kind": "score", "value": 1}) is True
    assert check_case({"value": 74.0}, {"kind": "range", "low": 60.0, "high": 90.0}) is True
    assert check_case({"value": 10.0}, {"kind": "range", "low": 60.0, "high": 90.0}) is False


def test_check_contract_valid_and_broken():
    good = {"type": "noul", "noul": 0.7, "status": "ok", "confidence": 0.4,
            "probabilities": {"a": 0.6, "b": 0.4}}
    assert check_contract(good) == []
    bad = {"type": "x", "confidence": 2.0, "probabilities": {"a": 0.5, "b": 0.2}}
    problems = check_contract(bad)
    assert "missing status" in problems
    assert any("confidence" in p for p in problems)
    assert any("sum" in p for p in problems)


def test_fake_battery_contract_all_ok():
    eng = FlashBullJevEngine(backend=get_backend("fake"))
    bad = 0
    for case in reaction_cases():
        res = run_pipeline(eng, case["state"], {"q": case["question"]})
        ans = res.answers["q"]
        if check_contract(ans):
            bad += 1
    assert bad == 0


def test_abstain_cases_shape():
    cases = abstain_cases()
    assert len(cases) == 12
    assert is_abstained({"type": "noul", "noul": None, "status": "insufficient_evidence", "confidence": 0.0}) is True
    assert is_abstained({"type": "choice", "choice": None, "status": "uncertain", "confidence": 0.0}) is True
    assert is_abstained({"type": "noul", "noul": 0.9, "status": "ok", "confidence": 0.8}) is False


def test_parse_json_letter():
    assert parse_json_letter('{"answer": "B"}', ["A", "B", "C"]) == "B"
    assert parse_json_letter("Answer: c", ["A", "B", "C"]) == "C"
    assert parse_json_letter("no letters here 123", ["A", "B"]) is None


def test_live_ollama_reaction_shape():
    try:
        eng = FlashBullJevEngine(backend=get_backend("ollama"))
        res = run_pipeline(eng, "URGENT: server down!", {"q": {"type": "noul", "instructions": "Does this convey urgency?"}})
    except (OSError, ValueError, RuntimeError, ConnectionError) as e:
        pytest.skip(f"ollama unreachable: {e}")
        return
    ans = res.answers["q"]
    assert check_contract(ans) == []
    assert ans.get("backend") == "ollama"
