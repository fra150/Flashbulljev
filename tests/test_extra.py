"""Extra coverage: schemas, GF no-save rule, calibration I/O, factory, pipeline."""
import numpy as np

from src.flashbulljev.backends import get_backend
from src.flashbulljev.calibration import load_temperature, save_fit
from src.flashbulljev.engine import FlashBullJevEngine
from src.flashbulljev.memory_gf import certify_fragment
from src.flashbulljev.pipeline import batch_states, certify_step, run_pipeline
from src.flashbulljev.schemas import (
    count_slots,
    list_question_keys,
    parse_questions,
)


def test_parse_all_types():
    raw = {
        "b": {"type": "noul", "instructions": "x?"},
        "c": {"type": "choice", "instructions": "y?", "criteria": {"a": "1", "b": "2"}},
        "s": {"type": "score", "instructions": "z?", "criteria": ["low", "high"]},
        "n": {"type": "numeric", "instructions": "w?", "anchors": [{"value": 0, "description": "e"}, {"value": 5, "description": "f"}], "unit": "pts"},
        "u": {"type": "unknown-type", "instructions": "fallback"},
    }
    parsed = parse_questions(raw)
    assert set(parsed.keys()) == {"b", "c", "s", "n", "u"}
    assert list_question_keys(parsed) == list(raw.keys())


def test_choice_slots_cap_26():
    parsed = parse_questions({"c": {"type": "choice", "instructions": "x?", "criteria": [f"o{i}" for i in range(40)]}})
    assert count_slots(parsed["c"]) == 26 + 1  # 26 options + abstain
    parsed2 = parse_questions({"b": {"type": "noul", "instructions": "x?", "allow_abstain": False}})
    assert count_slots(parsed2["b"]) == 2


def test_uncertified_never_saved():
    eng = FlashBullJevEngine()
    # force an uncertifiable state by monkeypatching certify to fail once
    import src.flashbulljev.engine as engmod

    orig = engmod.certify_fragment
    engmod.certify_fragment = lambda *a, **k: {"certified": False, "certificato": False, "quality": 0.0,
                                               "quiescence": {"active": False}, "reason": "forced", "motivo": "forced",
                                               "novelty_rel": 0.0, "novita_rel": 0.0}
    try:
        r1 = eng.decide("never certify this", {"q": {"type": "noul", "instructions": "x?"}})
        assert r1["cache"]["hit"] is False
        assert eng.memory.stats()["saves"] == 0
        r2 = eng.decide("never certify this", {"q": {"type": "noul", "instructions": "x?"}})
        assert r2["cache"]["hit"] is False  # still miss: nothing was stored
    finally:
        engmod.certify_fragment = orig


def test_certify_step_cache_hit():
    assert certify_step({"cache": {"hit": True}, "gf": {}}) is True
    assert certify_step({"cache": {"hit": False}, "gf": {"certified": True}}) is True
    assert certify_step({"cache": {"hit": False}, "gf": {"certified": False, "certificato": False}}) is False


def test_calibration_save_load_roundtrip(tmp_path):
    p = str(tmp_path / "fit.json")
    save_fit(p, 1.5, {"nll": 0.5, "n": 2})
    assert load_temperature(p) == 1.5
    assert load_temperature(str(tmp_path / "missing.json")) == 1.0


def test_backend_factory_names():
    assert get_backend("fake").name == "fake"
    assert get_backend("ollama").name == "ollama"
    assert get_backend("OLLAMA").name == "ollama"


def test_batch_and_pipeline_consistency():
    eng = FlashBullJevEngine()
    states = batch_states(["s1", "s2"])
    assert [d["state"] for d in states] == ["s1", "s2"]
    res = run_pipeline(eng, {"state": "dict-state"}, {"q": {"type": "score", "instructions": "s?", "criteria": ["a", "b", "c"]}})
    assert "score" in res.answers["q"]
    assert 0.0 <= res.answers["q"]["normalized_score"] <= 1.0


def test_numeric_answer_shape():
    eng = FlashBullJevEngine()
    res = run_pipeline(eng, "fill 75 percent", {"fill": {"type": "numeric", "instructions": "fill?",
                                                         "anchors": [{"value": 0, "description": "empty"},
                                                                     {"value": 50, "description": "half"},
                                                                     {"value": 100, "description": "full"}],
                                                         "unit": "percent"}})
    ans = res.answers["fill"]
    assert ans["unit"] == "percent"
    assert 0.0 <= ans["value"] <= 100.0
    assert abs(sum(ans["probabilities"].values()) - 1.0) < 1e-6


def test_certify_fragment_keys_bilingual():
    c = certify_fragment(np.ones(6), np.ones(6), np.zeros(6), np.ones(6))
    assert c["certified"] == c["certificato"]
    assert c["quality"] == c["qualita"]
