"""flashbulljev engine tests: functions, arrays, lists, fast+stored cache."""
from src.flashbulljev.engine import (
    confidence_from_probs,
    decide_many,
    fake_logits,
    softmax,
)
from src.flashbulljev.engine import FlashBullJevEngine


def test_softmax_list():
    probs = softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-6
    assert isinstance(probs, list)


def test_confidence():
    c = confidence_from_probs([0.9, 0.05, 0.05])
    assert 0.0 <= c <= 1.0


def test_fake_logits_array_len():
    lg = fake_logits("state", "q", 4)
    assert isinstance(lg, list)
    assert len(lg) == 4


def test_decide_many_keys():
    qs = {
        "is_urgent": {"type": "noul", "instructions": "urgent?"},
        "dept": {"type": "choice", "instructions": "team?", "criteria": {"a": "x", "b": "y"}},
    }
    ans = decide_many("payout failing", qs)
    assert list(sorted(ans.keys())) == ["dept", "is_urgent"]


def test_engine_miss_then_hit():
    eng = FlashBullJevEngine()
    qs = {"is_urgent": {"type": "noul", "instructions": "urgent?"}}
    r1 = eng.decide("Help payouts failing 3 days", qs)
    assert r1["cache"]["hit"] is False
    r2 = eng.decide("Help payouts failing 3 days", qs)
    # second round must be a ~1ms hit
    assert r2["cache"]["hit"] is True
    assert r2["cache"]["latency_ms"] < r1["cache"]["latency_ms"]
    assert r2["usage"]["output_tokens"] == 0
