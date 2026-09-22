"""Backend, prompt and calibration tests. Real prototype."""
from src.flashbulljev.backends import FakeBackend, OllamaBackend, get_backend
from src.flashbulljev.calibration import apply_temperature, brier, ece, fit_temperature, nll
from src.flashbulljev.prompts import build_boolean_prompt, build_mc_prompt, letters_for


def test_letters_list():
    ls = letters_for(3)
    assert ls == ["A", "B", "C"]
    assert isinstance(ls, list)


def test_build_mc_prompt():
    p = build_mc_prompt("state x", "choose?", ["a", "b"])
    assert "A) a" in p
    assert "B) b" in p
    assert "Answer with one letter" in p


def test_build_boolean():
    p = build_boolean_prompt("urgent payout", "urgent?")
    assert "A) Yes" in p


def test_fake_backend():
    be = FakeBackend()
    lg, lat = be.logits_for("prompt", 2)
    assert isinstance(lg, list) and len(lg) == 2
    assert lat == 0.0


def test_get_backend_fake():
    be = get_backend("fake")
    assert be.name == "fake"


def test_ollama_backend_init():
    be = OllamaBackend(model="qwen2.5:1.5b")
    assert be.model == "qwen2.5:1.5b"


def test_calibration_funcs():
    probs = [[0.8, 0.2], [0.3, 0.7]]
    labels = [0, 1]
    assert nll(probs, labels) > 0
    assert 0 <= brier(probs, labels) <= 1
    assert 0 <= ece(probs, labels) <= 1
    T, rep = fit_temperature([[2.0, 0.5], [0.2, 1.5]], labels)
    assert T in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    assert "nll" in rep


def test_apply_temp_list():
    out = apply_temperature([1.0, 2.0], 1.0)
    assert abs(sum(out) - 1.0) < 1e-6
