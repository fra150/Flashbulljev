"""FragmentMemory tests: ~1ms hit, recompute on miss, functions and lists."""
from src.flashbulljev.memory_gf import FragmentMemory, MemoriaGF, certify_fragment, state_to_vector, verify_quiescence


def test_state_to_vector_list():
    vec = state_to_vector([1, 2, 3])
    assert list(vec) == [1.0, 2.0, 3.0]


def test_state_to_vector_dict():
    vec = state_to_vector({"b": 2, "a": 1})
    assert len(list(vec)) == 2


def test_verify_quiescence_active():
    import numpy as np

    base = np.ones(8)
    q = verify_quiescence(base, base * 0.99, base)
    assert q["active"] is True
    assert q["attivo"] is True  # legacy alias
    assert "err_fx_fo" in q


def test_certify_never_without_quiescence():
    import numpy as np

    fo = np.ones(8)
    fx = np.ones(8) * 10.0
    fy = np.zeros(8)
    c = certify_fragment(fo, fx, fy, fo)
    assert c["certified"] is False
    assert c["certificato"] is False  # legacy alias


def test_memory_hit_miss():
    mem = FragmentMemory()
    k = mem.key("test state", ["q1", "q2"])
    hit, _ = mem.recall(k)
    assert hit is False
    mem.save(k, {"answers": {}})
    hit2, payload = mem.recall(k)
    assert hit2 is True
    assert payload == {"answers": {}}
    stats = mem.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["saves"] == 1


def test_legacy_aliases():
    assert MemoriaGF is FragmentMemory
    mem = MemoriaGF()
    k = mem.chiave("legacy state", ["q1"])
    mem.salva(k, {"answers": {}})
    hit, _ = mem.richiama(k)
    assert hit is True
