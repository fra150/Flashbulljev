"""SQLite persistence tests for FragmentMemory."""
import os

from src.flashbulljev.engine import FlashBullJevEngine
from src.flashbulljev.memory_gf import FragmentMemory


def test_attach_save_reload_roundtrip(tmp_path):
    db = str(tmp_path / "cache.db")
    mem = FragmentMemory()
    assert mem.attach_db(db) == 0
    k = mem.key("persistent state", ["q1"])
    mem.save(k, {"answers": {"q": 1}})
    mem2 = FragmentMemory()
    assert mem2.attach_db(db) == 1
    hit, payload = mem2.recall(k)
    assert hit is True
    assert payload == {"answers": {"q": 1}}


def test_engine_attaches_db_from_env(tmp_path, monkeypatch):
    db = str(tmp_path / "eng.db")
    monkeypatch.setenv("FLASHBULLJEV_CACHE_DB", db)
    eng = FlashBullJevEngine()
    r1 = eng.decide("remember me", {"q": {"type": "noul", "instructions": "x?"}})
    assert r1["cache"]["hit"] is False
    eng2 = FlashBullJevEngine()  # new process-like instance, same db file
    r2 = eng2.decide("remember me", {"q": {"type": "noul", "instructions": "x?"}})
    assert r2["cache"]["hit"] is True  # hit across restarts
    assert os.path.getsize(db) > 0
