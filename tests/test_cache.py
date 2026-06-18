"""The build-time LLM cache that makes the wiki rebuild incremental + deterministic.

`cached_chat` is content-addressed (key = model + messages + params), so an unchanged sheet
re-hits its cached parse/prose (no network, identical output) while an edited one misses and
recomputes. These tests pin that contract offline by monkeypatching the network `chat`.
"""

from __future__ import annotations

import pytest

from src.stella_kb import llm


@pytest.fixture
def counting_chat(monkeypatch):
    """Replace the network call with a deterministic counter so we can see hits vs misses."""
    calls = {"n": 0}

    def fake_chat(messages, temperature=0.0, max_tokens=512, timeout=60.0):
        calls["n"] += 1
        return f"reply-{calls['n']}"

    monkeypatch.setattr(llm, "chat", fake_chat)
    return calls


MSG = [{"role": "user", "content": "hello"}]


def test_miss_then_hit_skips_the_network(tmp_path, counting_chat):
    a = llm.cached_chat(MSG, cache_dir=str(tmp_path))
    assert counting_chat["n"] == 1                       # first call computes
    b = llm.cached_chat(MSG, cache_dir=str(tmp_path))
    assert counting_chat["n"] == 1                       # second call is a hit — no new call
    assert a == b == "reply-1"
    assert list(tmp_path.glob("*.json"))                 # entry persisted to disk


def test_changed_content_busts_the_cache(tmp_path, counting_chat):
    llm.cached_chat(MSG, cache_dir=str(tmp_path))
    llm.cached_chat([{"role": "user", "content": "different"}], cache_dir=str(tmp_path))
    assert counting_chat["n"] == 2                       # distinct content -> distinct key -> recompute


def test_changed_params_busts_the_cache(tmp_path, counting_chat):
    llm.cached_chat(MSG, cache_dir=str(tmp_path), max_tokens=100)
    llm.cached_chat(MSG, cache_dir=str(tmp_path), max_tokens=200)
    assert counting_chat["n"] == 2                       # max_tokens is part of the key


def test_failures_are_not_cached(tmp_path, monkeypatch):
    state = {"first": True}

    def flaky_chat(messages, temperature=0.0, max_tokens=512, timeout=60.0):
        if state["first"]:
            state["first"] = False
            raise RuntimeError("transient")
        return "ok"

    monkeypatch.setattr(llm, "chat", flaky_chat)
    with pytest.raises(RuntimeError):
        llm.cached_chat(MSG, cache_dir=str(tmp_path))
    assert not list(tmp_path.glob("*.json"))             # nothing cached on failure
    assert llm.cached_chat(MSG, cache_dir=str(tmp_path)) == "ok"  # retry succeeds + caches
    assert list(tmp_path.glob("*.json"))
