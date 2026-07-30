from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.amb.hermes_lcm_provider import Document, HermesLcmProvider


STUB = Path(__file__).with_name("stub_bridge.py")


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> HermesLcmProvider:
    monkeypatch.setenv("HERMES_LCM_BRIDGE_PATH", str(STUB))
    monkeypatch.setenv("STUB_BRIDGE_MODE", "normal")
    instance = HermesLcmProvider(request_timeout=1.0, initialize_timeout=1.0)
    yield instance
    instance.cleanup()


def test_initialize_performs_bridge_handshake(provider: HermesLcmProvider) -> None:
    provider.initialize()
    assert provider._bridge is not None
    assert provider._bridge.process.poll() is None


def test_ingest_maps_documents_and_retrieve_returns_documents(provider: HermesLcmProvider) -> None:
    provider.initialize()
    document = Document(
        id="sample-session_1",
        user_id="sample",
        timestamp="2024-01-02",
        content=json.dumps(
            [
                {"speaker": "Alice", "text": "hello"},
                {"speaker": "Bob", "text": "world"},
            ]
        ),
    )

    provider.ingest([document])
    documents, _raw = provider.retrieve("what happened?", k=3, user_id="sample")

    session = json.loads(documents[0].content)
    assert session["sessionId"] == "sample-session_1"
    assert session["metadata"] == {"date": "2024-01-02", "speakerA": "Alice"}
    assert session["messages"] == [
        {"role": "user", "content": "hello", "speaker": "Alice"},
        {"role": "assistant", "content": "world", "speaker": "Bob"},
    ]


def test_raw_response_is_results_only(provider: HermesLcmProvider) -> None:
    provider.initialize()
    _documents, raw = provider.retrieve("query", user_id="sample")

    assert set(raw) == {"results"}
    assert "provenance" not in raw
    assert "degraded" not in raw
    assert "degraded_reason" not in raw
    assert "internal" not in raw
    assert "must-not-reach-raw-response" not in json.dumps(raw)
    # Per-result internals must not reach the graded prompt either: metadata
    # is projected onto the reader-visible allowlist only.
    allowed = {"session_id", "date", "timestamp", "role", "kind"}
    for result in raw["results"]:
        assert set(result) == {"content", "metadata"}
        assert set(result["metadata"]) <= allowed


@pytest.mark.parametrize("mode", ["crash", "nonzero", "malformed"])
def test_bridge_failures_raise(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setenv("HERMES_LCM_BRIDGE_PATH", str(STUB))
    monkeypatch.setenv("STUB_BRIDGE_MODE", mode)
    provider = HermesLcmProvider(request_timeout=1.0, initialize_timeout=1.0)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            provider.initialize()
    finally:
        provider.cleanup()


def test_cleanup_terminates_bridge(provider: HermesLcmProvider) -> None:
    provider.initialize()
    assert provider._bridge is not None
    process = provider._bridge.process

    provider.cleanup()
    process.wait(timeout=2)
    assert process.poll() is not None
    provider.cleanup()


def test_prepare_without_reset_still_initializes(
    provider: HermesLcmProvider, tmp_path: Path
) -> None:
    provider.prepare(tmp_path / "stores", reset=False)
    # A reused store must still get a live bridge (was: "not initialized").
    _documents, raw = provider.retrieve("query", user_id="sample")
    assert set(raw) == {"results"}


def test_prepare_full_reset_wipes_persisted_store_files(
    provider: HermesLcmProvider, tmp_path: Path
) -> None:
    stores = tmp_path / "stores"
    stores.mkdir()
    stale = [
        stores / "old.db",
        stores / "old.db-wal",
        stores / "old.db-shm",
        stores / "old.dates.json",
    ]
    for path in stale:
        path.write_text("stale")
    keep = stores / "notes.txt"
    keep.write_text("not a store file")

    provider.prepare(stores, reset=True)

    assert all(not path.exists() for path in stale)
    assert keep.exists()


def test_retrieve_by_steps_fails_closed(provider: HermesLcmProvider) -> None:
    provider.initialize()
    with pytest.raises(NotImplementedError, match="step-scoped"):
        provider.retrieve_by_steps([1, 2], "query", user_id="sample")


def test_cleanup_removes_owned_temp_workdir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HERMES_LCM_BRIDGE_PATH", str(STUB))
    monkeypatch.setenv("STUB_BRIDGE_MODE", "normal")
    monkeypatch.delenv("HERMES_MB_WORKDIR", raising=False)
    instance = HermesLcmProvider(request_timeout=1.0, initialize_timeout=1.0)
    instance.initialize()
    workdir = instance._workdir
    assert workdir is not None and workdir.is_dir()

    instance.cleanup()
    assert not workdir.exists()


def test_speaker_identity_is_conversation_level(provider: HermesLcmProvider) -> None:
    provider.initialize()
    first = Document(
        id="conv-1-session_1",
        user_id="conv-1",
        content=json.dumps(
            {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "messages": [{"speaker": "Alice", "text": "hi"}],
            }
        ),
    )
    # Session 2 is a RAW turn list opening with speaker B: first-seen order
    # would invert every role without the conversation-level cache.
    second = Document(
        id="conv-1-session_2",
        user_id="conv-1",
        content=json.dumps(
            [
                {"speaker": "Bob", "text": "opening line"},
                {"speaker": "Alice", "text": "reply"},
            ]
        ),
    )

    provider.ingest([first, second])
    _documents, raw = provider.retrieve("opening", user_id="conv-1")
    session = json.loads(raw["results"][0]["content"])
    assert session["sessionId"] == "conv-1-session_2"
    assert session["messages"][0] == {
        "role": "assistant",
        "content": "opening line",
        "speaker": "Bob",
    }
    assert session["messages"][1] == {
        "role": "user",
        "content": "reply",
        "speaker": "Alice",
    }
