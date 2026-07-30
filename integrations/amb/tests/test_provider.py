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
