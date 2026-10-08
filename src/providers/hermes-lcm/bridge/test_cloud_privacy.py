"""Cloud-provider privacy parity for the bridge (lcm-x #984).

Drives the real bridge against the product checkout with the product's HTTP
transport replaced by an in-process fake and every socket connect refused, so
no provider is ever called. All conversation text below is synthetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sqlite3
import sys
from pathlib import Path

import pytest

BRIDGE_PATH = Path(__file__).with_name("hermes_lcm_bridge.py")
SPEC = importlib.util.spec_from_file_location("cloud_privacy_bridge_under_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
_saved_stdout = sys.stdout
SPEC.loader.exec_module(bridge)
sys.stdout = _saved_stdout

pytestmark = pytest.mark.skipif(
    not os.environ.get("HERMES_LCM_REPO"), reason="requires sandboxed product checkout"
)

SYNTHETIC_SECRET = "EXAMPLE-NOT-A-KEY"
PLACEHOLDER = "[LCM embedding privacy:"
# Each message clears the product's 40-token conversational chunk floor, so
# both chunk and summary units are embedded.
_FILLER = (" The beds sit along the south fence, the hose runs from the shed, and the "
           "drip lines were replaced last spring after the old ones cracked in the frost.")
SESSION = {
    "sessionId": "synthetic-session-0",
    "metadata": {"date": "2023-05-01T10:00:00.000Z"},
    "messages": [
        {"role": "user", "content": "I am setting up the garden irrigation timer for the "
                                    "tomato beds and want it to water at six every other day." + _FILLER},
        {"role": "assistant", "content": "A six o'clock schedule every other day suits tomato "
                                         "beds; check soil moisture after the first week." + _FILLER},
        {"role": "user", "content": f"Store this controller line too: api_key={SYNTHETIC_SECRET} "
                                    "so I can find it after the irrigation controller resets." + _FILLER},
    ],
}


def _refuse(*_args, **_kwargs):
    raise OSError("test: network disabled")


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(b / 255.0) - 0.5 for b in digest[:16]]


@pytest.fixture
def bridge_env(monkeypatch, tmp_path):
    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path / "stores"))
    monkeypatch.setenv("HERMES_MB_EMBEDDINGS", "on")
    monkeypatch.delenv("HERMES_MB_FUSION", raising=False)
    monkeypatch.delenv("LCM_LONGMEMEVAL_EMBED_CACHE", raising=False)
    return monkeypatch


@pytest.fixture
def cloud(bridge_env):
    """A Voyage bridge whose product transport is faked; returns (bridge, calls)."""
    bridge_env.setenv("HERMES_MB_PROVIDER", "voyage")
    bridge_env.setenv("HERMES_MB_MODEL", "voyage-context-3")
    bridge_env.setenv("VOYAGE_API_KEY", "test-dummy-not-a-key")
    instance = bridge.Bridge()  # makes the product importable
    import hermes_lcm.embedding_provider as ep

    calls: list[dict] = []
    phase = {"name": "initialize"}

    def fake_transport(*, url, payload, headers, timeout):
        groups = payload.get("inputs") or [[text] for text in payload.get("input", [])]
        texts = [text for group in groups for text in group]
        calls.append({"phase": phase["name"], "input_type": payload.get("input_type"), "texts": texts})
        if "inputs" in payload:
            data = [{"index": i, "data": [{"index": j, "embedding": _vector(t)} for j, t in enumerate(g)]}
                    for i, g in enumerate(groups)]
        else:
            data = [{"index": i, "embedding": _vector(t)} for i, t in enumerate(texts)]
        body = {"data": data, "usage": {"total_tokens": sum(len(t) // 4 + 1 for t in texts)}}
        return ep.HttpResponse(status=200, headers={}, body=json.dumps(body).encode())

    bridge_env.setattr(ep, "_default_http_transport", fake_transport)
    instance.initialize({})
    instance.test_phase = phase
    return instance, calls


def _profiles(instance, tag):
    connection = sqlite3.connect(f"file:{instance._db_path(tag)}?mode=ro", uri=True)
    try:
        return dict(connection.execute(
            "SELECT task, revision FROM lcm_embedding_profile WHERE active=1"
        ).fetchall())
    finally:
        connection.close()


def test_cloud_ingest_registers_privacy_revision_and_recall_uses_vectors(cloud):
    instance, calls = cloud
    from hermes_lcm.ingest_protection import embedding_privacy_revision

    instance.test_phase["name"] = "ingest"
    assert instance.ingest({"containerTag": "cloud", "session": SESSION})["ok"] is True
    revision = embedding_privacy_revision(instance._config(instance._db_path("cloud")))
    assert revision.startswith("privacy:")
    assert _profiles(instance, "cloud") == {"summary": revision, "chunk": revision}

    instance.test_phase["name"] = "search"
    response = instance.search(
        {"containerTag": "cloud", "query": "When does the irrigation timer water?", "limit": 5}
    )
    query_calls = [c for c in calls if c["phase"] == "search" and c["input_type"] == "query"]
    assert query_calls, "a recall query must dispatch a query embedding"
    coverage = response["provenance"].get("coverage") or {}
    assert coverage.get("summary") not in (None, "none")
    assert coverage.get("chunk") not in (None, "none")
    assert response["degraded"] is False
    assert "embedding_identity_stale" not in str(response["degraded_reason"] or "")


def test_cloud_documents_reach_transport_redacted(cloud):
    instance, calls = cloud
    instance.test_phase["name"] = "ingest"
    instance.ingest({"containerTag": "redact", "session": SESSION})
    documents = [t for c in calls if c["input_type"] == "document" for t in c["texts"]]
    assert documents, "ingest must dispatch document embeddings"
    assert not any(SYNTHETIC_SECRET in text for text in documents)
    assert any(PLACEHOLDER in text for text in documents)


def test_cloud_validator_block_refuses_ingest(cloud, bridge_env):
    instance, calls = cloud
    import hermes_lcm.ingest_protection as protection

    def block(texts, config, *, expected_revision):
        raise protection.EmbeddingPrivacyPolicyError(
            "cloud embedding privacy residual detector blocked pattern names: api_key"
        )

    original = protection.validate_embedding_privacy_dispatch
    bridge_env.setattr(protection, "validate_embedding_privacy_dispatch", block)
    instance.test_phase["name"] = "ingest"
    with pytest.raises(RuntimeError) as refused:
        instance.ingest({"containerTag": "blocked", "session": SESSION})
    message = str(refused.value)
    assert message.startswith("privacy_refused:")
    assert "api_key" in message
    assert SYNTHETIC_SECRET not in message and "irrigation" not in message
    assert not [c for c in calls if c["phase"] == "ingest" and c["input_type"] == "document"]
    # Fail closed: the session is never recorded as ingested.
    assert not instance._ingested_path("blocked").exists()
    # Nothing is persisted before the refusal, so a corrected policy can resume the same container.
    assert not instance._session_has_rows(instance._db_path("blocked"), SESSION["sessionId"])
    bridge_env.setattr(protection, "validate_embedding_privacy_dispatch", original)
    instance.ingest({"containerTag": "blocked", "session": SESSION})
    assert instance._ingested_path("blocked").exists()


def test_local_provider_keeps_empty_revision_and_raw_documents(bridge_env):
    bridge_env.setenv("HERMES_MB_PROVIDER", "stub")
    bridge_env.setenv("HERMES_MB_MODEL", "stub-hash-64")
    instance = bridge.Bridge()
    instance.initialize({})
    sent: list[str] = []
    embed_documents = instance.embedder.embed_documents

    def recording(texts):
        sent.extend(texts)
        return embed_documents(texts)

    instance.embedder.embed_documents = recording
    instance.ingest({"containerTag": "local", "session": SESSION})
    assert _profiles(instance, "local") == {"summary": "", "chunk": ""}
    assert any(f"api_key={SYNTHETIC_SECRET}" in text for text in sent)
    assert not any(PLACEHOLDER in text for text in sent)
    assert sent[-1] == instance._deterministic_session_summary(
        [{"role": m["role"], "content": m["content"]} for m in SESSION["messages"]]
    )
