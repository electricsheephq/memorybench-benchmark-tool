from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sqlite3
import sys
import types
from pathlib import Path

import pytest

BRIDGE_PATH = Path(__file__).with_name("hermes_lcm_bridge.py")
SPEC = importlib.util.spec_from_file_location("embeddings_bridge_under_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
_saved_stdout = sys.stdout
SPEC.loader.exec_module(bridge)
sys.stdout = _saved_stdout


@pytest.fixture
def make_bridge(monkeypatch, tmp_path):
    repo = os.environ.get("HERMES_LCM_REPO")
    if not repo:
        # Isolated switch tests need no product imports or model artifacts.
        repo = str(tmp_path)
        monkeypatch.setenv("HERMES_LCM_REPO", repo)
        harness = types.ModuleType("benchmarking.longmemeval")
        harness._ensure_hermes_lcm_package = lambda: None
        harness.deterministic_session_summary = lambda messages: "fixture summary"
        harness.resolve_harness_provider = lambda name, model: types.SimpleNamespace(
            dim=64, model_id="stub-hash-64"
        )
        package = types.ModuleType("benchmarking")
        package.__path__ = []
        monkeypatch.setitem(sys.modules, "benchmarking", package)
        monkeypatch.setitem(sys.modules, "benchmarking.longmemeval", harness)
        config = types.ModuleType("hermes_lcm.config")
        config.LCMConfig = lambda **kwargs: types.SimpleNamespace(**kwargs)
        hermes = types.ModuleType("hermes_lcm")
        hermes.__path__ = []
        monkeypatch.setitem(sys.modules, "hermes_lcm", hermes)
        monkeypatch.setitem(sys.modules, "hermes_lcm.config", config)
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MB_PROVIDER", "stub")
    monkeypatch.setenv("HERMES_MB_MODEL", "stub-hash-64")
    monkeypatch.delenv("HERMES_MB_FUSION", raising=False)
    monkeypatch.delenv("LCM_LONGMEMEVAL_EMBED_CACHE", raising=False)

    def make(mode):
        if mode is None:
            monkeypatch.delenv("HERMES_MB_EMBEDDINGS", raising=False)
        else:
            monkeypatch.setenv("HERMES_MB_EMBEDDINGS", mode)
        return bridge.Bridge()

    return make


def forbidden(*_args, **_kwargs):
    raise AssertionError("off mode must never resolve an embedder")


@pytest.mark.parametrize("provider_name", ["stub", "voyage"])
def test_off_initialize_never_resolves_provider(make_bridge, monkeypatch, provider_name):
    instance = make_bridge("off")
    monkeypatch.setattr(instance, "_resolve_harness_provider", forbidden)
    # Disabling embeddings also avoids the metered-provider key preflight.
    instance.provider_name = provider_name
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    reply = instance.initialize({})
    assert reply["embeddings_enabled"] is False
    assert instance.embedder is None and reply["dim"] == 0
    assert instance._config(instance._db_path("off")).embeddings_enabled is False


@pytest.mark.parametrize("fusion", ["quota:fts=1,chunk=2", "invalid", " "])
def test_off_fusion_refused_before_provider(make_bridge, monkeypatch, fusion):
    instance = make_bridge("off")
    monkeypatch.setenv("HERMES_MB_FUSION", fusion)
    monkeypatch.setattr(instance, "_resolve_harness_provider", forbidden)
    with pytest.raises(RuntimeError, match="quota fusion needs chunk vectors"):
        instance.initialize({})


@pytest.mark.parametrize("mode", [None, "on"])
def test_on_and_default_resolve_once(make_bridge, monkeypatch, mode):
    instance = make_bridge(mode)
    resolver = instance._resolve_harness_provider
    calls = []

    def counted(*args):
        calls.append(args)
        return resolver(*args)

    monkeypatch.setattr(instance, "_resolve_harness_provider", counted)
    reply = instance.initialize({})
    instance._ensure_embedder()
    assert reply["embeddings_enabled"] is True
    assert len(calls) == 1 and reply["dim"] > 0
    assert instance._config(instance._db_path("on")).embeddings_enabled is True


@pytest.mark.parametrize("mode", ["", "false", "ON"])
def test_invalid_switch_is_refused(make_bridge, mode):
    with pytest.raises(RuntimeError, match="HERMES_MB_EMBEDDINGS must be on or off"):
        make_bridge(mode)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_requires_initialize(make_bridge, mode):
    instance = make_bridge(mode)
    with pytest.raises(RuntimeError, match="ingest before initialize"):
        instance.ingest({})
    with pytest.raises(RuntimeError, match="search before initialize"):
        instance.search({})


def _short_conversation():
    dataset = Path(__file__).resolve().parents[4] / "data" / "locomo10.json"
    items = json.loads(dataset.read_text())
    # Use one complete short session from the pinned LoCoMo corpus, not QA gold.
    sessions = [
        (item, key, turns)
        for item in items
        for key, turns in item["conversation"].items()
        if re.fullmatch(r"session_[0-9]+", key) and isinstance(turns, list)
        and len(turns) >= 3
        and all(re.search(r"[A-Za-z]{5,}", turn["text"]) for turn in turns[:3])
    ]
    item, key, turns = min(sessions, key=lambda row: sum(len(t["text"]) for t in row[2]))
    messages = []
    for turn in turns:
        content = turn["text"]
        caption = str(turn.get("blip_caption") or "").strip()
        if caption:
            content += f" [shared image: {caption}]"
        messages.append({
            "role": "user" if turn["speaker"] == item["conversation"]["speaker_a"] else "assistant",
            "content": content,
        })
    queries = []
    for message in messages[:3]:
        words = re.findall(r"[A-Za-z]{5,}", message["content"])
        assert bool(words), "selected session must supply three lexical queries"
        queries.append(max(words, key=len))
    return {"sessionId": f"control-{key}", "messages": messages}, queries


def _table_count(connection, table):
    if not connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone():
        return 0
    return connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


@pytest.mark.skipif(not os.environ.get("HERMES_LCM_REPO"), reason="requires sandboxed product checkout")
def test_product_positive_control(make_bridge, monkeypatch, tmp_path):
    session, queries = _short_conversation()
    receipts = []
    for mode in ("off", "on"):
        instance = make_bridge(mode)
        counts = {"documents": 0, "query": 0, "resolutions": 0}
        resolver = instance._resolve_harness_provider

        def counted_resolver(*args, counts=counts, resolver=resolver):
            counts["resolutions"] += 1
            provider = resolver(*args)
            documents, query = provider.embed_documents, provider.embed_query

            def counted_documents(texts, counts=counts):
                counts["documents"] += 1
                return documents(texts)

            def counted_query(text, counts=counts):
                counts["query"] += 1
                return query(text)

            provider.embed_documents = counted_documents
            provider.embed_query = counted_query
            return provider

        monkeypatch.setattr(instance, "_resolve_harness_provider", counted_resolver if mode == "on" else forbidden)
        reply = instance.initialize({})
        tag = f"control-{mode}"
        ingested = instance.ingest({"containerTag": tag, "session": session})
        searches = []
        for query in queries:
            response = instance.search({"containerTag": tag, "query": query, "limit": 5})
            # Do not emit corpus text even when a bar fails.
            assert bool(response["results"]), "production recall must return evidence"
            assert all(bool(r["content"]) for r in response["results"])
            searches.append({
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "hits": len(response["results"]),
                "content_sha256": [hashlib.sha256(r["content"].encode()).hexdigest() for r in response["results"]],
                "store_ids": [r["metadata"].get("store_id") for r in response["results"]],
                "coverage": response["provenance"].get("coverage"),
                "degraded": response["degraded"],
            })
        with sqlite3.connect(instance._db_path(tag)) as connection:
            storage = {table: _table_count(connection, table) for table in (
                "messages", "summary_nodes", "lcm_embedding_vectors", "lcm_chunk_vectors", "lcm_embedding_profile"
            )}
        assert storage["messages"] == len(session["messages"])
        assert storage["summary_nodes"] == 1
        if mode == "off":
            assert counts == {"documents": 0, "query": 0, "resolutions": 0}
            assert storage["lcm_embedding_vectors"] == storage["lcm_chunk_vectors"] == 0
            assert storage["lcm_embedding_profile"] == 0
            assert all(s["coverage"]["fts"] == "ok" for s in searches)  # product lcm_recall: "ok" | "none"
        else:
            assert counts["documents"] > 0 and counts["query"] > 0
            assert storage["lcm_embedding_vectors"] > 0 and storage["lcm_embedding_profile"] > 0
            # Also exercise the drifted private helpers on the declared quota path.
            monkeypatch.setenv("HERMES_MB_FUSION", "quota:fts=1,chunk=2")
            quota = instance.search({"containerTag": tag, "query": queries[0], "limit": 5})
            assert bool(quota["results"]) and quota["provenance"]["arms_run"] == ["fts", "chunk"]
            monkeypatch.delenv("HERMES_MB_FUSION")
        receipts.append({
            "mode": mode, "embeddings_enabled": reply["embeddings_enabled"],
            "conversation_messages": len(session["messages"]),
            "ordinary_searches": 3, "additional_quota_probe": mode == "on",
            "provider": reply["provider"], "model": reply["model"],
            "embed_calls": counts, "storage": storage, "searches": searches,
            "document_ids": ingested["documentIds"],
            "session_sha256": hashlib.sha256(json.dumps(session, sort_keys=True).encode()).hexdigest(),
        })
    evidence = os.environ.get("HERMES_MB_CONTROL_RECEIPT")
    if evidence:
        Path(evidence).write_text(json.dumps(receipts, indent=2) + "\n")


def test_resume_skips_a_session_already_ingested(make_bridge, tmp_path):
    instance = make_bridge("off")
    instance.initialize({})
    instance._ingested_path("resume").write_text(json.dumps({"s1": [4, 5, 6]}), encoding="utf-8")
    reply = instance.ingest({"containerTag": "resume", "session": {"sessionId": "s1", "messages": []}})
    assert reply == {"ok": True, "documentIds": ["4", "5", "6"], "resumed": True}
    assert not instance._db_path("resume").exists()


def test_unreadable_completion_record_stops_the_run(make_bridge):
    instance = make_bridge("off")
    instance.initialize({})
    instance._ingested_path("resume").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        instance.ingest({"containerTag": "resume", "session": {"sessionId": "s1", "messages": []}})


@pytest.mark.skipif(not os.environ.get("HERMES_LCM_REPO"), reason="requires sandboxed product checkout")
def test_resumed_ingest_never_stores_a_session_twice(make_bridge, monkeypatch):
    session, _queries = _short_conversation()

    def message_rows(db):
        connection = sqlite3.connect(db)
        try:
            return connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        finally:
            connection.close()

    first = make_bridge("off")
    monkeypatch.setattr(first, "_resolve_harness_provider", forbidden)
    first.initialize({})
    stored = first.ingest({"containerTag": "resume", "session": session})
    db = first._db_path("resume")
    rows = message_rows(db)
    assert rows == len(session["messages"])
    # A watchdog resume starts a new bridge process that re-sends the session.
    second = make_bridge("off")
    monkeypatch.setattr(second, "_resolve_harness_provider", forbidden)
    second.initialize({})
    again = second.ingest({"containerTag": "resume", "session": session})
    assert again["resumed"] is True and again["documentIds"] == stored["documentIds"]
    assert message_rows(db) == rows
    # Rows stored but no completion record: an ingest cut off mid-way stops the run.
    second._ingested_path("resume").unlink()
    with pytest.raises(RuntimeError, match="partly stored"):
        second.ingest({"containerTag": "resume", "session": session})
    assert message_rows(db) == rows


def test_clear_removes_the_ingest_record(make_bridge):
    instance = make_bridge("off")
    instance.initialize({})
    record = instance._ingested_path("resume")
    record.write_text(json.dumps({"s1": [1]}), encoding="utf-8")
    record.with_suffix(".tmp").write_text("{}", encoding="utf-8")
    assert instance.clear({"containerTag": "resume"}) == {"ok": True}
    assert not record.exists() and not record.with_suffix(".tmp").exists()
