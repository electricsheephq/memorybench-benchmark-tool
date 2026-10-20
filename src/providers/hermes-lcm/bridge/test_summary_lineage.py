from __future__ import annotations

import json
import os
import sqlite3

import pytest

from test_embeddings_switch import make_bridge  # noqa: F401


pytestmark = pytest.mark.skipif(
    not os.environ.get("HERMES_LCM_REPO"),
    reason="requires sandboxed product checkout (v0.28.1 for B3 acceptance)",
)


def session(session_id, contents):
    return {
        "sessionId": session_id,
        "messages": [
            {"role": "user" if i % 2 == 0 else "assistant", "content": content}
            for i, content in enumerate(contents)
        ],
    }


@pytest.mark.parametrize("mode", ["on", "off"])
def test_summary_lineage_is_exact_ingest_order(make_bridge, mode):
    instance = make_bridge(mode)
    instance.initialize({})
    sessions = [
        session("preceding", ["Earlier note", "Earlier reply"]),
        session("target", ["First detail", "Second detail", "Third detail"]),
        session("empty", []),
    ]
    for ordinal, item in enumerate(sessions, 1):
        reply = instance.ingest({"containerTag": "lineage", "session": item})
        with sqlite3.connect(instance._db_path("lineage")) as connection:
            node = connection.execute(
                "SELECT summary, token_count, source_token_count, source_type, "
                "created_at, source_ids FROM summary_nodes WHERE session_id = ?",
                (item["sessionId"],),
            ).fetchone()
            rows = connection.execute(
                "SELECT store_id, content FROM messages WHERE session_id = ? ORDER BY store_id",
                (item["sessionId"],),
            ).fetchall()
        expected_ids = [int(value) for value in reply["documentIds"]] if rows else []
        assert [row[0] for row in rows] == expected_ids
        assert [row[1] for row in rows] == [m["content"] for m in item["messages"]]
        summary = instance._deterministic_session_summary(item["messages"])
        assert node[:5] == (
            summary, len(summary.split()),
            sum(len(m["content"].split()) for m in item["messages"]),
            "messages", float(ordinal),
        )
        assert json.loads(node[5]) == expected_ids


def test_summary_only_match_delivers_strict_citable_sources(make_bridge):
    instance = make_bridge("on")
    instance.initialize({})
    item = session("summary-only", ["Quarterly budget note", "Reconciliation complete"])
    reply = instance.ingest({"containerTag": "semantic", "session": item})
    # Role labels occur in the real deterministic summary, not source content.
    query = "assistant"
    assert query in instance._deterministic_session_summary(item["messages"])
    assert all(query not in m["content"] for m in item["messages"])
    with sqlite3.connect(instance._db_path("semantic")) as connection:
        # Fixture isolation: chunk KNN must not mask a missing summary lineage.
        connection.execute("DELETE FROM lcm_chunk_vectors")
        assert connection.execute("SELECT COUNT(*) FROM lcm_chunk_vectors").fetchone()[0] == 0
    result = instance.search({"containerTag": "semantic", "query": query, "limit": 5})
    # FTS "ok" means the arm ran successfully, even when it found no rows.
    assert result["provenance"]["coverage"]["fts"] == "ok"
    assert result["provenance"]["coverage"]["summary"] == "full"
    assert result["results"], "summary matches must survive reference-strict delivery"
    source_rows = dict(zip(map(int, reply["documentIds"]), item["messages"]))
    for hit in result["results"]:
        metadata = hit["metadata"]
        assert metadata["kind"] == "message_excerpt"
        assert metadata["arms"] == ["summary"]
        assert metadata["session_id"] == item["sessionId"]
        assert metadata["content_offset"] == 0
        assert hit["content"] == source_rows[metadata["store_id"]]["content"]
        assert metadata["expand_hint"]


def test_embeddings_off_delivered_output_unchanged_by_lineage(make_bridge):
    instance = make_bridge("off")
    instance.initialize({})
    item = session("lexical", ["Quarterly budget note", "Reconciliation complete"])
    instance.ingest({"containerTag": "lexical", "session": item})
    request = {"containerTag": "lexical", "query": "budget", "limit": 5}
    with_lineage = instance.search(request)
    with sqlite3.connect(instance._db_path("lexical")) as connection:
        connection.execute("UPDATE summary_nodes SET source_ids = '[]'")
    without_lineage = instance.search(request)
    assert with_lineage["results"]
    assert with_lineage["results"] == without_lineage["results"]
    assert with_lineage["degraded"] == without_lineage["degraded"]
    assert with_lineage["provenance"]["coverage"] == without_lineage["provenance"]["coverage"]
