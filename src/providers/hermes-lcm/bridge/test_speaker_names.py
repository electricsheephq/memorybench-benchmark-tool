from __future__ import annotations

import os
import sqlite3

import pytest

from test_embeddings_switch import make_bridge  # noqa: F401


pytestmark = pytest.mark.skipif(
    not os.environ.get("HERMES_LCM_REPO"),
    reason="requires sandboxed product checkout",
)


def two_speakers():
    return {
        "sessionId": "two-speakers",
        "messages": [
            {"role": "user", "speaker": "Alice", "content": "The telescope is blue."},
            {"role": "assistant", "speaker": "Bob", "content": "The telescope is green."},
        ],
    }


@pytest.mark.parametrize("mode", ["off", "on"])
def test_ingest_preserves_both_speaker_names_and_roles(make_bridge, mode):
    instance = make_bridge(mode)
    instance.initialize({})
    item = two_speakers()
    instance.ingest({"containerTag": "speakers", "session": item})
    with sqlite3.connect(instance._db_path("speakers")) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY store_id"
        ).fetchall()
    assert rows == [
        (message["role"], f"{message['speaker']}: {message['content']}")
        for message in item["messages"]
    ]


@pytest.mark.parametrize("mode", ["off", "on"])
def test_lme_content_bytes_and_summary_are_unchanged(make_bridge, mode):
    instance = make_bridge(mode)
    instance.initialize({})
    messages = [
        {"role": "user", "content": "  Résumé\n東京\t"},
        {"role": "assistant", "content": "Reply\r\nwith whitespace  ", "speaker": ""},
        {"role": "user", "content": "No attribution", "speaker": None},
    ]
    instance.ingest({
        "containerTag": "lme", "session": {"sessionId": "lme", "messages": messages},
    })
    with sqlite3.connect(instance._db_path("lme")) as connection:
        rows = connection.execute(
            "SELECT role, content FROM messages ORDER BY store_id"
        ).fetchall()
        summary = connection.execute("SELECT summary FROM summary_nodes").fetchone()[0]
    assert [(role, content.encode("utf-8")) for role, content in rows] == [
        (message["role"], message["content"].encode("utf-8")) for message in messages
    ]
    assert summary.encode("utf-8") == instance._deterministic_session_summary(messages).encode("utf-8")


@pytest.mark.parametrize("mode", ["off", "on"])
def test_search_delivers_both_speaker_names(make_bridge, mode):
    instance = make_bridge(mode)
    instance.initialize({})
    instance.ingest({"containerTag": "delivery", "session": two_speakers()})
    result = instance.search({"containerTag": "delivery", "query": "telescope", "limit": 5})
    delivered = {hit["content"] for hit in result["results"]}
    assert "Alice: The telescope is blue." in delivered
    assert "Bob: The telescope is green." in delivered
