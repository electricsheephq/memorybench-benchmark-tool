from __future__ import annotations

import json
import sys
import types
from datetime import datetime, timezone

import pytest

from test_embeddings_switch import bridge, make_bridge  # noqa: F401


SESSION = {
    "sessionId": "fixture-session",
    "metadata": {"date": "2023-05-30T18:09:00.000Z"},
    "messages": [
        {"role": "user", "content": "First turn", "speaker": "Alpha\n\x00 \t Team"},
        {"role": "assistant", "content": "Second turn", "speaker": "Beta"},
        {"role": "user", "content": "No speaker"},
    ],
}
EXPECTED_EPOCH = datetime(2023, 5, 30, 18, 9, tzinfo=timezone.utc).timestamp()


@pytest.fixture
def ingest_capture(make_bridge, monkeypatch):
    """Exercise real bridge ingest, capturing its product append_batch call."""
    captured = []

    class Store:
        def __init__(self, *_args, **_kwargs):
            pass

        def append_batch(self, session_id, messages, **kwargs):
            captured.append((session_id, json.dumps(messages).encode(), kwargs))
            return list(range(1, len(messages) + 1))

        def get_time_bounds(self, _ids):
            return (1.0, 2.0)

        def close(self):
            pass

    class Dag(Store):
        def add_node(self, _node):
            return 1

    for name, attributes in {
        "store": {"MessageStore": Store},
        "dag": {"SummaryDAG": Dag, "SummaryNode": types.SimpleNamespace},
        "vector_store": {"VectorStore": Store, "EmbeddingIdentity": object},
        "chunking": {"iter_message_chunks": lambda *_args, **_kwargs: []},
        "ingest_protection": {
            "embedding_privacy_revision": lambda _config: None,
            "embedding_provider_requires_privacy": lambda _name: False,
        },
    }.items():
        module = types.ModuleType(f"hermes_lcm.{name}")
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    def run(event_time=None, sender_render=None, metadata=None):
        for name, value in (
            ("HERMES_MB_EVENT_TIME", event_time),
            ("HERMES_MB_SENDER_RENDER", sender_render),
        ):
            if value is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, value)
        instance = make_bridge("off")
        instance.initialize({})
        monkeypatch.setattr(instance, "_session_has_rows", lambda *_args: False)
        tag = f"capture-{len(captured)}"
        session = {**SESSION, "metadata": metadata if metadata is not None else SESSION["metadata"]}
        response = instance.ingest({"containerTag": tag, "session": session})
        return instance, response, captured[-1], session, tag

    return run


@pytest.mark.parametrize("mode", [None, "off"])
def test_off_product_payload_is_byte_identical_to_base(ingest_capture, mode):
    instance, response, (session_id, payload, kwargs), _session, _tag = ingest_capture(mode, mode)
    base = [{"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in SESSION["messages"]]
    assert payload == json.dumps(base).encode()
    assert session_id == SESSION["sessionId"]
    assert kwargs == {"source": "benchmark", "conversation_id": session_id}
    assert response == {"ok": True, "documentIds": ["1", "2", "3"]}
    assert instance._harness_settings() == {
        "HERMES_MB_EVENT_TIME": "off", "HERMES_MB_SENDER_RENDER": "off",
    }


@pytest.mark.parametrize("value", [
    "2023-05-30T18:09:00.000Z",
    "2023-05-30T18:09:00",
    "2023-05-30 18:09:00",
    "2023-05-30T20:09:00+02:00",
    "6:09 pm on 30 May, 2023",
    "6:09 PM on 30 May 2023",
])
def test_session_timestamp_is_numeric_and_increases_in_utc(ingest_capture, value):
    _instance, response, (_sid, payload, _kwargs), _session, _tag = ingest_capture(
        "session", metadata={"date": value}
    )
    messages = json.loads(payload)
    assert all(isinstance(m["timestamp"], (int, float)) for m in messages)
    assert [m["timestamp"] for m in messages] == [EXPECTED_EPOCH + i for i in range(3)]
    assert response["unparsed_session_dates"] == 0


def test_formatted_date_fallback_and_both_modes(ingest_capture):
    _instance, response, (_sid, payload, _kwargs), _session, _tag = ingest_capture(
        "session", "gateway", {"date": "invalid", "formattedDate": "6:09 pm on 30 May, 2023"}
    )
    messages = json.loads(payload)
    assert messages[0] == {"role": "user", "content": "[Alpha Team] First turn", "timestamp": EXPECTED_EPOCH}
    assert messages[1] == {"role": "assistant", "content": "[Beta] Second turn", "timestamp": EXPECTED_EPOCH + 1}
    assert messages[2]["content"] == "No speaker"
    assert response["unparsed_session_dates"] == 0


@pytest.mark.parametrize("metadata", [
    {"date": "not a date"}, {}, {"formattedDate": "1:00 pm on 31 February, 2023"},
    {"date": "2023-13-01T00:00:00"},
])
def test_unparseable_session_has_no_timestamp_and_is_counted(ingest_capture, metadata):
    instance, response, (_sid, payload, _kwargs), session, tag = ingest_capture(
        "session", metadata=metadata
    )
    assert all("timestamp" not in m for m in json.loads(payload))
    assert response["unparsed_session_dates"] == 1
    resumed = instance.ingest({"containerTag": tag, "session": session})
    assert resumed["resumed"] is True and resumed["unparsed_session_dates"] == 1


def test_gateway_only_leaves_time_and_speakerless_turns_unchanged(ingest_capture):
    _instance, _response, (_sid, payload, _kwargs), _session, _tag = ingest_capture("off", "gateway")
    messages = json.loads(payload)
    assert [m["content"] for m in messages] == ["[Alpha Team] First turn", "[Beta] Second turn", "No speaker"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert all("timestamp" not in m for m in messages)


@pytest.mark.parametrize("name, enabled", [
    ("HERMES_MB_EVENT_TIME", "session"), ("HERMES_MB_SENDER_RENDER", "gateway"),
])
def test_modes_validate_at_bridge_construction(make_bridge, monkeypatch, name, enabled):
    monkeypatch.setenv(name, "undeclared")
    with pytest.raises(RuntimeError, match=f"{name} must be off or {enabled}"):
        make_bridge("off")


@pytest.mark.parametrize("hour, expected_hour", [("12:09 am", 0), ("12:09 pm", 12)])
def test_formatted_midnight_noon_and_abbreviated_month(hour, expected_hour):
    assert bridge._session_epoch(f"{hour} on 30 Sept, 2023") == datetime(
        2023, 9, 30, expected_hour, 9, tzinfo=timezone.utc
    ).timestamp()


@pytest.mark.parametrize("hydrated", [False, True])
def test_event_facets_survive_hydration_and_metadata(hydrated):
    hit = {
        "kind": "message_excerpt", "store_id": 1, "session_id": "s1",
        "event_time": "2023-05-30T18:09:00Z", "event_time_source": "host_timestamp",
    }
    if hydrated:
        hit.update(content="Fixture", content_chars=7, content_returned_chars=7, content_truncated=False)
    result = bridge._hydrate_answer_ready_hit(
        hit, store=types.SimpleNamespace(get=lambda _id: {"role": "user", "content": "Fixture"}),
        dag=None, query="Fixture", char_cap=100,
    )
    metadata = bridge._metadata_for_recall_hit(result, {})
    assert metadata["event_time"] == hit["event_time"]
    assert metadata["event_time_source"] == hit["event_time_source"]
    without = bridge._metadata_for_recall_hit({"store_id": 1}, {})
    assert "event_time" not in without and "event_time_source" not in without
