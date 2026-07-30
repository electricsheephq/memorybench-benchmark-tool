from __future__ import annotations

import json

import pytest

from integrations.amb.hermes_lcm_provider import Document
from integrations.amb.locomo_mapper import document_to_session


def test_maps_raw_locomo_turns_to_bridge_session() -> None:
    document = Document(
        id="sample-session_1",
        user_id="sample",
        timestamp="2024-01-02T03:04:05Z",
        content=json.dumps(
            [
                {"speaker": "Alice", "dia_id": "1", "text": "Hello"},
                {"speaker": "Bob", "dia_id": "2", "text": "Hi there"},
                {"speaker": "Alice", "dia_id": "3", "text": "Remember this"},
            ]
        ),
    )

    assert document_to_session(document) == {
        "sessionId": "sample-session_1",
        "metadata": {"date": "2024-01-02T03:04:05Z", "speakerA": "Alice"},
        "messages": [
            {"role": "user", "content": "Hello", "speaker": "Alice"},
            {"role": "assistant", "content": "Hi there", "speaker": "Bob"},
            {"role": "user", "content": "Remember this", "speaker": "Alice"},
        ],
    }


def test_honors_explicit_speaker_metadata_wrapper() -> None:
    document = {
        "id": "session-2",
        "timestamp": "2024-02-03",
        "content": json.dumps(
            {
                "speaker_a": "Person A",
                "speaker_b": "Person B",
                "messages": [
                    {"speaker": "Person B", "text": "answer"},
                    {"speaker": "Person A", "text": "question"},
                ],
            }
        ),
    }

    session = document_to_session(document)
    assert session["messages"] == [
        {"role": "assistant", "content": "answer", "speaker": "Person B"},
        {"role": "user", "content": "question", "speaker": "Person A"},
    ]
    assert session["metadata"] == {
        "date": "2024-02-03",
        "speakerA": "Person A",
        "speakerB": "Person B",
    }


@pytest.mark.parametrize(
    "document, message",
    [
        ({"id": "bad", "content": "not json"}, "valid JSON"),
        ({"content": "[]"}, "missing id"),
        ({"id": "bad", "content": "[{\"speaker\": \"A\"}]"}, "missing text"),
    ],
)
def test_rejects_invalid_documents(document: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        document_to_session(document)
