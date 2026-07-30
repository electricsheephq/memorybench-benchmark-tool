"""Pure LoCoMo ``Document`` to hermes-lcm session normalization."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


def _field(document: Any, name: str, default: Any = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(name, default)
    return getattr(document, name, default)


def _turns(document: Any) -> tuple[list[Mapping[str, Any]], str | None, str | None]:
    raw = _field(document, "content", "")
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            messages = _field(document, "messages")
            if messages is None:
                raise ValueError("LoCoMo document content is not valid JSON") from exc
            payload = messages
    else:
        payload = raw

    speaker_a: str | None = None
    speaker_b: str | None = None
    if isinstance(payload, Mapping):
        speaker_a = str(payload["speaker_a"]) if payload.get("speaker_a") is not None else None
        speaker_b = str(payload["speaker_b"]) if payload.get("speaker_b") is not None else None
        candidate = payload.get("messages", payload.get("turns", payload))
        payload = candidate if isinstance(candidate, Sequence) and not isinstance(candidate, str) else [candidate]
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("LoCoMo document content must contain a turn list")
    turns = [turn for turn in payload if isinstance(turn, Mapping)]
    if len(turns) != len(payload):
        raise ValueError("LoCoMo turn list contains a non-object entry")
    return turns, speaker_a, speaker_b


def document_to_session(document: Any) -> dict[str, Any]:
    """Map one AMB LoCoMo document to the bridge's session payload.

    AMB stores raw ``speaker``/``text`` turn JSON in ``Document.content``.
    The first distinct speaker follows the TS adapter's ``speaker_a`` → user
    convention; later speakers become assistant messages.
    """

    document_id = _field(document, "id")
    if document_id is None:
        raise ValueError("LoCoMo document is missing id")
    turns, speaker_a, speaker_b = _turns(document)
    if speaker_a is None:
        for turn in turns:
            if turn.get("speaker") is not None:
                speaker_a = str(turn["speaker"])
                break

    messages: list[dict[str, str]] = []
    for turn in turns:
        explicit_role = turn.get("role")
        speaker = turn.get("speaker")
        if explicit_role is not None and str(explicit_role).lower() in {"user", "assistant"}:
            role = str(explicit_role).lower()
        else:
            role = "user" if speaker_a is None or str(speaker) == speaker_a else "assistant"
        text = turn.get("text", turn.get("content"))
        if text is None:
            raise ValueError("LoCoMo turn is missing text/content")
        message: dict[str, str] = {"role": role, "content": str(text)}
        # Payload parity with the TS lane: the bridge ignores speaker fields
        # today (the F46 §3 misattribution surface), but a speaker-aware bridge
        # must see identical payloads from both lanes.
        if speaker is not None:
            message["speaker"] = str(speaker)
        messages.append(message)

    metadata: dict[str, str] = {}
    timestamp = _field(document, "timestamp")
    if timestamp is not None:
        metadata["date"] = str(timestamp)
    if speaker_a is not None:
        metadata["speakerA"] = speaker_a
    if speaker_b is not None:
        metadata["speakerB"] = speaker_b
    return {"sessionId": str(document_id), "metadata": metadata, "messages": messages}


__all__ = ["document_to_session"]
