from __future__ import annotations

import json
from pathlib import Path

import pytest

from integrations.ama.hermes_lcm_method import HermesLcmMethod


STUB = Path(__file__).resolve().parents[2] / "amb" / "tests" / "stub_bridge.py"


@pytest.fixture
def method(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> HermesLcmMethod:
    monkeypatch.setenv("STUB_BRIDGE_MODE", "normal")
    instance = HermesLcmMethod(
        bridge_path=STUB,
        workdir=tmp_path / "bridge-workdir",
        request_timeout=1.0,
        initialize_timeout=1.0,
    )
    yield instance
    instance.cleanup()


def test_construction_maps_steps_to_bridge_messages(
    method: HermesLcmMethod,
) -> None:
    handle = method.memory_construction(
        (
            "Step 0:\n"
            "Action: inspect\n"
            "Observation: found config\n\n"
            "Step 1:\n"
            "Action: edit\n"
            "Observation: saved\n\n"
        ),
        "update the configuration",
    )

    assert handle.startswith("ama-")
    evidence = method.memory_retrieve(handle, "what changed?")
    session = json.loads(evidence)
    assert session["sessionId"] == handle
    assert session["metadata"] == {"task": "update the configuration"}
    assert session["messages"] == [
        {"role": "assistant", "content": "inspect"},
        {"role": "user", "content": "found config"},
        {"role": "assistant", "content": "edit"},
        {"role": "user", "content": "saved"},
    ]


def test_retrieve_exposes_content_and_date_but_no_bridge_internals(
    method: HermesLcmMethod,
) -> None:
    handle = method.memory_construction("", "empty task")
    evidence = method.memory_retrieve(handle, "query")

    assert "sessionId" in evidence
    assert "must-not-reach-raw-response" not in evidence
    assert "provenance" not in evidence
    assert "score" not in evidence

    assert HermesLcmMethod.render_evidence(
        [
            {
                "content": "remember this",
                "metadata": {
                    "date": "2026-07-30",
                    "score": 0.99,
                    "session_id": "private",
                },
            }
        ]
    ) == "Date: 2026-07-30\nremember this"


def test_context_manager_clears_and_closes_bridge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STUB_BRIDGE_MODE", "normal")
    with HermesLcmMethod(
        bridge_path=STUB,
        workdir=tmp_path / "bridge-workdir",
        request_timeout=1.0,
        initialize_timeout=1.0,
    ) as method:
        method.memory_construction("", "lifecycle")
        assert method._bridge is not None
        process = method._bridge.process

    process.wait(timeout=2)
    assert process.poll() is not None
    method.cleanup()
