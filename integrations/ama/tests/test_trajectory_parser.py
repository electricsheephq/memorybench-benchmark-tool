from __future__ import annotations

import pytest

from integrations.ama.trajectory_parser import TrajectoryStep, parse_trajectory


def test_parses_well_formed_multi_step_trajectory() -> None:
    text = (
        "Step 0:\n"
        "Action: inspect the file\n"
        "Observation: it contains a config\n\n"
        "Step 1:\n"
        "Action: update the config\n"
        "Observation: the change was accepted\n\n"
    )

    assert parse_trajectory(text) == [
        TrajectoryStep(0, "inspect the file", "it contains a config"),
        TrajectoryStep(1, "update the config", "the change was accepted"),
    ]


def test_malformed_segment_fails_with_offset() -> None:
    valid = (
        "Step 0:\n"
        "Action: inspect\n"
        "Observation: found it\n\n"
    )
    with pytest.raises(ValueError, match=rf"offset {len(valid)}"):
        parse_trajectory(valid + "Broken line\n")


def test_empty_trajectory_has_no_steps() -> None:
    assert parse_trajectory("") == []


def test_accepts_ama_terminal_single_newline() -> None:
    assert parse_trajectory(
        "Step 0:\nAction: inspect\nObservation: found it\n"
    ) == [TrajectoryStep(0, "inspect", "found it")]


def test_non_sequential_step_numbers_fail_loud():
    import pytest as _pytest

    from integrations.ama.trajectory_parser import parse_trajectory

    # An observation embedding step-shaped text would mis-split; the sequence
    # guard turns that silent corruption into a loud failure.
    text = (
        "Step 1:\nAction: browse\nObservation: quoting a log:\n\n"
        "Step 7:\nAction: fake\nObservation: quoted content\n"
    )
    with _pytest.raises(ValueError, match="non-sequential step number 7"):
        parse_trajectory(text)


def test_sequential_steps_parse_regardless_of_base():
    from integrations.ama.trajectory_parser import parse_trajectory

    text = (
        "Step 4:\nAction: a\nObservation: o\n\n"
        "Step 5:\nAction: b\nObservation: p\n"
    )
    assert [s.turn_idx for s in parse_trajectory(text)] == [4, 5]
