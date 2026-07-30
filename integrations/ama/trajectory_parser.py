"""Strict parser for AMA-Bench's flattened trajectory representation.

AMA-Bench currently emits one block per turn in this exact shape::

    Step N:\nAction: <action>\nObservation: <observation>\n\n

The parser deliberately consumes only that format.  It does not search for
the next apparently valid block, because silently skipping text would turn an
upstream format change into incorrect memory.  A malformed block therefore
raises :class:`ValueError` with the byte/character offset at which parsing
stopped.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class TrajectoryStep:
    """One action/observation pair from a flattened AMA episode."""

    turn_idx: int
    action: str
    observation: str

    @property
    def index(self) -> int:
        """Compatibility spelling for callers that call the step number index."""

        return self.turn_idx


# The double newline separates blocks in MemoryQAInterface._trajectory_to_text.
# Its current join-based implementation emits one terminal newline for the
# final block, while the documented representation may end in a double
# newline.  Accept either terminal spelling, but never skip a gap.  ``.*?``
# permits ordinary newlines inside an action or observation while still
# stopping at the block delimiter.
_STEP_RE = re.compile(
    r"Step (?P<turn_idx>[0-9]+):\n"
    r"Action: (?P<action>.*?)\n"
    r"Observation: (?P<observation>.*?)(?:\n\n|\n\Z)",
    re.DOTALL,
)


def parse_trajectory(traj_text: str) -> list[TrajectoryStep]:
    """Parse a complete AMA-Bench flattened trajectory strictly.

    ``traj_text == ""`` is the one valid empty input and returns no steps.
    Every non-empty character must belong to a complete deterministic block;
    a gap or malformed block raises ``ValueError`` and names its offset.
    """

    if not isinstance(traj_text, str):
        raise TypeError(f"trajectory must be str, got {type(traj_text).__name__}")
    if traj_text == "":
        return []

    steps: list[TrajectoryStep] = []
    offset = 0
    while offset < len(traj_text):
        match = _STEP_RE.match(traj_text, offset)
        if match is None:
            raise ValueError(f"malformed trajectory segment at offset {offset}")
        steps.append(
            TrajectoryStep(
                turn_idx=int(match.group("turn_idx")),
                action=match.group("action"),
                observation=match.group("observation"),
            )
        )
        offset = match.end()
    return steps


# Explicit aliases keep the pure parser convenient for an overlay that wants
# to name the source representation rather than the benchmark method.
parse_flattened_trajectory = parse_trajectory
parse = parse_trajectory


__all__ = [
    "TrajectoryStep",
    "parse",
    "parse_flattened_trajectory",
    "parse_trajectory",
]
