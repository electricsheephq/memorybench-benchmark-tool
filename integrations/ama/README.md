# Hermes LCM AMA-Bench overlay

This directory is a local, dependency-free AMA-Bench method overlay. It does
not clone AMA-Bench, download its dataset, or change the hermes-lcm bridge.
The method imports the AMB integration's `_BridgeHandle` so the JSONL
subprocess protocol, timeout handling, and crash-loud behavior stay shared.

## Register in a local AMA clone

Keep the AMA checkout separate and register this structural `BaseMethod`
before its normal entry point is loaded:

## Making the overlay importable from the AMA checkout

This repository ships no Python packaging metadata, so the AMA process must be
given the repo root on `PYTHONPATH` explicitly (this also provides the
`integrations.amb` dependency the method reuses):

```bash
export PYTHONPATH="/path/to/memorybench-benchmark-tool:$PYTHONPATH"
```

Run AMA from its own checkout with that environment set; the registration
snippet below then imports cleanly. (Alternative: symlink `integrations/` into
the AMA checkout root — PYTHONPATH is preferred because it needs no files
inside their tree.)

```python
from integrations.ama.hermes_lcm_method import HermesLcmMethod
from src.method.base_method import BaseMethod
from src.method_register import register_method

# The current AMA registry checks nominal inheritance. Keep this overlay
# structural and add the clone-local nominal shim only at registration.
class RegisteredHermesLcmMethod(HermesLcmMethod, BaseMethod):
    pass

register_method("hermes_lcm", RegisteredHermesLcmMethod)
```

The bridge path defaults to this checkout's
`src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py`. When the overlay is
copied elsewhere, set `HERMES_LCM_BRIDGE_PATH` and `HERMES_LCM_REPO`. Set
`HERMES_MB_WORKDIR` to a persistent per-run directory when desired; otherwise
the adapter creates owned scratch under
`/Volumes/LEXAR/Codex/session-notes/2026-07-30/hermes-ama-adapter/artifacts/`.
`HERMES_MB_PROVIDER` and `HERMES_MB_MODEL` are passed through to the bridge.

## Trajectory grammar and evidence boundary

AMA-Bench currently flattens each episode as repeated blocks of exactly:

```text
Step N:
Action: <action>
Observation: <observation>

```

The parser accepts numeric non-negative step labels, permits ordinary newlines
inside action/observation text, and requires the double-newline separator
between blocks. It accepts either the benchmark's current terminal newline or
the documented terminal double newline. Any gap, missing label, missing field,
or other drift raises a `ValueError` with its character offset. Empty text is
the only empty episode.
Actions become `assistant` messages and observations become `user` messages;
the task is stored in session metadata. Retrieval exposes only content and
optional `Date:` lines; bridge provenance, scores, and internal metadata never
reach the answer prompt.

## Run-time mitigations (apply on the AMA clone)

These are run-owner mitigations, intentionally documented rather than
implemented here:

1. Run `--subset openend` only. The public dataset currently does not provide
   the documented MCQ file.
2. Pin the AMA-Bench repo at `ddfd319e` and the Hugging Face dataset at
   `a5777378` (the full dataset commit is
   `a5777378066f53229a94557a7b192435cd027909`).
3. Patch the judge so an unparseable yes/no response fails instead of silently
   falling back to token-level F1, and patch the model client so a
   context-overflow truncate-and-retry fails closed. Record both patches in
   the run packet.

## Local proof

From this checkout, the bounded tests use only the existing stub bridge and no
network or dataset:

```bash
uv run --with pytest python3 -m pytest integrations/ama/tests -q -p no:xdist
python3 -m compileall -q integrations/ama
```

These checks prove adapter behavior and lifecycle only; they are not
AMA-Bench scores, release evidence, or runtime proof against a real bridge.
