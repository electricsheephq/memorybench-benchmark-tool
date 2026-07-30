# Hermes LCM AMB overlay

This directory is a local integration overlay for the unmodified
`hermes_lcm_bridge.py`. It is not an AMB fork and does not download datasets or
run benchmarks. The unit tests use only the stub bridge in `tests/`.

## Install into a local AMB checkout

Keep AMB as a local checkout. Copy or symlink both Python modules into its
provider package (the names may be adjusted to match the checkout layout):

```bash
cp integrations/amb/hermes_lcm_provider.py /path/to/agent-memory-benchmark/src/memory_bench/memory/
cp integrations/amb/locomo_mapper.py /path/to/agent-memory-benchmark/src/memory_bench/memory/
```

Apply this two-line registry patch in the local clone:

```python
from .hermes_lcm_provider import HermesLcmProvider
REGISTRY["hermes-lcm"] = HermesLcmProvider
```

The provider defaults to the bridge at
`src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py` relative to this repo.
Set `HERMES_LCM_BRIDGE_PATH` when the overlay is copied elsewhere, and set
`HERMES_LCM_REPO` / `HERMES_MB_WORKDIR` for the real hermes-lcm checkout and
its persistent store. All `LCM_*` environment variables are inherited by the
bridge, as are the existing `HERMES_MB_PROVIDER` and `HERMES_MB_MODEL`
embedder selectors.

## Run discipline

For any real AMB run, pin and record the dataset SHA-256, set
`OMB_ANSWER_LLM` explicitly (and record its model), always pass a unique
`--name`, and snapshot the complete output directory before analysis. Do not
claim benchmark or release evidence from the local stub tests.

There is no public fork or redistribution permission for AMB: its repository
does not contain a license. Keep this overlay and any AMB checkout private and
local unless its owner grants separate permission.

The adapter intentionally fails closed on bridge crashes, nonzero exits,
malformed JSON, timeouts, and error responses. `raw_response` is exactly
`{"results": [...]}`; bridge provenance, degraded status/reason, and other
top-level internals are not passed into AMB's graded prompt.
