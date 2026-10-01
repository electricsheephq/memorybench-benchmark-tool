#!/usr/bin/env bash
# Append output to WATCHDOG_LOG (default data/runs/<run-id>/run.log).
# STALL_MIN=20, MAX_RESUMES=3; WATCHDOG_POLL_S=10, WATCHDOG_TERM_GRACE_S=30.
# WATCHDOG_PYTHON selects a local Python 3. No dependencies or shell eval.
set -euo pipefail
exec "${WATCHDOG_PYTHON:-python3}" - "$@" <<'PY'
import datetime
import fcntl
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def fail(message):
    sys.exit("watchdog: " + message)


args = sys.argv[1:]
if len(args) < 3 or args[1] != "--" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args[0]):
    fail("usage: run-with-watchdog.sh <run-id> -- <command ... -r run-id>")
run_id, command = args[0], args[2:]
run_ids = [command[i + 1] for i, arg in enumerate(command[:-1]) if arg in {"-r", "--run-id"}]
if run_ids != [run_id] or "--force" in command:
    fail("command must contain exactly one matching -r/--run-id; --force prevents checkpoint resume")
try:
    stall_s = float(os.environ.get("STALL_MIN", "20")) * 60
    poll_s = float(os.environ.get("WATCHDOG_POLL_S", "10"))
    grace_s = float(os.environ.get("WATCHDOG_TERM_GRACE_S", "30"))
    max_resumes = int(os.environ.get("MAX_RESUMES", "3"))
    if any(not math.isfinite(v) or v <= 0 for v in (stall_s, poll_s, grace_s)) or max_resumes < 0:
        raise ValueError()
except ValueError:
    fail("timings must be finite and positive; MAX_RESUMES must be a non-negative integer")
log_path = Path(os.environ.get("WATCHDOG_LOG", f"data/runs/{run_id}/run.log"))
log_path.parent.mkdir(parents=True, exist_ok=True)
actions = log_path.with_suffix(log_path.suffix + ".watchdog.log").open("a", buffering=1)
try:
    fcntl.flock(actions, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    fail("another watchdog owns this run log")


def record(message):
    utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{utc} run={run_id} {message}"
    print(line, file=actions, flush=True)
    print(line, file=sys.stderr, flush=True)


def group_exists(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_group(proc):
    pgid = proc.pid  # start_new_session makes this exact child the group leader.
    if group_exists(pgid):
        record(f"TERM pgid={pgid}")
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        end = time.monotonic() + grace_s
        while group_exists(pgid) and time.monotonic() < end:
            proc.poll()  # Reap the leader while waiting for descendants.
            time.sleep(min(0.1, grace_s))
        if group_exists(pgid):
            record(f"KILL pgid={pgid}")
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    proc.wait()


def interrupted(signum, _frame):
    raise SystemExit(128 + signum)


for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
    signal.signal(signum, interrupted)
for attempt in range(max_resumes + 1):
    with log_path.open("ab", buffering=0) as output:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            log_path.with_suffix(log_path.suffix + ".pgid").write_text(str(proc.pid) + "\n")
            record(f"START attempt={attempt} pgid={proc.pid}")
            size, last_growth = log_path.stat().st_size, time.monotonic()
            stalled = False
            while proc.poll() is None:
                time.sleep(poll_s)
                current = log_path.stat().st_size
                if current != size:
                    size, last_growth = current, time.monotonic()
                if time.monotonic() - last_growth < stall_s or proc.poll() is not None:
                    continue
                try:
                    cpu_result = subprocess.run(["ps", "-o", "%cpu=", "-p", str(proc.pid)],
                                                capture_output=True, text=True, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    record(f"CPU unavailable pgid={proc.pid}; observe")
                    continue
                try:
                    cpu = float(cpu_result.stdout.strip())
                except ValueError:
                    record(f"CPU unavailable pgid={proc.pid}; observe")
                    continue
                if cpu_result.returncode == 0 and cpu < 1.0:
                    record(f"STALL pgid={proc.pid} cpu={cpu:.2f} idle_s={time.monotonic() - last_growth:.2f}")
                    stalled = True
                    break
        finally:
            stop_group(proc)  # Also clean descendants after normal leader exit.
    if not stalled:
        record(f"EXIT code={proc.returncode}")
        sys.exit(proc.returncode if proc.returncode >= 0 else 128 - proc.returncode)
    if attempt == max_resumes:
        record(f"EXHAUSTED resumes={max_resumes}")
        sys.exit(124)
    record(f"RESUME number={attempt + 1}")
PY
