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


def cpu_seconds(text):
    # ps TIME: macOS "m:ss.xx" / "h:mm:ss.xx", procps "[d-]hh:mm:ss".
    days, _, clock = text.rpartition("-")
    total = 0.0
    for part in clock.split(":"):
        total = total * 60 + float(part)
    return total + (int(days) * 86400 if days else 0)


def group_snapshot(pgid):
    """(live members, summed CPU seconds, live pids) of one process group; zombies are not live."""
    out = subprocess.run(["ps", "-A", "-o", "pgid=,pid=,stat=,time="],
                         capture_output=True, text=True, timeout=5, check=True).stdout
    live, cpu, pids = 0, 0.0, []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0] == str(pgid) and not fields[2].startswith("Z"):
            live += 1
            cpu += cpu_seconds(fields[3])
            pids.append(fields[1])
    return live, cpu, frozenset(pids)


def group_exists(pgid):
    try:
        return group_snapshot(pgid)[0] > 0
    except (OSError, ValueError, subprocess.SubprocessError):
        try:  # Fall back to the signal probe (counts zombies) rather than assume the group is gone.
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
            samples = []  # (monotonic, group CPU seconds, live pids) over the last stall window
            while proc.poll() is None:
                time.sleep(poll_s)
                now = time.monotonic()
                current = log_path.stat().st_size
                if current != size:
                    size, last_growth = current, now
                try:
                    _live, cpu_now, pids = group_snapshot(proc.pid)
                except (OSError, ValueError, subprocess.SubprocessError):
                    record(f"CPU unavailable pgid={proc.pid}; observe")
                    continue
                # A member that exits takes its CPU time out of the sum; restart the window
                # whenever membership changes so a shrinking sum never reads as idle.
                if samples and (pids != samples[-1][2] or cpu_now < samples[-1][1]):
                    samples = []
                samples.append((now, cpu_now, pids))
                while len(samples) > 1 and now - samples[1][0] >= stall_s:
                    samples.pop(0)
                if now - last_growth < stall_s or now - samples[0][0] < stall_s or proc.poll() is not None:
                    continue
                # Recent group CPU, not a lifetime average: a process that worked hard and then
                # deadlocked reads near 0 here once a full quiet window has passed.
                cpu = 100.0 * (samples[-1][1] - samples[0][1]) / (now - samples[0][0])
                if cpu < 1.0:
                    record(f"STALL pgid={proc.pid} cpu={cpu:.2f} idle_s={now - last_growth:.2f}")
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
