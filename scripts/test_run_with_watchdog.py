from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

WATCHDOG = Path(__file__).with_name("run-with-watchdog.sh")
FAKE = r'''
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(os.environ["FAKE_ROOT"])
attempts = root / "attempts.jsonl"
attempt = len(attempts.read_text().splitlines()) if attempts.exists() else 0
with attempts.open("a") as output:
    output.write(json.dumps(sys.argv[1:]) + "\n")
if os.environ.get("FAKE_ALWAYS_STALL") or attempt == 0:
    child = subprocess.Popen([
        sys.executable, "-c",
        "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"
    ])
    (root / f"child-{attempt}.pid").write_text(str(child.pid))
    # Readiness is printed after the child has installed its TERM handler.
    time.sleep(0.2)
    print("fake command ready", flush=True)
    time.sleep(120)
print("checkpoint resumed", flush=True)
'''


FAKE_PS = """#!PYTHON
import os, subprocess, sys
from pathlib import Path
mode = os.environ.get("FAKE_CPU", "")
if mode == "unavailable":
    sys.exit(1)
out = subprocess.run(["/bin/ps", *sys.argv[1:]], capture_output=True, text=True).stdout
if mode == "busy":
    counter = Path(os.environ["FAKE_ROOT"]) / "ps-calls"
    calls = int(counter.read_text()) + 1 if counter.exists() else 1
    counter.write_text(str(calls))
    out = "".join(" ".join(line.split()[:2] + [f"0:{calls * 10}.00"]) + "\\n" for line in out.splitlines())
sys.stdout.write(out)
"""


def env_for(tmp_path):
    # Real process-group state from /bin/ps; FAKE_CPU=busy makes the group's CPU time climb on
    # every call (recent work), FAKE_CPU=unavailable makes ps fail.
    tools = tmp_path / "tools"
    tools.mkdir(exist_ok=True)
    ps = tools / "ps"
    ps.write_text(FAKE_PS.replace("PYTHON", sys.executable))
    ps.chmod(0o755)
    return {
        **os.environ,
        "PATH": str(tools) + os.pathsep + os.environ.get("PATH", ""),
        "WATCHDOG_PYTHON": sys.executable,
        "WATCHDOG_LOG": str(tmp_path / "run.log"),
        "STALL_MIN": "0.04",
        "MAX_RESUMES": "1",
        "WATCHDOG_POLL_S": "0.1",
        "WATCHDOG_TERM_GRACE_S": "0.2",
        "FAKE_ROOT": str(tmp_path),
    }


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def wait_gone(pid):
    deadline = time.monotonic() + 5
    while not gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return gone(pid)


def run_fake(tmp_path, **overrides):
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    env = env_for(tmp_path)
    env.update(overrides)
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-u", str(fake), "-r", "fake-run"]
    return subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)


def test_stall_kills_group_child_and_resumes_same_command(tmp_path):
    # An unrelated sleeping process must survive the exact-PGID cleanup.
    sentinel = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        result = run_fake(tmp_path)
        assert result.returncode == 0, result.stderr
        actions = (tmp_path / "run.log.watchdog.log").read_text()
        starts = [int(pid) for pid in re.findall(r"START attempt=\d+ pgid=(\d+)", actions)]
        assert len(starts) == 2 and starts[0] != starts[1]
        assert f"TERM pgid={starts[0]}" in actions
        assert f"KILL pgid={starts[0]}" in actions
        assert "RESUME number=1" in actions
        assert all(re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ ", line) for line in actions.splitlines())
        child = int((tmp_path / "child-0.pid").read_text())
        assert wait_gone(child), "TERM-ignoring descendant must be gone"
        assert all(wait_gone(pid) for pid in starts), "both leaders must be reaped"
        assert sentinel.poll() is None
        attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
        assert attempts == [["-r", "fake-run"], ["-r", "fake-run"]]
        assert "checkpoint resumed" in (tmp_path / "run.log").read_text()
        assert int((tmp_path / "run.log.pgid").read_text()) == starts[-1]
        receipt_path = os.environ.get("WATCHDOG_TEST_RECEIPT")
        if receipt_path:
            Path(receipt_path).write_text(json.dumps({
                "leaders": starts, "child_pid": child, "child_gone": True,
                "leaders_reaped": True, "unrelated_child_survived": True,
                "attempts": len(attempts), "resumes": 1,
                "same_command_and_run_id": True, "utc_actions": True,
                "cpu_source": "deterministic ps fixture (0.0 percent)",
            }, indent=2) + "\n")
    finally:
        sentinel.terminate()
        sentinel.wait(timeout=5)


def test_resume_limit_is_bounded_and_descendants_are_gone(tmp_path):
    result = run_fake(tmp_path, FAKE_ALWAYS_STALL="1")
    assert result.returncode == 124, result.stderr
    actions = (tmp_path / "run.log.watchdog.log").read_text()
    assert actions.count("START attempt=") == 2
    assert actions.count("RESUME number=") == 1
    assert "EXHAUSTED resumes=1" in actions
    assert all(wait_gone(int(pid.read_text())) for pid in tmp_path.glob("child-*.pid"))


@pytest.mark.parametrize("arguments", [["-r", "wrong"], ["-r", "fake-run", "--force"], []])
def test_refuses_commands_that_cannot_resume_run_id(tmp_path, arguments):
    marker = tmp_path / "must-not-run"
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-c",
               f"from pathlib import Path; Path({str(marker)!r}).touch()", *arguments]
    result = subprocess.run(command, env=env_for(tmp_path), capture_output=True, text=True, check=False, timeout=5)
    assert result.returncode != 0 and not marker.exists()


def test_log_growth_prevents_stall(tmp_path):
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-u", "-c",
               "import time; [(print('progress',flush=True),time.sleep(0.2)) for _ in range(10)]",
               "-r", "fake-run"]
    env = env_for(tmp_path)
    env["STALL_MIN"] = "0.01"
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0
    assert "STALL" not in (tmp_path / "run.log.watchdog.log").read_text()


def test_normal_exit_cleans_leftover_child(tmp_path):
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-c",
               "import subprocess,sys; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(child.pid,flush=True)",
               "-r", "fake-run"]
    result = subprocess.run(command, env=env_for(tmp_path), capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0
    child = int((tmp_path / "run.log").read_text().strip())
    assert wait_gone(child)


def test_work_then_deadlock_is_a_stall(tmp_path):
    # CPU time spent before the hang must not mask it (no lifetime average).
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-c",
               "import time\nend=time.monotonic()+1.5\nwhile time.monotonic()<end: pass\ntime.sleep(120)",
               "-r", "fake-run"]
    env = env_for(tmp_path)
    env.update(STALL_MIN="0.02", MAX_RESUMES="0")
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=20)
    actions = (tmp_path / "run.log.watchdog.log").read_text()
    assert "STALL" in actions and "EXHAUSTED" in actions
    assert result.returncode == 124


@pytest.mark.parametrize("cpu", ["busy", "unavailable"])
def test_high_or_unavailable_cpu_without_log_growth_is_observed(tmp_path, cpu):
    command = [str(WATCHDOG), "fake-run", "--", sys.executable, "-c",
               "import time; time.sleep(2)", "-r", "fake-run"]
    env = env_for(tmp_path)
    env.update(FAKE_CPU=cpu, STALL_MIN="0.01")
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 0
    actions = (tmp_path / "run.log.watchdog.log").read_text()
    assert "STALL" not in actions and "RESUME" not in actions
