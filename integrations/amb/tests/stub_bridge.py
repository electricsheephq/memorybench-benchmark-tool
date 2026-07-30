#!/usr/bin/env python3
"""Tiny in-process test double for hermes_lcm_bridge.py."""

from __future__ import annotations

import json
import os
import sys


mode = os.environ.get("STUB_BRIDGE_MODE", "normal")
if mode == "crash":
    raise SystemExit(7)
if mode == "nonzero":
    raise SystemExit(23)

last_session = None
for line in sys.stdin:
    if mode == "malformed":
        print("{this is not json", flush=True)
        break
    request = json.loads(line)
    command = request.get("cmd")
    if command == "initialize":
        response = {"ok": True, "provider": "stub", "model": "none", "dim": 0}
    elif command == "ingest":
        last_session = request["session"]
        response = {"ok": True, "documentIds": [last_session["sessionId"]]}
    elif command == "search":
        if last_session is not None:
            results = [
                {
                    "content": json.dumps(last_session, sort_keys=True),
                    "metadata": {"session_id": last_session["sessionId"]},
                }
            ]
        else:
            results = [{"content": "stub memory", "metadata": {"session_id": "stub"}}]
        response = {
            "ok": True,
            "results": results,
            "provenance": "must-not-reach-raw-response",
            "degraded": True,
            "degraded_reason": "must-not-reach-raw-response",
            "internal": {"secret": "must-not-reach-raw-response"},
        }
    elif command == "clear":
        last_session = None
        response = {"ok": True}
    else:
        response = {"ok": False, "error": f"unknown command: {command}"}
    print(json.dumps(response), flush=True)
