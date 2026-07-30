"""AMA-Bench method backed by the shared hermes-lcm JSONL bridge.

This module has no AMA-Bench import.  In a local AMA clone, register it before
the benchmark entry point is loaded with the clone's normal registry overlay::

    from integrations.ama.hermes_lcm_method import HermesLcmMethod
    from src.method.base_method import BaseMethod
    from src.method_register import register_method

    # The current AMA registry checks nominal inheritance.  Keep this overlay
    # structural and add the clone-local nominal shim only at registration.
    class RegisteredHermesLcmMethod(HermesLcmMethod, BaseMethod):
        pass

    register_method("hermes_lcm", RegisteredHermesLcmMethod)

Transport is intentionally reused from the AMB integration.  In particular,
``_BridgeHandle`` owns the serialized JSONL request/response protocol and its
crash-loud reader thread; this adapter only supplies AMA's method contract and
the AMA-specific session payload.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
from typing import Any

from integrations.amb.hermes_lcm_provider import (
    BridgeProtocolError,
    _BridgeHandle,
)

from .trajectory_parser import parse_trajectory


class HermesLcmMethod:
    """Structural AMA ``BaseMethod`` implementation using hermes-lcm memory."""

    def __init__(
        self,
        bridge_path: str | os.PathLike[str] | None = None,
        python_executable: str | None = None,
        request_timeout: float | None = None,
        initialize_timeout: float | None = None,
        workdir: str | os.PathLike[str] | None = None,
        search_limit: int | None = None,
    ) -> None:
        self._bridge_override = Path(bridge_path) if bridge_path is not None else None
        self._python_executable = python_executable or os.environ.get(
            "HERMES_LCM_PYTHON", sys.executable
        )
        self._request_timeout = request_timeout or float(
            os.environ.get("HERMES_LCM_REQUEST_TIMEOUT_SECONDS", "180")
        )
        self._initialize_timeout = initialize_timeout or float(
            os.environ.get("HERMES_LCM_INITIALIZE_TIMEOUT_SECONDS", "300")
        )
        configured_workdir = workdir or os.environ.get("HERMES_MB_WORKDIR")
        self._workdir = Path(configured_workdir) if configured_workdir else None
        self._owns_workdir = False
        self._search_limit = search_limit or int(
            os.environ.get("HERMES_LCM_AMA_SEARCH_LIMIT", "25")
        )
        if self._search_limit <= 0:
            raise ValueError("search_limit must be positive")
        self._bridge: _BridgeHandle | None = None
        self._containers: set[str] = set()

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_bridge_path(self) -> Path:
        raw = self._bridge_override or os.environ.get("HERMES_LCM_BRIDGE_PATH")
        path = (
            Path(raw)
            if raw
            else self._repo_root() / "src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"
        )
        return path if path.is_absolute() else self._repo_root() / path

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        sibling_repo = self._repo_root().parent / "hermes-lcm"
        default_repo = sibling_repo if sibling_repo.is_dir() else self._repo_root()
        env.setdefault("HERMES_LCM_REPO", str(default_repo))

        if self._workdir is None:
            # Keep bridge-owned scratch on the requested LEXAR evidence volume,
            # rather than silently creating benchmark state under /tmp.
            artifact_root = Path(
                "/Volumes/LEXAR/Codex/session-notes/2026-07-30/"
                "hermes-ama-adapter/artifacts"
            )
            artifact_root.mkdir(parents=True, exist_ok=True)
            self._workdir = Path(
                tempfile.mkdtemp(prefix="hermes-lcm-ama-", dir=str(artifact_root))
            )
            self._owns_workdir = True
        self._workdir.mkdir(parents=True, exist_ok=True)
        env["HERMES_MB_WORKDIR"] = str(self._workdir)
        env.setdefault("HERMES_MB_PROVIDER", "fastembed")
        return env

    def _require_bridge(self) -> _BridgeHandle:
        bridge = self._bridge
        if bridge is None:
            raise RuntimeError("hermes-lcm AMA method is not initialized")
        dead = bridge.dead_error
        if dead is not None:
            raise RuntimeError(str(dead)) from dead
        return bridge

    def initialize(self) -> None:
        """Start the shared bridge and complete its initialization handshake."""

        if self._bridge is not None and self._bridge.dead_error is None:
            return
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None

        bridge_path = self._resolve_bridge_path()
        if not bridge_path.is_file():
            raise FileNotFoundError(f"hermes-lcm bridge script not found at {bridge_path}")
        env = self._build_env()
        try:
            process = subprocess.Popen(
                [self._python_executable, str(bridge_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to start hermes-lcm bridge: {exc}") from exc

        handle = _BridgeHandle(process, "ama")
        self._bridge = handle
        try:
            handle.request({"cmd": "initialize"}, self._initialize_timeout)
        except BaseException:
            handle.close()
            self._bridge = None
            raise

    def _ensure_bridge(self) -> _BridgeHandle:
        if self._bridge is None:
            self.initialize()
        return self._require_bridge()

    @staticmethod
    def _new_container_tag() -> str:
        return f"ama-{uuid.uuid4().hex}"

    def memory_construction(self, traj_text: str, task: str = "") -> str:
        """Ingest one flattened AMA episode and return its container handle."""

        # Parse before starting or mutating the bridge: format drift must fail
        # before any partial episode can be persisted.
        steps = parse_trajectory(traj_text)
        bridge = self._ensure_bridge()
        container_tag = self._new_container_tag()
        messages: list[dict[str, str]] = []
        for step in steps:
            messages.append({"role": "assistant", "content": step.action})
            messages.append({"role": "user", "content": step.observation})

        bridge.request(
            {
                "cmd": "ingest",
                "containerTag": container_tag,
                "session": {
                    "sessionId": container_tag,
                    "metadata": {"task": str(task)},
                    "messages": messages,
                },
            },
            self._request_timeout,
        )
        self._containers.add(container_tag)
        return container_tag

    @staticmethod
    def render_evidence(results: Sequence[Mapping[str, Any]]) -> str:
        """Render only reader-visible content and optional date lines."""

        blocks: list[str] = []
        for result in results:
            if not isinstance(result, Mapping) or "content" not in result:
                raise BridgeProtocolError(f"invalid hermes-lcm search result: {result!r}")
            metadata = result.get("metadata")
            metadata_dict = metadata if isinstance(metadata, Mapping) else {}
            date = next(
                (
                    metadata_dict[key]
                    for key in ("date", "timestamp", "formattedDate")
                    if metadata_dict.get(key) is not None
                ),
                None,
            )
            lines: list[str] = []
            if date is not None:
                lines.append(f"Date: {date}")
            lines.append(str(result["content"]))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def memory_retrieve(self, memory: str, question: str) -> str:
        """Search one episode container and return compact evidence text."""

        if not isinstance(memory, str) or not memory:
            raise ValueError("memory must be a non-empty containerTag string")
        bridge = self._ensure_bridge()
        response = bridge.request(
            {
                "cmd": "search",
                "containerTag": memory,
                "query": str(question),
                "limit": self._search_limit,
            },
            self._request_timeout,
        )
        results = response.get("results")
        if not isinstance(results, list):
            raise BridgeProtocolError(
                f"hermes-lcm search response missing list results: {response!r}"
            )
        return self.render_evidence(results)

    def cleanup(self) -> None:
        """Clear every episode container, then close the shared bridge."""

        bridge, self._bridge = self._bridge, None
        clear_error: BaseException | None = None
        if bridge is not None:
            for container_tag in sorted(self._containers):
                try:
                    bridge.request(
                        {"cmd": "clear", "containerTag": container_tag},
                        self._request_timeout,
                    )
                except BaseException as exc:  # close must still run
                    if clear_error is None:
                        clear_error = exc
            bridge.close()
        self._containers.clear()
        if self._owns_workdir and self._workdir is not None:
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
            self._owns_workdir = False
        if clear_error is not None:
            raise clear_error

    def __enter__(self) -> "HermesLcmMethod":
        self.initialize()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            self.cleanup()
        except BaseException:
            if exc_type is None:
                raise
        return False


__all__ = ["HermesLcmMethod"]
