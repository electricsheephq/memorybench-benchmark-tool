"""AMB adapter for the hermes-lcm JSONL bridge.

Copy this module and :mod:`locomo_mapper` into a local AMB clone, then apply:

    from memory.hermes_lcm_provider import HermesLcmProvider
    PROVIDERS["hermes-lcm"] = HermesLcmProvider

The adapter intentionally has no AMB import.  AMB is optional at development
time, and its ``Document`` object is used structurally here.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from typing import Any, Mapping

from .locomo_mapper import document_to_session


@dataclass
class Document:
    """Small structural stand-in for AMB's ``memory_bench.models.Document``."""

    id: str
    content: str
    user_id: str | None = None
    messages: list[dict[str, Any]] | None = None
    timestamp: str | None = None
    context: str | None = None


class BridgeProtocolError(RuntimeError):
    """Raised when the JSONL bridge violates its response protocol."""


class _BridgeFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class _BridgeHandle:
    """One serialized, crash-loud JSONL subprocess handle."""

    def __init__(self, process: subprocess.Popen[str], tag: str) -> None:
        self.process = process
        self.tag = tag
        self._responses: queue.Queue[str | _BridgeFailure] = queue.Queue()
        self._request_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._dead_error: BaseException | None = None
        self._closed = False

        self._stdout_thread = threading.Thread(
            target=self._read_stdout, name=f"hermes-lcm-{tag}-stdout", daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, name=f"hermes-lcm-{tag}-stderr", daemon=True
        )
        self._stderr_thread.start()

    @property
    def dead_error(self) -> BaseException | None:
        with self._state_lock:
            return self._dead_error

    def _mark_dead(self, error: BaseException) -> None:
        with self._state_lock:
            if self._closed or self._dead_error is not None:
                return
            self._dead_error = error
        self._responses.put(_BridgeFailure(error))

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                line = line.rstrip("\r\n")
                if line.strip():
                    self._responses.put(line)
        except BaseException as exc:  # pragma: no cover - OS-level stream errors
            self._mark_dead(RuntimeError(f"hermes-lcm bridge stdout failed: {exc}"))
        finally:
            if not self._closed:
                code = self.process.poll()
                self._mark_dead(
                    RuntimeError(
                        f"hermes-lcm bridge ({self.tag}) exited "
                        f"(code={code}, signal=None)"
                    )
                )

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        try:
            for _line in self.process.stderr:
                # The bridge deliberately sends diagnostics to stderr.  Drain
                # it so a verbose model/library cannot block the child.
                pass
        except BaseException:
            pass

    def request(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        with self._request_lock:
            dead = self.dead_error
            if dead is not None:
                raise RuntimeError(str(dead)) from dead
            if self._closed:
                raise RuntimeError(f"hermes-lcm bridge ({self.tag}) is closed")
            if self.process.poll() is not None:
                error = RuntimeError(
                    f"hermes-lcm bridge ({self.tag}) exited with code {self.process.returncode}"
                )
                self._mark_dead(error)
                raise error

            line = json.dumps(dict(payload), separators=(",", ":")) + "\n"
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                error = RuntimeError(f"hermes-lcm bridge ({self.tag}) write failed: {exc}")
                self._mark_dead(error)
                raise error from exc

            try:
                item = self._responses.get(timeout=timeout)
            except queue.Empty as exc:
                error = RuntimeError(
                    f"hermes-lcm bridge ({self.tag}) timed out after {timeout:g}s "
                    f"on {payload.get('cmd')}"
                )
                self._mark_dead(error)
                self._terminate()
                raise error from exc

            if isinstance(item, _BridgeFailure):
                raise RuntimeError(str(item.error)) from item.error
            try:
                response = json.loads(item)
            except json.JSONDecodeError as exc:
                error = BridgeProtocolError(
                    f"hermes-lcm bridge sent invalid JSON: {item!r} ({exc})"
                )
                self._mark_dead(error)
                raise error from exc
            if not isinstance(response, dict):
                error = BridgeProtocolError(
                    f"hermes-lcm bridge response was not an object: {response!r}"
                )
                self._mark_dead(error)
                raise error
            if response.get("ok") is False:
                raise RuntimeError(
                    f"hermes-lcm {payload.get('cmd')} failed: {response.get('error')}"
                )
            if response.get("ok") is not True:
                raise BridgeProtocolError(
                    f"hermes-lcm bridge response missing ok=true: {response!r}"
                )
            return response

    def _terminate(self) -> None:
        try:
            self.process.terminate()
        except OSError:
            return
        try:
            self.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            try:
                self.process.kill()
            except OSError:
                pass
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self.process.stdin is not None:
                self.process.stdin.close()
        except OSError:
            pass
        self._terminate()
        for stream in (self.process.stdout, self.process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass


class HermesLcmProvider:
    """A synchronous AMB ``MemoryProvider`` backed by one JSONL bridge."""

    name = "hermes-lcm"
    description = "hermes-lcm lossless context management through its JSONL bridge"
    kind = "local"
    provider = "hermes-lcm"
    variant = "jsonl"
    concurrency = 1

    def __init__(
        self,
        bridge_path: str | os.PathLike[str] | None = None,
        python_executable: str | None = None,
        request_timeout: float | None = None,
        initialize_timeout: float | None = None,
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
        self._workdir: Path | None = None
        self._env: dict[str, str] | None = None
        self._bridge: _BridgeHandle | None = None

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _resolve_bridge_path(self) -> Path:
        raw = self._bridge_override or os.environ.get("HERMES_LCM_BRIDGE_PATH")
        path = Path(raw) if raw else self._repo_root() / "src/providers/hermes-lcm/bridge/hermes_lcm_bridge.py"
        return path if path.is_absolute() else self._repo_root() / path

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        sibling_repo = self._repo_root().parent / "hermes-lcm"
        default_repo = sibling_repo if sibling_repo.is_dir() else self._repo_root()
        env.setdefault("HERMES_LCM_REPO", str(default_repo))
        if self._workdir is None:
            configured = os.environ.get("HERMES_MB_WORKDIR")
            self._workdir = Path(configured) if configured else Path(
                tempfile.mkdtemp(prefix="hermes-lcm-amb-")
            )
        env["HERMES_MB_WORKDIR"] = str(self._workdir)
        env.setdefault("HERMES_MB_PROVIDER", "fastembed")
        return env

    def _require_bridge(self) -> _BridgeHandle:
        bridge = self._bridge
        if bridge is None:
            raise RuntimeError("hermes-lcm provider is not initialized")
        dead = bridge.dead_error
        if dead is not None:
            raise RuntimeError(str(dead)) from dead
        return bridge

    def _container_tag(self, document: Any) -> str:
        user_id = getattr(document, "user_id", None)
        if user_id is None and isinstance(document, Mapping):
            user_id = document.get("user_id")
        return str(user_id or "default")

    def initialize(self) -> None:
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

        handle = _BridgeHandle(process, "amb")
        self._bridge = handle
        try:
            handle.request({"cmd": "initialize"}, self._initialize_timeout)
        except BaseException:
            handle.close()
            self._bridge = None
            raise

    def prepare(
        self,
        store_dir: Path,
        unit_ids: set[str] | None = None,
        reset: bool = True,
    ) -> None:
        """Implement AMB's optional run setup without importing AMB."""

        desired_workdir = Path(store_dir)
        if self._workdir != desired_workdir:
            if self._bridge is not None:
                self._bridge.close()
                self._bridge = None
            self._workdir = desired_workdir
        if not reset:
            return
        self.initialize()
        bridge = self._require_bridge()
        for unit_id in sorted(unit_ids or set()):
            bridge.request(
                {"cmd": "clear", "containerTag": str(unit_id)}, self._request_timeout
            )

    def ingest(self, documents: list[Document]) -> None:
        bridge = self._require_bridge()
        for document in documents:
            bridge.request(
                {
                    "cmd": "ingest",
                    "containerTag": self._container_tag(document),
                    "session": document_to_session(document),
                },
                self._request_timeout,
            )

    async def async_ingest(self, documents: list[Document]) -> None:
        """Match AMB's optional async wrapper without importing AMB."""

        await asyncio.to_thread(self.ingest, documents)

    def retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict[str, Any]]:
        bridge = self._require_bridge()
        payload: dict[str, Any] = {
            "cmd": "search",
            "containerTag": str(user_id or "default"),
            "query": str(query),
            "limit": int(k),
        }
        if query_timestamp is not None:
            # Current bridge versions ignore this optional field; retaining it
            # makes the adapter forward-compatible with date-aware bridges.
            payload["queryTimestamp"] = query_timestamp
        response = bridge.request(payload, self._request_timeout)
        results = response.get("results")
        if not isinstance(results, list):
            raise BridgeProtocolError(
                f"hermes-lcm search response missing list results: {response!r}"
            )

        documents: list[Document] = []
        for index, result in enumerate(results):
            if not isinstance(result, Mapping) or "content" not in result:
                raise BridgeProtocolError(f"invalid hermes-lcm search result: {result!r}")
            metadata = result.get("metadata")
            metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
            result_id = (
                metadata_dict.get("store_id")
                or metadata_dict.get("node_id")
                or metadata_dict.get("session_id")
                or f"{user_id or 'default'}:{index}"
            )
            documents.append(
                Document(
                    id=str(result_id),
                    content=str(result["content"]),
                    user_id=user_id,
                    timestamp=(
                        str(metadata_dict["date"])
                        if metadata_dict.get("date") is not None
                        else None
                    ),
                    context=None,
                )
            )

        # AMB may serialize raw_response into the graded answer prompt.  Never
        # expose bridge provenance/degraded/internal fields there — neither the
        # top-level ones nor per-result internals (score/arms/store_id/
        # chunk_span/...): the graded surface carries only what our own harness
        # renders to a reader.
        sanitized = []
        for result in results:
            metadata = result.get("metadata")
            metadata_dict = dict(metadata) if isinstance(metadata, Mapping) else {}
            kept = {
                key: metadata_dict[key]
                for key in ("session_id", "date", "timestamp", "role", "kind")
                if metadata_dict.get(key) is not None
            }
            sanitized.append({"content": str(result["content"]), "metadata": kept})
        return documents, {"results": sanitized}

    async def async_retrieve(
        self,
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict[str, Any]]:
        """Match AMB's optional async retrieval wrapper."""

        return await asyncio.to_thread(self.retrieve, query, k, user_id, query_timestamp)

    def retrieve_by_steps(
        self,
        steps: list[int],
        query: str,
        k: int = 10,
        user_id: str | None = None,
        query_timestamp: str | None = None,
    ) -> tuple[list[Document], dict[str, Any]]:
        """LoCoMo has no step-scoped retrieval verb; use ordinary search."""

        del steps
        return self.retrieve(query, k, user_id, query_timestamp)

    def cleanup(self) -> None:
        bridge, self._bridge = self._bridge, None
        if bridge is not None:
            bridge.close()


__all__ = ["BridgeProtocolError", "Document", "HermesLcmProvider"]
