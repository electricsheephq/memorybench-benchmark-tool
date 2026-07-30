#!/usr/bin/env python3
"""JSON-line bridge exposing hermes-lcm as a memorybench Provider backend.

The TypeScript ``HermesLcmProvider`` spawns this script in ``serve`` mode and
speaks newline-delimited JSON over stdin/stdout: one request object per line,
one response object per line. It implements the four stateful provider methods
(initialize / ingest / search / clear); ``awaitIndexing`` is a no-op on the TS
side because ingest is fully synchronous here.

Design contract (faithful to ``benchmarking/longmemeval.py`` in the hermes-lcm
repo, which this imports rather than reimplements):

* ``ingest`` accumulates ONE harness session at a time into a per-container LCM
  store on disk (the harness calls ``provider.ingest([session], ...)`` in a loop),
  preserving session ids/order, building the SAME deterministic per-session
  summary the in-house harness uses, recording summary + conversational-chunk
  embeddings. Embeds are batched per call.
* ``search`` invokes the PRODUCTION ``tools.lcm_recall`` with its opt-in
  ``detail=answer_ready`` contract through a ``SimpleNamespace`` engine with a
  fresh, dataset-disjoint ``current_session_id`` (so the scope prior never
  silently lifts an evidence session). The bridge keeps product-returned
  expanded content and exact-read hydrates any remaining selected hits from the
  already-open store/DAG; it never performs an additional retrieval search.

Fairness: the bridge only ever sees what the harness hands it (the session
messages + the query). No dataset-specific logic, no evidence peeking.

The hermes-lcm plugin repo is NEVER modified: it is made importable via the same
``sys.path`` + package-spec bootstrap the repo's own harness uses.

Environment:
    HERMES_LCM_REPO                path to the hermes-lcm checkout (required)
    HERMES_MB_WORKDIR              base dir for per-container LCM dbs (required)
    HERMES_MB_PROVIDER            embedding provider: fastembed (default) | voyage
    HERMES_MB_MODEL              embedding model id (default per provider)
    HERMES_MB_ANSWER_READY_CONTENT_CHARS
                                  per-result exact-read cap (default 2400)
    LCM_LONGMEMEVAL_FASTEMBED_CACHE  fastembed model cache dir
    VOYAGE_API_KEY               required when HERMES_MB_PROVIDER=voyage
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_DEFAULT_MODELS = {
    "fastembed": "BAAI/bge-small-en-v1.5",
    "voyage": "voyage-context-3",
}
_DEFAULT_ANSWER_READY_CONTENT_CHARS = 2_400

# Preserve the real stdout for protocol responses, then redirect stdout to
# stderr so any library chatter (model downloads, warnings) can never corrupt
# the newline-delimited JSON channel.
_RESPONSE_OUT = sys.stdout
sys.stdout = sys.stderr


def _log(message: str) -> None:
    print(f"[hermes-lcm-bridge] {message}", file=sys.stderr, flush=True)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-._") or "container"


def _question_date(value: Any) -> str | None:
    """Normalize an explicit host question/turn anchor without inventing one."""
    raw = str(value or "").strip()
    match = re.match(r"^(\d{4})[/-](\d{2})[/-](\d{2})(?:\D|$)", raw)
    if match is None:
        return None
    normalized = "-".join(match.groups())
    try:
        date.fromisoformat(normalized)
    except ValueError:
        return None
    return normalized


def _positive_int_env(name: str, default: int) -> int:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _content_window(
    content: str,
    *,
    match_start: int,
    match_end: int,
    char_cap: int,
) -> dict[str, Any]:
    """Return a bounded exact-read window with truthful length metadata."""
    content_chars = len(content)
    start = min(max(0, match_start), content_chars)
    end = min(max(start, match_end), content_chars)
    if content_chars <= char_cap:
        offset = 0
    else:
        midpoint = (start + end) // 2
        offset = min(max(0, midpoint - char_cap // 2), content_chars - char_cap)
    bounded = content[offset:offset + char_cap]
    return {
        "content": bounded,
        "content_chars": content_chars,
        "content_offset": offset,
        "content_returned_chars": len(bounded),
        "content_truncated": len(bounded) < content_chars,
    }


def _hydrate_answer_ready_hit(
    hit: dict[str, Any],
    *,
    store: Any,
    dag: Any,
    query: str,
    char_cap: int,
) -> dict[str, Any]:
    """Exact-read an answer-ready hit when product hydration is absent/incomplete."""
    required_metadata = (
        "content_chars",
        "content_returned_chars",
        "content_truncated",
    )
    if hit.get("content") is not None and all(
        hit.get(field) is not None for field in required_metadata
    ):
        return hit

    hydrated = dict(hit)
    if hit.get("kind") != "summary":
        # A pre-existing MESSAGE exact_ref encodes the OLD content window;
        # re-hydration replaces content/offset below, so drop it and let
        # downstream recompute from the delivered content (store_id present).
        # Summary hits keep theirs: re-hydration re-reads the SAME node's
        # summary deterministically, and the summary branch carries only
        # node_id — no store_id to derive a replacement reference, so dropping
        # it leaves the evidence-card path referenceless (it throws).
        hydrated.pop("exact_ref", None)
        hydrated.pop("exact_ref_source", None)
    content = ""
    match_start = 0
    match_end = 0

    if hit.get("kind") == "summary":
        node_id = hit.get("node_id")
        node = dag.get_node(int(node_id)) if node_id is not None else None
        if node is None:
            raise RuntimeError(
                f"cannot hydrate answer-ready summary hit without node_id {node_id!r}"
            )
        content = str(node.summary or "")
        match_end = min(len(content), 300)
        hydrated["content_source"] = "summary"
        hydrated["source"] = hydrated.get("source") or "summary"
    else:
        store_id = hit.get("store_id")
        stored = store.get(int(store_id)) if store_id is not None else None
        if stored is None:
            raise RuntimeError(
                f"cannot hydrate answer-ready message hit without store_id {store_id!r}"
            )
        content = str(stored.get("content") or "")
        span = hit.get("chunk_span") or {}
        try:
            match_start = int(span["char_start"])
            match_end = int(span["char_end"])
        except (KeyError, TypeError, ValueError):
            match_start = content.lower().find(query.lower())
            if match_start < 0:
                match_start = 0
            match_end = match_start + min(max(1, len(query)), 300)
        hydrated["content_source"] = "message"
        hydrated["role"] = stored.get("role")
        hydrated["source"] = stored.get("source") or ""

    hydrated.update(
        _content_window(
            content,
            match_start=match_start,
            match_end=match_end,
            char_cap=char_cap,
        )
    )
    return hydrated


def _metadata_for_recall_hit(
    hit: dict[str, Any], dates: dict[str, str]
) -> dict[str, Any]:
    """Translate only fields already returned by production lcm_recall."""
    session_id = hit.get("session_id")
    metadata: dict[str, Any] = {
        "session_id": session_id,
        "date": dates.get(str(session_id)),
        "kind": hit.get("kind"),
        "score": hit.get("score"),
        "arms": hit.get("arms"),
        "from_current_session": hit.get("from_current_session"),
        "answer_ready": bool(hit.get("content")),
        "content_truncated": bool(hit.get("content_truncated")),
    }
    for facet in (
        "exact_ref",
        "timestamp",
        "role",
        "source",
        "content_source",
        "content_chars",
        "content_offset",
        "content_returned_chars",
        "expand_hint",
    ):
        if hit.get(facet) is not None:
            metadata[facet] = hit.get(facet)
    if hit.get("kind") == "summary":
        metadata["node_id"] = hit.get("node_id")
    else:
        metadata["store_id"] = hit.get("store_id")
        if hit.get("chunk_span"):
            metadata["chunk_span"] = hit.get("chunk_span")
    return metadata


def _exact_ref_for_content(
    store_id: int, source: str, content: str
) -> dict[str, Any] | None:
    """Resolve one already-returned evidence string to a unique exact span."""
    if not content:
        return None
    offset = source.find(content)
    if offset < 0 or source.find(content, offset + 1) >= 0:
        return None
    span_end = offset + len(content)
    return {
        "exact_ref": f"lcm:{store_id}:{offset}-{span_end}",
        "exact_span": {"char_start": offset, "char_end": span_end},
        "exact_ref_source": "deterministic_cached_content_match",
    }


class Bridge:
    def __init__(self) -> None:
        repo = os.environ.get("HERMES_LCM_REPO")
        if not repo:
            raise RuntimeError("HERMES_LCM_REPO is not set")
        self.repo_root = Path(repo).resolve()
        if not self.repo_root.is_dir():
            raise RuntimeError(f"HERMES_LCM_REPO does not exist: {self.repo_root}")

        workdir = os.environ.get("HERMES_MB_WORKDIR")
        if not workdir:
            raise RuntimeError("HERMES_MB_WORKDIR is not set")
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.provider_name = (
            (os.environ.get("HERMES_MB_PROVIDER") or "fastembed").strip().lower()
        )
        if self.provider_name in {"fast-embed"}:
            self.provider_name = "fastembed"
        self.model = os.environ.get("HERMES_MB_MODEL") or _DEFAULT_MODELS.get(
            self.provider_name, ""
        )
        if not self.model:
            raise RuntimeError(
                f"no embedding model for provider {self.provider_name!r}"
            )
        self.answer_ready_content_chars = _positive_int_env(
            "HERMES_MB_ANSWER_READY_CONTENT_CHARS",
            _DEFAULT_ANSWER_READY_CONTENT_CHARS,
        )

        # Make the plugin importable exactly the way the repo's own harness does.
        if str(self.repo_root) not in sys.path:
            sys.path.insert(0, str(self.repo_root))
        from benchmarking.longmemeval import (  # noqa: E402
            _ensure_hermes_lcm_package,
            deterministic_session_summary,
            resolve_harness_provider,
        )

        _ensure_hermes_lcm_package()
        self._deterministic_session_summary = deterministic_session_summary
        self._resolve_harness_provider = resolve_harness_provider

        # Lazily populated on initialize().
        self.embedder: Any = None
        self.dim: int = 0
        # Per-container monotonic session order (recency prior in lcm_recall).
        self._order: dict[str, int] = {}

    # -- lifecycle ------------------------------------------------------------

    def initialize(self, _req: dict[str, Any]) -> dict[str, Any]:
        if self.provider_name == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("HERMES_MB_PROVIDER=voyage but VOYAGE_API_KEY is unset")
        # Warm the embedder once so the model download/load happens here, not
        # inside a per-question path, and .dim is populated.
        self._ensure_embedder()
        _log(
            f"initialized provider={self.provider_name} model={self.model} dim={self.dim} "
            f"workdir={self.workdir}"
        )
        return {
            "ok": True,
            "provider": self.provider_name,
            "model": self.model,
            "dim": self.dim,
            "embeddings_enabled": True,
        }

    def _ensure_embedder(self) -> None:
        if self.embedder is not None:
            return
        if self.provider_name == "voyage" and not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("HERMES_MB_PROVIDER=voyage but VOYAGE_API_KEY is unset")
        self.embedder = self._resolve_harness_provider(self.provider_name, self.model)
        self.dim = int(self.embedder.dim)
        self.model = self.embedder.model_id

    # -- helpers --------------------------------------------------------------

    def _db_path(self, container_tag: str) -> Path:
        return self.workdir / f"{_safe(container_tag)}.db"

    def _dates_path(self, container_tag: str) -> Path:
        # A sidecar mapping session_id -> harness-provided session date. The
        # plugin's append_batch stamps ingest wall-clock time (it takes no
        # per-message timestamp and must not be modified), so the real session
        # date -- data the harness gives EVERY provider -- is preserved here and
        # surfaced onto each search hit's metadata for temporal questions.
        return self.workdir / f"{_safe(container_tag)}.dates.json"

    def _load_dates(self, container_tag: str) -> dict[str, Any]:
        path = self._dates_path(container_tag)
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _config(self, db_path: Path):
        from hermes_lcm.config import LCMConfig

        return LCMConfig(
            database_path=str(db_path),
            embeddings_enabled=True,
            embedding_provider=self.provider_name,
            embedding_model=self.model,
        )

    # -- ingest ---------------------------------------------------------------

    def ingest(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.embedder is None:
            raise RuntimeError("ingest before initialize")
        container_tag = str(req["containerTag"])
        session = req["session"]
        session_id = str(session["sessionId"])
        session_meta = session.get("metadata") or {}
        session_date = session_meta.get("date") or session_meta.get("formattedDate")
        messages = [
            {
                "role": str(m.get("role", "user")),
                "content": str(m.get("content", "")),
            }
            for m in session.get("messages", [])
        ]

        from hermes_lcm.chunking import iter_message_chunks
        from hermes_lcm.dag import SummaryDAG, SummaryNode
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import EmbeddingIdentity, VectorStore

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        # Opening these bootstraps the schema on first touch and re-opens
        # idempotently thereafter, so successive sessions accumulate.
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        try:
            vector_store.register_profile(self.model, self.provider_name, self.dim)
            identity = vector_store.capture_identity(
                self.model, provider=self.provider_name
            )
            vector_store.register_profile(
                self.model, self.provider_name, self.dim, task="chunk"
            )
            chunk_identity = EmbeddingIdentity.canonical(
                self.provider_name,
                self.model,
                "",
                self.dim,
                "float32",
                "little",
                "chunk",
            )

            store_ids: list[int] = []
            if messages:
                store_ids = store.append_batch(
                    session_id, messages, source="benchmark", conversation_id=session_id
                )
                rows = [
                    {"store_id": sid, "role": m["role"], "content": m["content"]}
                    for sid, m in zip(store_ids, messages)
                ]
                chunk_texts: list[str] = []
                chunk_meta: list[Any] = []
                for chunk in iter_message_chunks(rows, policy="conversational"):
                    chunk_texts.append(chunk.text)
                    chunk_meta.append(chunk)
                if chunk_texts:
                    chunk_vectors = self.embedder.embed_documents(chunk_texts)
                    for chunk, vector in zip(chunk_meta, chunk_vectors):
                        vector_store.record_chunk_embedding(
                            chunk.chunk_id,
                            self.model,
                            vector,
                            store_id=chunk.store_id,
                            chunk_index=chunk.chunk_index,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                            token_estimate=chunk.token_estimate,
                            identity=chunk_identity,
                        )

            summary_text = self._deterministic_session_summary(messages)
            order = self._order.get(container_tag, 0) + 1
            self._order[container_tag] = order
            node_id = dag.add_node(
                SummaryNode(
                    session_id=session_id,
                    depth=0,
                    summary=summary_text,
                    token_count=len(summary_text.split()),
                    source_token_count=sum(len(m["content"].split()) for m in messages),
                    source_type="messages",
                    created_at=float(order),
                )
            )
            summary_vector = self.embedder.embed_documents([summary_text])[0]
            vector_store.record_embedding(
                str(node_id), "summary", self.model, summary_vector, identity=identity
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        if session_date:
            dates = self._load_dates(container_tag)
            dates[session_id] = session_date
            self._dates_path(container_tag).write_text(
                json.dumps(dates), encoding="utf-8"
            )

        return {
            "ok": True,
            "documentIds": [str(sid) for sid in store_ids] or [session_id],
        }

    # -- search ---------------------------------------------------------------

    def search(self, req: dict[str, Any]) -> dict[str, Any]:
        if self.embedder is None:
            raise RuntimeError("search before initialize")
        container_tag = str(req["containerTag"])
        query = str(req.get("query", ""))
        limit = int(req.get("limit", 25))

        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore
        import hermes_lcm.tools as lcm_tools

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)  # noqa: F841 (keeps db warm)
        dates = self._load_dates(container_tag)
        try:
            # A probe current-session id disjoint from any dataset session id
            # (the harness uses "<qid>-session-<i>"); the scope prior may boost
            # the current conversation, so it must NOT be an evidence session.
            fresh_session = f"__hermes_lcm_recall_probe__{container_tag}"
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _hermes_home=str(self.workdir),
                current_session_id=fresh_session,
            )
            cache_key = (
                self.provider_name.strip().lower(),
                str(self.embedder.model_id).strip(),
            )
            engine._lcm_embedding_provider_cache = (cache_key, self.embedder)

            payload = json.loads(
                lcm_tools.lcm_recall(
                    {"query": query, "limit": limit, "detail": "answer_ready"},
                    engine=engine,
                )
            )
            if "error" in payload:
                raise RuntimeError(f"lcm_recall error: {payload['error']}")

            results: list[dict[str, Any]] = []
            bridge_hydrated_count = 0
            for raw_hit in payload.get("hits", [])[:limit]:
                hit = _hydrate_answer_ready_hit(
                    raw_hit,
                    store=store,
                    dag=dag,
                    query=query,
                    char_cap=self.answer_ready_content_chars,
                )
                if (
                    hit is not raw_hit
                    and hit.get("content") is not None
                    and hit.get("content_chars") is not None
                    and hit.get("content_returned_chars") is not None
                    and hit.get("content_truncated") is not None
                ):
                    bridge_hydrated_count += 1
                content = hit.get("content") or hit.get("snippet") or ""
                # Preserve mechanically attributable product or exact-read
                # facets for host-side evidence validation.
                metadata = _metadata_for_recall_hit(hit, dates)
                results.append({"content": content, "metadata": metadata})
        finally:
            vector_store.close()
            dag.close()
            store.close()

        provenance = dict(payload.get("provenance", {}))
        provenance["bridge_answer_ready"] = {
            "content_char_cap": self.answer_ready_content_chars,
            "exact_read_hydrated_count": bridge_hydrated_count,
        }
        return {
            "ok": True,
            "results": results[:limit],
            "provenance": provenance,
            "degraded": payload.get("degraded", False),
            "degraded_reason": payload.get("degraded_reason"),
        }

    def resolve_exact_refs(self, req: dict[str, Any]) -> dict[str, Any]:
        """Attach exact raw-message refs without adding or changing evidence text.

        Frozen answer-ready result files predate the bridge fields that expose a
        hydrated window's content offset.  Cached reasoning still needs exact
        refs for production ``lcm_compute`` grounding, so this read-only helper
        locates each already-returned content string in its cited ``store_id``.
        Ambiguous or missing matches fail closed and remain unannotated.
        """
        container_tag = str(req["containerTag"])
        raw_results = req.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError("resolve_exact_refs requires at most 50 result objects")

        from hermes_lcm.store import MessageStore

        db_path = self._db_path(container_tag)
        store = MessageStore(
            str(db_path),
            ingest_protection_config=self._config(db_path),
        )
        annotated: list[Any] = []
        resolved = 0
        unresolved = 0
        try:
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    annotated.append(raw_result)
                    unresolved += 1
                    continue
                result = dict(raw_result)
                metadata = dict(result.get("metadata") or {})
                content = str(result.get("content") or "")
                raw_store_id = metadata.get("store_id")
                if raw_store_id is None or not content:
                    annotated.append(result)
                    unresolved += 1
                    continue
                try:
                    store_id = int(raw_store_id)
                except (TypeError, ValueError, OverflowError):
                    annotated.append(result)
                    unresolved += 1
                    continue
                stored = store.get(store_id)
                source = str((stored or {}).get("content") or "")
                exact = _exact_ref_for_content(store_id, source, content)
                if exact is None:
                    annotated.append(result)
                    unresolved += 1
                    continue
                metadata.update(exact)
                result["metadata"] = metadata
                annotated.append(result)
                resolved += 1
        finally:
            store.close()
        return {
            "ok": True,
            "results": annotated,
            "provenance": {
                "mode": "deterministic_cached_content_match",
                "evidence_text_changed": False,
                "resolved": resolved,
                "unresolved": unresolved,
            },
        }

    def preanswer_evidence(self, req: dict[str, Any]) -> dict[str, Any]:
        """Invoke the product-owned V4.5 helper over cached baseline evidence.

        The bridge resolves only exact refs for evidence text already present in
        the frozen search result.  All planning, missing-requirement decisions,
        validation, delta retrieval, computation, and fallback behavior remain
        in ``hermes_lcm.preanswer_evidence``.
        """
        container_tag = str(req["containerTag"])
        question = str(req.get("question") or "")
        raw_results = req.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError("preanswer_evidence requires at most 50 result objects")

        import hermes_lcm.tools as lcm_tools
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.preanswer_evidence import build_preanswer_evidence
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        dates = self._load_dates(container_tag)
        candidates: list[dict[str, Any]] = []
        try:
            for raw_result in raw_results:
                if not isinstance(raw_result, dict):
                    continue
                content = str(raw_result.get("content") or "")
                metadata = raw_result.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                exact_ref = str(metadata.get("exact_ref") or "").strip()
                if not exact_ref and content and metadata.get("store_id") is not None:
                    try:
                        store_id = int(metadata["store_id"])
                    except (TypeError, ValueError, OverflowError):
                        store_id = 0
                    stored = store.get(store_id) if store_id > 0 else None
                    exact = _exact_ref_for_content(
                        store_id, str((stored or {}).get("content") or ""), content
                    )
                    exact_ref = str((exact or {}).get("exact_ref") or "")
                if exact_ref and content:
                    candidates.append({"exact_ref": exact_ref, "quote": content})

            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=dates,
                current_session_id=f"__hermes_lcm_preanswer_probe__{container_tag}",
            )

            def _retrieve(args: dict[str, Any]) -> str:
                self._ensure_embedder()
                cache_key = (
                    self.provider_name.strip().lower(),
                    str(self.embedder.model_id).strip(),
                )
                engine._lcm_embedding_provider_cache = (cache_key, self.embedder)
                return lcm_tools.lcm_recall(args, engine=engine)

            trace = build_preanswer_evidence(
                question,
                engine=engine,
                baseline_refs=candidates,
                question_date=_question_date(req.get("questionDate")),
                retrieve=_retrieve,
                enabled=True,
                context_engine_enabled=True,
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        return {
            "ok": True,
            "augmentation": trace.get("context"),
            "trace": trace,
            "provenance": {
                "implementation": "hermes_lcm.preanswer_evidence.build_preanswer_evidence",
                "product_owned": True,
                "baseline_search_bytes_changed": False,
            },
        }

    def selective_answer_evidence(self, req: dict[str, Any]) -> dict[str, Any]:
        """Invoke the V4.6.2 product session bundle over cached baseline bytes."""
        container_tag = str(req["containerTag"])
        question = str(req.get("question") or "")
        raw_results = req.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError(
                "selective_answer_evidence requires at most 50 result objects"
            )

        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.selective_recall import build_selective_session_bundle
        from hermes_lcm.store import MessageStore

        candidates, resolution = self._host_evidence_candidates(
            container_tag, raw_results
        )
        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        dates = self._load_dates(container_tag)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=dates,
                current_session_id=(
                    f"__hermes_lcm_selective_answer__{container_tag}"
                ),
            )
            trace = build_selective_session_bundle(
                question,
                engine=engine,
                baseline_refs=candidates,
                question_date=_question_date(req.get("questionDate")),
                enabled=True,
            )
        finally:
            dag.close()
            store.close()

        return {
            "ok": True,
            "augmentation": trace.get("context"),
            "trace": trace,
            "provenance": {
                "implementation": (
                    "hermes_lcm.selective_recall."
                    "build_selective_session_bundle"
                ),
                "product_owned": True,
                "baseline_search_bytes_changed": False,
                "exact_ref_resolution": resolution,
            },
        }

    def requirements_answer_evidence(self, req: dict[str, Any]) -> dict[str, Any]:
        """Invoke the V4.6.3 deterministic compiler over cached baseline bytes.

        The bridge resolves exact refs and supplies the existing product recall
        callback.  Contract parsing, slot closure, exact validation, finite
        coverage, computation, and the no-op decision remain in Hermes-LCM.
        """
        container_tag = str(req["containerTag"])
        question = str(req.get("question") or "")
        raw_results = req.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError(
                "requirements_answer_evidence requires at most 50 result objects"
            )

        import hermes_lcm.tools as lcm_tools
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.evidence_compiler import compile_preanswer_evidence
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        candidates, resolution = self._host_evidence_candidates(
            container_tag, raw_results
        )
        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=self._load_dates(container_tag),
                current_session_id=(
                    f"__hermes_lcm_requirements_compiler__{container_tag}"
                ),
            )

            def _retrieve(args: dict[str, Any]) -> str:
                self._ensure_embedder()
                cache_key = (
                    self.provider_name.strip().lower(),
                    str(self.embedder.model_id).strip(),
                )
                engine._lcm_embedding_provider_cache = (cache_key, self.embedder)
                return lcm_tools.lcm_recall(args, engine=engine)

            trace = compile_preanswer_evidence(
                question,
                engine=engine,
                baseline_refs=candidates,
                question_as_of=_question_date(req.get("questionDate")),
                retrieve=_retrieve,
                enabled=True,
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        return {
            "ok": True,
            "augmentation": trace.get("context"),
            "trace": trace,
            "provenance": {
                "implementation": (
                    "hermes_lcm.requirements_compiler."
                    "compile_preanswer_evidence"
                ),
                "product_owned": True,
                "provider_neutral_contract": True,
                "baseline_search_bytes_changed": False,
                "exact_ref_resolution": resolution,
            },
        }

    def _host_evidence_candidates(
        self, container_tag: str, raw_results: Any
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Resolve the cached baseline to bounded exact product references."""
        resolved = self.resolve_exact_refs(
            {"containerTag": container_tag, "results": raw_results}
        )
        candidates: list[dict[str, Any]] = []
        for raw_result in resolved.get("results", []):
            if not isinstance(raw_result, dict):
                continue
            content = str(raw_result.get("content") or "")
            metadata = raw_result.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            exact_ref = str(metadata.get("exact_ref") or "").strip()
            if exact_ref and content:
                candidate: dict[str, Any] = {"exact_ref": exact_ref, "quote": content}
                if metadata.get("date"):
                    candidate["date"] = metadata["date"]
                candidates.append(candidate)
        provenance = resolved.get("provenance")
        return candidates, provenance if isinstance(provenance, dict) else {}

    def selective_compiler_prepare(self, req: dict[str, Any]) -> dict[str, Any]:
        """Build the V4.6.2 code-owned minimal selector envelope."""
        container_tag = str(req["containerTag"])
        raw_results = req.get("results")
        session_evidence = req.get("sessionEvidence") or []
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError("selective_compiler_prepare requires at most 50 results")
        if not isinstance(session_evidence, list) or len(session_evidence) > 12:
            raise ValueError("selective_compiler_prepare requires bounded session evidence")

        from hermes_lcm.selective_compiler import prepare_selective_compiler

        baseline, resolution = self._host_evidence_candidates(container_tag, raw_results)
        candidates: list[dict[str, Any]] = list(baseline)
        seen = {str(item.get("exact_ref") or "") for item in candidates}
        for raw in session_evidence:
            if not isinstance(raw, dict):
                continue
            exact_ref = str(raw.get("exact_ref") or "").strip()
            quote = str(raw.get("quote") or "")
            if exact_ref and quote and exact_ref not in seen:
                item = {"exact_ref": exact_ref, "quote": quote}
                if raw.get("date"):
                    item["date"] = raw["date"]
                candidates.append(item)
                seen.add(exact_ref)

        prepared = prepare_selective_compiler(
            req.get("question"),
            baseline_refs=candidates,
            question_date=_question_date(req.get("questionDate")),
        )
        return {
            "ok": True,
            "status": prepared["status"],
            "reasonCode": prepared["reason_code"],
            "prompt": prepared.get("prompt"),
            "envelopeDigest": prepared.get("envelope_sha256"),
            "baselineDigest": resolution.get("resolved_exact_refs_sha256"),
            "baselineRefCount": resolution.get("resolved_exact_ref_count", 0),
            "selectorEvidenceDigest": prepared.get("envelope_sha256"),
            "selectorEvidenceRefCount": len(prepared.get("compiler_refs") or []),
            "compilerEvidence": prepared.get("compiler_refs") or [],
            "request": prepared.get("request"),
            "provenance": {
                **prepared.get("provenance", {}),
                "implementation": (
                    "hermes_lcm.selective_compiler.prepare_selective_compiler"
                ),
                "product_owned": True,
                "baseline_search_bytes_changed": False,
                "exact_ref_resolution": resolution,
            },
        }

    def selective_compiler_compile(self, req: dict[str, Any]) -> dict[str, Any]:
        """Validate the minimal proposal and run the exact product compiler."""
        container_tag = str(req["containerTag"])
        compiler_evidence = req.get("compilerEvidence")
        selector_proposal = req.get("selectorProposal")
        if not isinstance(compiler_evidence, list) or len(compiler_evidence) > 18:
            raise ValueError("selective_compiler_compile requires bounded evidence")
        if not isinstance(selector_proposal, dict):
            raise ValueError("selective_compiler_compile requires one proposal object")

        from hermes_lcm.selective_compiler import compile_selective_evidence
        from hermes_lcm.store import MessageStore

        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=self._load_dates(container_tag),
                current_session_id=f"__hermes_lcm_selective_compiler__{container_tag}",
            )
            trace = compile_selective_evidence(
                req.get("question"),
                engine=engine,
                compiler_refs=compiler_evidence,
                selector_proposal=selector_proposal,
                question_date=_question_date(req.get("questionDate")),
                enabled=True,
            )
        finally:
            store.close()
        return {
            "ok": True,
            "augmentation": trace.get("context"),
            "trace": trace,
            "provenance": {
                "implementation": (
                    "hermes_lcm.selective_compiler.compile_selective_evidence"
                ),
                "product_owned": True,
                "baseline_search_bytes_changed": False,
            },
        }

    def host_evidence_prepare(self, req: dict[str, Any]) -> dict[str, Any]:
        """Ask product code to construct the immutable selector envelope."""
        container_tag = str(req["containerTag"])
        question = str(req.get("question") or "")
        raw_results = req.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError("host_evidence_prepare requires at most 50 result objects")

        import hermes_lcm.tools as lcm_tools
        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.host_evidence import prepare_host_evidence_selector
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        candidates, resolution = self._host_evidence_candidates(container_tag, raw_results)
        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        dates = self._load_dates(container_tag)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=dates,
                current_session_id=f"__hermes_lcm_host_evidence__{container_tag}",
            )

            def _retrieve(args: dict[str, Any]) -> str:
                self._ensure_embedder()
                cache_key = (
                    self.provider_name.strip().lower(),
                    str(self.embedder.model_id).strip(),
                )
                engine._lcm_embedding_provider_cache = (cache_key, self.embedder)
                return lcm_tools.lcm_recall(args, engine=engine)

            prepared = prepare_host_evidence_selector(
                question,
                baseline_refs=candidates,
                question_date=_question_date(req.get("questionDate")),
                retrieve=_retrieve,
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()
        return {
            "ok": True,
            "prompt": prepared["prompt"],
            "envelopeDigest": prepared["envelope_sha256"],
            "baselineDigest": prepared["baseline_exact_refs_sha256"],
            "baselineRefCount": prepared["baseline_exact_ref_count"],
            "selectorEvidenceDigest": prepared["selector_exact_refs_sha256"],
            "selectorEvidenceRefCount": prepared["selector_exact_ref_count"],
            "compilerEvidence": prepared["compiler_refs"],
            "preparedRetrieval": prepared["retrieval"],
            "request": prepared["request"],
            "budgets": prepared["budgets"],
            "provenance": {
                **prepared.get("provenance", {}),
                "implementation": "hermes_lcm.host_evidence.prepare_host_evidence_selector",
                "product_owned": True,
                "baseline_search_bytes_changed": False,
                "exact_ref_resolution": resolution,
            },
        }

    def host_evidence_compile(self, req: dict[str, Any]) -> dict[str, Any]:
        """Validate one semantic proposal and compile product-owned evidence."""
        container_tag = str(req["containerTag"])
        question = str(req.get("question") or "")
        raw_results = req.get("results")
        selector_proposal = req.get("selectorProposal")
        compiler_evidence = req.get("compilerEvidence")
        prepared_retrieval = req.get("preparedRetrieval")
        if not isinstance(raw_results, list) or len(raw_results) > 50:
            raise ValueError("host_evidence_compile requires at most 50 result objects")
        if not isinstance(selector_proposal, dict):
            raise ValueError("host_evidence_compile requires a selector proposal object")
        if not isinstance(compiler_evidence, list) or len(compiler_evidence) > 50:
            raise ValueError("host_evidence_compile requires bounded compiler evidence")
        if not isinstance(prepared_retrieval, dict):
            raise ValueError("host_evidence_compile requires prepared retrieval provenance")

        from hermes_lcm.dag import SummaryDAG
        from hermes_lcm.host_evidence import build_host_supplied_evidence
        from hermes_lcm.store import MessageStore
        from hermes_lcm.vector_store import VectorStore

        baseline_candidates, resolution = self._host_evidence_candidates(
            container_tag, raw_results
        )
        candidates: list[dict[str, str]] = []
        seen_refs: set[str] = set()
        for item in [*baseline_candidates, *compiler_evidence]:
            if not isinstance(item, dict):
                continue
            exact_ref = str(item.get("exact_ref") or "").strip()
            quote = str(item.get("quote") or "")
            if exact_ref and quote and exact_ref not in seen_refs:
                candidates.append({"exact_ref": exact_ref, "quote": quote})
                seen_refs.add(exact_ref)
        db_path = self._db_path(container_tag)
        config = self._config(db_path)
        store = MessageStore(str(db_path), ingest_protection_config=config)
        dag = SummaryDAG(str(db_path))
        vector_store = VectorStore(str(db_path), config=config)
        dates = self._load_dates(container_tag)
        try:
            engine = SimpleNamespace(
                _config=config,
                _store=store,
                _dag=dag,
                _assertions=None,
                _hermes_home=str(self.workdir),
                _session_occurrence_dates=dates,
                current_session_id=f"__hermes_lcm_host_evidence__{container_tag}",
            )

            trace = build_host_supplied_evidence(
                question,
                engine=engine,
                baseline_refs=candidates,
                question_date=_question_date(req.get("questionDate")),
                selector=lambda _request: selector_proposal,
                retrieve=None,
                enabled=True,
                budgets={"max_retrieval_calls": 0},
                prepared_retrieval=prepared_retrieval,
            )
        finally:
            vector_store.close()
            dag.close()
            store.close()

        return {
            "ok": True,
            "augmentation": trace.get("context"),
            "trace": trace,
            "provenance": {
                "implementation": "hermes_lcm.host_evidence.build_host_supplied_evidence",
                "product_owned": True,
                "baseline_search_bytes_changed": False,
                "exact_ref_resolution": resolution,
            },
        }

    def verify_computation_answer(self, req: dict[str, Any]) -> dict[str, Any]:
        """Run the product's pure immutable-trace verifier after final wording."""
        raw_trace = req.get("trace")
        if not isinstance(raw_trace, dict):
            raise ValueError("verify_computation_answer requires a trace object")
        from hermes_lcm.reasoning import ComputationTrace, verify_final_answer

        result_value = raw_trace.get("result_value")
        if isinstance(result_value, list):
            result_value = tuple(str(value) for value in result_value)
        trace = ComputationTrace(
            operation=str(raw_trace.get("operation") or ""),
            result=str(raw_trace.get("result") or ""),
            result_value=result_value,
            unit=(
                str(raw_trace["unit"]) if raw_trace.get("unit") is not None else None
            ),
            citations=tuple(str(value) for value in raw_trace.get("citations") or ()),
            entities=tuple(str(value) for value in raw_trace.get("entities") or ()),
            evidence_dates=tuple(
                str(value) for value in raw_trace.get("evidence_dates") or ()
            ),
            steps=tuple(str(value) for value in raw_trace.get("steps") or ()),
            answer=str(raw_trace.get("answer") or ""),
        )
        decision = verify_final_answer(req.get("candidate"), trace)
        return {
            "ok": True,
            "status": decision.status,
            "reason": decision.reason,
            "canonicalAnswer": trace.answer,
            "provenance": {
                "transport": "deterministic_local",
                "implementation": "verify_final_answer",
                "provider": "none",
                "model": "none",
            },
        }

    # -- clear ----------------------------------------------------------------

    def clear(self, req: dict[str, Any]) -> dict[str, Any]:
        container_tag = str(req["containerTag"])
        db_path = self._db_path(container_tag)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(db_path) + suffix)
            if candidate.exists():
                candidate.unlink()
        dates_path = self._dates_path(container_tag)
        if dates_path.exists():
            dates_path.unlink()
        self._order.pop(container_tag, None)
        return {"ok": True}

    # -- dispatch -------------------------------------------------------------

    def handle(self, req: dict[str, Any]) -> dict[str, Any]:
        cmd = req.get("cmd")
        if cmd == "initialize":
            return self.initialize(req)
        if cmd == "ingest":
            return self.ingest(req)
        if cmd == "search":
            return self.search(req)
        if cmd == "resolve_exact_refs":
            return self.resolve_exact_refs(req)
        if cmd == "preanswer_evidence":
            return self.preanswer_evidence(req)
        if cmd == "selective_answer_evidence":
            return self.selective_answer_evidence(req)
        if cmd == "requirements_answer_evidence":
            return self.requirements_answer_evidence(req)
        if cmd == "selective_compiler_prepare":
            return self.selective_compiler_prepare(req)
        if cmd == "selective_compiler_compile":
            return self.selective_compiler_compile(req)
        if cmd == "host_evidence_prepare":
            return self.host_evidence_prepare(req)
        if cmd == "host_evidence_compile":
            return self.host_evidence_compile(req)
        if cmd == "verify_computation_answer":
            return self.verify_computation_answer(req)
        if cmd == "clear":
            return self.clear(req)
        if cmd == "ping":
            return {"ok": True}
        raise RuntimeError(f"unknown cmd: {cmd!r}")


def main() -> int:
    bridge = Bridge()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                json.dumps({"ok": False, "error": f"bad json: {exc}"}),
                file=_RESPONSE_OUT,
                flush=True,
            )
            continue
        try:
            response = bridge.handle(req)
        except Exception as exc:  # noqa: BLE001 - report every failure loudly
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), file=_RESPONSE_OUT, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
