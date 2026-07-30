from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


BRIDGE_PATH = Path(__file__).with_name("hermes_lcm_bridge.py")
SPEC = importlib.util.spec_from_file_location("hermes_lcm_bridge_under_test", BRIDGE_PATH)
assert SPEC is not None and SPEC.loader is not None
bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bridge
SPEC.loader.exec_module(bridge)


def hit(store_id: int, *, node_id: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"store_id": store_id}
    if node_id is not None:
        result["node_id"] = node_id
    return result


def stores(selected: list[dict[str, object]]) -> list[int]:
    return [int(entry["hit"]["store_id"]) for entry in selected]


def test_dedup_arm_keeps_first_node_or_message_hit() -> None:
    assert bridge.dedup_arm(
        [hit(1), hit(1), hit(2, node_id=9), hit(3, node_id=9), hit(4)]
    ) == [hit(1), hit(2, node_id=9), hit(4)]


def test_quota_fixture_floor_round_robin_cross_arm_dedup_and_limit() -> None:
    # Replay by hand: floor f1; round 1 pulls f2 then c2 (duplicate), c3, and
    # c5 to satisfy the two successful chunk pulls; round 2 pulls f4.
    fts = [hit(1), hit(1), hit(2), hit(4)]
    chunk = [hit(2), hit(3), hit(5), hit(6)]
    selected = bridge.quota_merge(
        fts,
        chunk,
        limit=5,
        q_fts=1,
        q_chunk=2,
        floor_fts=1,
    )
    assert stores(selected) == [1, 2, 3, 5, 4]
    assert [(entry["arm"], entry["arm_rank"]) for entry in selected] == [
        ("fts", 1),
        ("fts", 2),
        ("chunk", 2),
        ("chunk", 3),
        ("fts", 3),
    ]


def test_quota_fixture_round_robin_limit_cut() -> None:
    # Replay by hand: f1, c4/c5, f2, c6; f3 is cut by limit=5.
    selected = bridge.quota_merge(
        [hit(1), hit(2), hit(3)],
        [hit(4), hit(5), hit(6)],
        limit=5,
        q_fts=1,
        q_chunk=2,
        floor_fts=0,
    )
    assert stores(selected) == [1, 4, 5, 2, 6]


def test_quota_fixture_arm_exhaustion_backfills_from_remaining_arm() -> None:
    # Replay by hand: f1, c2/c3, then c4 after the empty FTS arm exhausts.
    selected = bridge.quota_merge(
        [hit(1)],
        [hit(2), hit(3), hit(4)],
        limit=4,
        q_fts=1,
        q_chunk=2,
        floor_fts=0,
    )
    assert stores(selected) == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "raw",
    [
        "rrf",
        "quota:fts=1,chunk=2,floor=-1",
        "quota:fts=1,chunk=2, floor=0",
        "quota:fts=1,chunk=2,extra=0",
        " quota:fts=1,chunk=2",
    ],
)
def test_fusion_env_rejects_every_undeclared_value(raw: str) -> None:
    with pytest.raises(RuntimeError, match="HERMES_MB_FUSION"):
        bridge._parse_fusion_mode(raw)


def test_fusion_env_accepts_declared_grammar_and_empty_default() -> None:
    assert bridge._parse_fusion_mode("") is None
    assert bridge._parse_fusion_mode("quota:fts=1,chunk=2") == (1, 2, 0)
    assert bridge._parse_fusion_mode("quota:fts=0,chunk=2,floor=3") == (0, 2, 3)


def test_collect_quota_arms_matches_replay_candidate_queries() -> None:
    calls: list[tuple[str, object]] = []
    tools = types.SimpleNamespace()

    def fts_arm(
        engine: object, query: str, *, candidate_limit: int, deadline: float
    ) -> tuple[list[dict[str, object]], None]:
        calls.append(("fts", (engine, query, candidate_limit, deadline)))
        return [{"store_id": 1}], None

    def embed_query(provider: object, query: str, *, remaining_s: float) -> list[float]:
        calls.append(("embed", (provider, query, remaining_s)))
        return [0.25, 0.5]

    def chunk_arm(
        engine: object,
        *,
        query_vector: list[float],
        provider: object,
        candidate_limit: int,
        deadline: float,
    ) -> tuple[list[dict[str, object]], str]:
        calls.append(("chunk", (engine, query_vector, provider, candidate_limit, deadline)))
        return [{"store_id": 2}], "full"

    tools._lcm_recall_fts_arm = fts_arm
    tools._lcm_grep_embed_query = embed_query
    tools._lcm_recall_chunk_arm = chunk_arm
    provider = object()
    arms = bridge._collect_quota_arms(
        "engine",
        "same query",
        provider,
        lcm_tools=tools,
        deadline=10_000.0,
    )

    assert arms == {"fts": [{"store_id": 1}], "chunk": [{"store_id": 2}]}
    assert calls[0][0] == "fts"
    assert calls[0][1][1:3] == ("same query", 200)  # type: ignore[index]
    assert calls[1][0] == "embed"
    assert calls[1][1][1] == "same query"  # type: ignore[index]
    assert calls[2][0] == "chunk"
    assert calls[2][1][3] == 200  # type: ignore[index]


class _FakeStore:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def get(self, _store_id: int) -> None:
        return None

    def close(self) -> None:
        pass


class _FakeDag:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeVectorStore:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeEmbedder:
    model_id = "fixture-model"


def install_fake_hermes_modules(monkeypatch: pytest.MonkeyPatch, tools: types.ModuleType) -> None:
    package = types.ModuleType("hermes_lcm")
    package.__path__ = []  # type: ignore[attr-defined]
    dag = types.ModuleType("hermes_lcm.dag")
    dag.SummaryDAG = _FakeDag
    store = types.ModuleType("hermes_lcm.store")
    store.MessageStore = _FakeStore
    vector_store = types.ModuleType("hermes_lcm.vector_store")
    vector_store.VectorStore = _FakeVectorStore
    monkeypatch.setitem(sys.modules, "hermes_lcm", package)
    monkeypatch.setitem(sys.modules, "hermes_lcm.dag", dag)
    monkeypatch.setitem(sys.modules, "hermes_lcm.store", store)
    monkeypatch.setitem(sys.modules, "hermes_lcm.vector_store", vector_store)
    monkeypatch.setitem(sys.modules, "hermes_lcm.tools", tools)


def fixture_bridge(monkeypatch: pytest.MonkeyPatch) -> bridge.Bridge:
    instance = bridge.Bridge.__new__(bridge.Bridge)
    instance.embedder = _FakeEmbedder()
    instance.provider_name = "fastembed"
    instance.workdir = Path("/fixture-workdir")
    instance.answer_ready_content_chars = 2400
    instance._db_path = lambda _tag: Path("/fixture-workdir/container.db")
    instance._config = lambda _path: object()
    instance._load_dates = lambda _tag: {"session-1": "2023-05-30"}
    return instance


def answer_ready_hit(store_id: int, arm: str) -> dict[str, object]:
    content = f"{arm}-{store_id}"
    return {
        "kind": "message_excerpt",
        "store_id": store_id,
        "session_id": "session-1",
        "content": content,
        "content_chars": len(content),
        "content_returned_chars": len(content),
        "content_truncated": False,
    }


def test_search_default_keeps_lcm_recall_and_skips_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    tools = types.ModuleType("hermes_lcm.tools")

    def lcm_recall(args: dict[str, object], *, engine: object) -> str:
        calls.append(f"lcm_recall:{args['query']}")
        return json.dumps(
            {
                "hits": [],
                "provenance": {"source": "fixture-default"},
                "degraded": False,
                "degraded_reason": None,
            }
        )

    tools.lcm_recall = lcm_recall
    install_fake_hermes_modules(monkeypatch, tools)
    monkeypatch.delenv("HERMES_MB_FUSION", raising=False)
    instance = fixture_bridge(monkeypatch)
    monkeypatch.setattr(
        bridge,
        "_collect_quota_arms",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("quota arm collection must not run on the default path")
        ),
    )

    response = instance.search({"containerTag": "fixture", "query": "ordinary", "limit": 3})

    assert calls == ["lcm_recall:ordinary"]
    assert response == {
        "ok": True,
        "results": [],
        "provenance": {
            "source": "fixture-default",
            "bridge_answer_ready": {
                "content_char_cap": 2400,
                "exact_read_hydrated_count": 0,
            },
        },
        "degraded": False,
        "degraded_reason": None,
    }


def test_search_quota_routes_canned_arms_and_emits_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = types.ModuleType("hermes_lcm.tools")

    def unexpected_lcm_recall(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("quota mode must not call lcm_recall")

    tools.lcm_recall = unexpected_lcm_recall
    install_fake_hermes_modules(monkeypatch, tools)
    monkeypatch.setenv("HERMES_MB_FUSION", "quota:fts=1,chunk=2")
    instance = fixture_bridge(monkeypatch)
    calls: list[tuple[str, int]] = []

    def collect(*_args: object, **_kwargs: object) -> dict[str, list[dict[str, object]]]:
        calls.append(("collect", 200))
        return {
            "fts": [answer_ready_hit(1, "fts"), answer_ready_hit(2, "fts")],
            "chunk": [
                answer_ready_hit(3, "chunk"),
                answer_ready_hit(2, "chunk"),
                answer_ready_hit(4, "chunk"),
            ],
        }

    monkeypatch.setattr(bridge, "_collect_quota_arms", collect)

    response = instance.search({"containerTag": "fixture", "query": "quota", "limit": 3})

    assert calls == [("collect", 200)]
    assert response["fusion_mode"] == "quota:fts=1,chunk=2,floor=0"
    assert response["degraded"] is False
    assert response["degraded_reason"] is None
    assert [row["metadata"]["store_id"] for row in response["results"]] == [1, 3, 2]
    assert [row["metadata"]["arms"] for row in response["results"]] == [
        "fts",
        "chunk",
        "chunk",
    ]
    assert [row["metadata"]["arm_rank"] for row in response["results"]] == [1, 1, 2]


def test_search_malformed_fusion_fails_before_opening_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HERMES_MB_FUSION", "quota:fts=1,chunk=oops")
    instance = fixture_bridge(monkeypatch)
    opened = False

    def unexpected_config(_path: Path) -> object:
        nonlocal opened
        opened = True
        raise AssertionError("malformed fusion must fail before store/config setup")

    instance._config = unexpected_config
    with pytest.raises(RuntimeError, match="HERMES_MB_FUSION"):
        instance.search({"containerTag": "fixture", "query": "bad", "limit": 3})
    assert opened is False
