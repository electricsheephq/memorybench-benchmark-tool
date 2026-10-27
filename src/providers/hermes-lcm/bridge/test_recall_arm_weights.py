from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

BRIDGE_PATH = Path(__file__).with_name("hermes_lcm_bridge.py")
SPEC = importlib.util.spec_from_file_location("recall_weights_bridge_under_test", BRIDGE_PATH)
bridge = importlib.util.module_from_spec(SPEC)
_saved_stdout = sys.stdout
SPEC.loader.exec_module(bridge)
sys.stdout = _saved_stdout

ENV = "HERMES_MB_RECALL_ARM_WEIGHTS"


@pytest.fixture
def make_bridge(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_MB_WORKDIR", str(tmp_path))
    monkeypatch.setenv("HERMES_MB_PROVIDER", "stub")
    monkeypatch.setenv("HERMES_MB_MODEL", "stub-hash-64")
    monkeypatch.setenv("HERMES_MB_EMBEDDINGS", "off")
    monkeypatch.delenv("HERMES_MB_FUSION", raising=False)

    def make(raw):
        if raw is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, raw)
        return bridge.Bridge()

    return make


@pytest.mark.parametrize("raw", [None, ""])
def test_default_config_is_byte_identical_and_constructor_unchanged(make_bridge, monkeypatch, raw):
    instance = make_bridge(raw)
    import hermes_lcm.config as product

    db_path = instance._db_path("default")
    kwargs = {
        "database_path": str(db_path),
        "embeddings_enabled": False,
        "embedding_provider": "stub",
        "embedding_model": "stub-hash-64",
    }
    baseline = product.LCMConfig(**kwargs)
    assignments = []
    original_setattr = product.LCMConfig.__setattr__

    def tracked_setattr(self, name, value):
        if name == "recall_arm_weights":
            assignments.append(value)
        original_setattr(self, name, value)

    monkeypatch.setattr(product.LCMConfig, "__setattr__", tracked_setattr)
    constructor = Mock(wraps=product.LCMConfig)
    monkeypatch.setattr(product, "LCMConfig", constructor)
    config = instance._config(db_path)

    constructor.assert_called_once_with(**kwargs)
    assert assignments == [product._DEFAULT_RECALL_ARM_WEIGHTS]  # constructor only
    assert config.recall_arm_weights == product._DEFAULT_RECALL_ARM_WEIGHTS
    assert json.dumps(vars(config), sort_keys=True).encode() == json.dumps(
        vars(baseline), sort_keys=True
    ).encode()


@pytest.mark.parametrize(("raw", "expected"), [
    ("fts=1.0,summary=1.0,chunk=1.0", {"fts": 1.0, "summary": 1.0, "chunk": 1.0}),
    ("chunk=1.0,fts=0.75,summary=1.0", {"fts": 0.75, "summary": 1.0, "chunk": 1.0}),
    ("fts=0,summary=1e0,chunk=.5", {"fts": 0.0, "summary": 1.0, "chunk": 0.5}),
])
def test_requested_weights_reach_config(make_bridge, raw, expected):
    instance = make_bridge(raw)
    assert instance._config(instance._db_path("weighted")).recall_arm_weights == expected


@pytest.mark.parametrize("raw", [
    "fts=1,summary=1",
    "fts=1,summary=1,unknown=1",
    "fts=1,summary=1,chunk=oops",
    "fts=-1,summary=1,chunk=1",
    "fts=1,summary=1,chunk=1junk",
    "fts=1,summary=1,chunk=1,",
    "fts=1,summary=1,chunk=1,unknown=1",
    "fts=1,summary=1,chunk=1,fts=0.5",
    "fts=1,fts=1,chunk=1",
    "FTS=1,summary=1,chunk=1",
    "fts=1, summary=1,chunk=1",
    " ",
    "fts=nan,summary=1,chunk=1",
    "fts=inf,summary=1,chunk=1",
    "fts=1e999,summary=1,chunk=1",
])
def test_invalid_flag_fails_closed(make_bridge, raw):
    with pytest.raises(RuntimeError, match=ENV):
        instance = make_bridge(raw)
        instance._config(instance._db_path("invalid"))


@pytest.mark.parametrize("failure", ["warning", "missing", "changed"])
def test_product_parser_is_used_and_cannot_silently_fallback(make_bridge, monkeypatch, failure):
    import hermes_lcm.config as product

    parser = product._parse_arm_weights
    calls = []

    def parse(raw, defaults, *, warnings):
        calls.append((raw, defaults))
        result = parser(raw, defaults, warnings=warnings)
        if failure == "warning":
            warnings.append("ignored item")
        elif failure == "missing":
            result.pop("chunk")
        else:
            result["fts"] = 0.5
        return result

    monkeypatch.setattr(product, "_parse_arm_weights", parse)
    with pytest.raises(RuntimeError, match=ENV):
        instance = make_bridge("fts=1,summary=1,chunk=1")
        instance._config(instance._db_path("fallback"))
    assert calls == [("fts=1,summary=1,chunk=1", product._DEFAULT_RECALL_ARM_WEIGHTS)]


def test_flag_is_read_once_and_logged_at_initialize(make_bridge, monkeypatch, capsys):
    instance = make_bridge("fts=1,summary=1,chunk=1")
    monkeypatch.setenv(ENV, "fts=0.5,summary=1,chunk=1")
    instance.initialize({})
    expected = {"fts": 1.0, "summary": 1.0, "chunk": 1.0}
    assert instance._config(instance._db_path("first")).recall_arm_weights == expected
    assert instance._config(instance._db_path("second")).recall_arm_weights == expected
    log = capsys.readouterr().err
    assert log.count("recall_arm_weights=") == 1
    assert 'recall_arm_weights={"fts": 1.0, "summary": 1.0, "chunk": 1.0}' in log
