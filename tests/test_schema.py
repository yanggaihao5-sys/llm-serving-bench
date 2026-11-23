import json
from pathlib import Path

import pytest

from llmops.schema import create_run, validate


def test_validate_ok():
    meta = {"run_id": "x", "target": "t", "kind": "llm-serving", "created_at": 1,
            "status": "running", "model": {}, "engine": {}, "load": {}}
    assert validate(meta) == []


def test_validate_bad_kind():
    meta = {"run_id": "x", "target": "t", "kind": "rl", "created_at": 1,
            "status": "running", "model": {}, "engine": {}, "load": {}}
    assert any("kind" in e for e in validate(meta))


def test_create_run_isolated(tmp_path):
    cfg = {"target": "t", "runs_root": str(tmp_path / "targets" / "t" / "runs")}
    root = create_run(cfg, "r1", {"name": "m"}, {"name": "e"}, {"concurrency": 1})
    meta = json.loads((root / "run.json").read_text(encoding="utf-8"))
    assert meta["run_id"] == "r1" and meta["target"] == "t"
    with pytest.raises(FileExistsError):
        create_run(cfg, "r1", {}, {}, {})
