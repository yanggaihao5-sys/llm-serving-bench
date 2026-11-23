from __future__ import annotations

import json
import time
from pathlib import Path

REQUIRED = ("run_id", "target", "kind", "created_at", "status", "model", "engine", "load")


def validate(meta: dict) -> list[str]:
    errors = [f"missing {k}" for k in REQUIRED if k not in meta]
    if meta.get("kind") != "llm-serving":
        errors.append("kind must be llm-serving")
    if meta.get("status") not in ("running", "finished", "failed"):
        errors.append("bad status")
    if not isinstance(meta.get("load"), dict):
        errors.append("load must be object")
    return errors


def create_run(cfg: dict, run_id: str, model: str, engine: dict, load: dict) -> Path:
    root = Path(cfg["runs_root"]) / run_id
    if root.exists():
        raise FileExistsError(f"run exists: {root}")
    root.mkdir(parents=True)
    meta = {
        "run_id": run_id, "target": cfg["target"], "kind": "llm-serving",
        "created_at": time.time(), "status": "running",
        "model": model, "engine": engine, "load": load,
    }
    (root / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return root


def finish_run(root: Path, status: str = "finished") -> None:
    p = Path(root) / "run.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["status"] = status
    meta["finished_at"] = time.time()
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
