from __future__ import annotations

import json
from pathlib import Path

REQUIRED = ("target", "engine", "base_url", "model", "runs_root", "slo")


def load(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"config missing: {missing}")
    target = cfg["target"]
    runs = str(cfg["runs_root"]).replace("\\", "/")
    if runs.startswith("/") or ":" in runs.split("/")[0]:
        raise ValueError(f"runs_root must be a repo-relative path, got {runs}")
    for key in ("ttft_p95_ms", "tpot_p95_ms"):
        if key not in cfg["slo"]:
            raise ValueError(f"slo missing: {key}")
    return cfg


def runs_root(cfg: dict) -> Path:
    return Path(cfg["runs_root"])
