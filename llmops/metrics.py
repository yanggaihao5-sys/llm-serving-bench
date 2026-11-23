from __future__ import annotations

import json
from pathlib import Path


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def parse_nvidia_smi(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        try:
            rows.append({
                "index": int(f[0]),
                "mem_used_mb": float(f[1].replace("MiB", "").strip()),
                "mem_total_mb": float(f[2].replace("MiB", "").strip()),
                "util_pct": float(f[3].replace("%", "").strip()),
            })
        except ValueError:
            continue
    return rows


def records(prom: dict[str, float], gpus: list[dict], ts: float) -> list[dict]:
    out = []
    for name, value in prom.items():
        out.append({"ts": ts, "step": 0, "name": "serve/" + name, "value": value})
    for g in gpus:
        i = g["index"]
        out.append({"ts": ts, "step": 0, "name": f"gpu/{i}/mem_used_mb",
                    "value": g["mem_used_mb"]})
        out.append({"ts": ts, "step": 0, "name": f"gpu/{i}/util_pct",
                    "value": g["util_pct"]})
    return out


def append_jsonl(path: str | Path, recs: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
