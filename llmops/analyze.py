from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bench import summarize


def analyze(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    bench = json.loads((root / "bench.json").read_text(encoding="utf-8"))
    summary = bench.get("summary") or summarize(bench.get("samples", []),
                                                bench.get("wall_s", 0.0))
    n = max(1, summary.get("n", 1))
    summary["goodput_req_s"] = round(
        sum(1 for s in bench.get("samples", []) if s.get("ok")
            and s.get("ttft_s", 99) * 1000 <= 2000
            and s.get("tpot_s", 99) * 1000 <= 50) / max(summary.get("wall_s", 1), 1e-9), 3)
    return summary


def compare(a: dict, b: dict, key: str) -> float:
    av, bv = float(a.get(key, 0)), float(b.get(key, 0))
    return round((bv - av) / av * 100, 2) if av else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    print(json.dumps(analyze(args.run), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
