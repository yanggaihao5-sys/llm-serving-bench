"""Frozen benchmark protocol: constants, prompt-set loading and hash verification.

Freezing rules
--------------
1. The prompt set is a file on disk. At run time it is read-only, used in file
   order, and verified by sha256; it must never be generated per round,
   reordered, or edited.
2. Dispatch mode is a protocol constant (continuous / barrier); concurrency and
   batch boundaries are fixed by the protocol.
3. The number of warm-up requests is fixed.
4. Defaults: 32 concurrent clients / 64 output tokens / 60 s windows /
   temperature 0 / a single endpoint.

Every deployment-specific value (endpoint, model, output root) is read from
CLI flags or environment variables; this module contains no endpoint or model
name of any specific deployment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "bench"
PROMPT_SET = BENCH_DIR / "prompts-frozen-v1.jsonl"
PROMPT_META = BENCH_DIR / "prompts-frozen-v1.meta.json"

# A local placeholder base URL; never a real deployment. CI uses --fake only.
# Override with $LLM_BENCH_BASE_URL or --endpoint.
_DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def default_base_url() -> str:
    return os.environ.get("LLM_BENCH_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")


def default_endpoint() -> str:
    """Chat-completions endpoint from $LLM_BENCH_BASE_URL if set, else local."""
    base = default_base_url()
    if base.endswith("/v1/chat/completions"):
        return base
    return base + "/v1/chat/completions"


def default_model() -> str:
    return os.environ.get("LLM_BENCH_MODEL", "")


# Every number in a run is derived from these constants; scripts must not carry
# their own copy.
PROTOCOL = {
    "protocol_id": "FROZEN-v1",
    "concurrency": 32,
    "max_tokens": 64,
    "window_s": 60.0,
    "temperature": 0,
    "stream": True,
    "include_usage": True,
    "repeats": 5,
    "warmup_requests": 8,
    # Steady-state handling (learned from 18 historical runs; see METHODOLOGY):
    # warm-up runs once before the first window only; per-window steady skip is
    # kept as a diagnostic switch, default off, because offline recomputation
    # showed window-start trimming did not fix between-window spread.
    "protocol_rev": "v3-steady",
    "warmup_scope": "first",   # "each" reproduces the legacy behaviour
    "steady_skip_s": 0.0,
    # Stability rule: R=(max-min)/median is reference only — a single outlier
    # can fail the whole run; MAD/median is the primary gate.
    "stability_rule_v3": {
        "primary": "MAD_pct",
        "threshold_pct": 10.0,
        "outlier_k": 3.0,
        "note": "R=(max-min)/median is reference only; outliers need a cause",
    },
    "prompt_set": "bench/prompts-frozen-v1.jsonl",
    "modes": {
        "continuous": {
            "kind": "continuous",
            "workers": 32,
            "dispatch": "a worker sends the next request as soon as it is free "
                        "(ASAP, no sleep)",
            "inter_request_delay_s": 0.0,
            "batch_boundary": None,
        },
        "barrier": {
            "kind": "barrier",
            "batch_size": 32,
            "dispatch": "32 requests fired together; next batch only after all "
                        "return",
            "inter_request_delay_s": 0.0,
            "inter_batch_delay_s": 0.0,
            "batch_boundary": "batch = 32 concurrent; edge = last return",
        },
        "openloop": {
            "kind": "openloop",
            "dispatch": "fixed schedule (one request per interval), independent "
                        "of prior completions",
            "inter_request_delay_s": None,
            "batch_boundary": None,
            "note": "Open loop decouples arrival from completion. Under closed "
                    "loop (continuous), short requests finish first and their "
                    "workers re-issue sooner, so the short-input share amplifies "
                    "itself and two input mixes stop being comparable. Under "
                    "open loop the arrival sequence is fixed by k*interval.",
        },
    },
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_prompt_set(path: str | Path = PROMPT_SET,
                    meta_path: str | Path = PROMPT_META) -> dict:
    """Read the frozen prompt set; refuse to run on sha256/meta mismatch."""
    path = Path(path)
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    meta: dict = {}
    meta_path = Path(meta_path)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expected = meta.get("sha256")
        if expected and expected != digest:
            raise ValueError(
                f"prompt set hash drift: meta={expected[:12]} file={digest[:12]}")
    rows = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not row.get("id") or not row.get("text"):
            raise ValueError(f"bad prompt row: {row!r}")
        rows.append({"id": row["id"], "text": row["text"]})
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate prompt id")
    # Callers may pass a relative path (--prompt-set) while ROOT is absolute.
    # If the corpus lives outside the repo (external data), record as-is.
    try:
        rel = str(path.resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "sha256": digest,
        "count": len(rows),
        "prompts": rows,
        "meta": meta,
    }


def protocol_snapshot() -> dict:
    """Protocol snapshot for run.json / report metadata."""
    snap = json.loads(json.dumps(PROTOCOL))
    snap["prompt_set"] = load_prompt_set()
    snap["prompt_set"].pop("prompts", None)
    return snap


def prompt_cycle(prompts: list[dict], n: int, start: int = 0) -> list[dict]:
    """Take n prompts cycling in file order (no reordering)."""
    if not prompts:
        raise ValueError("empty prompt set")
    return [prompts[(start + i) % len(prompts)] for i in range(n)]
