from __future__ import annotations

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    def q(p):
        i = min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))
        return float(s[i])
    return {"p50": q(0.50), "p95": q(0.95), "p99": q(0.99)}


def summarize(samples: list[dict], wall_s: float) -> dict:
    ok = [s for s in samples if s.get("ok")]
    ttfts = [s["ttft_s"] * 1000 for s in ok]
    tpots = [s["tpot_s"] * 1000 for s in ok if s.get("tpot_s")]
    toks = sum(int(s.get("completion_tokens") or 0) for s in ok)
    return {
        "n": len(samples), "ok": len(ok),
        "error_rate": round(1 - len(ok) / len(samples), 4) if samples else 0.0,
        "wall_s": round(wall_s, 3),
        "throughput_tok_s": round(toks / wall_s, 2) if wall_s else 0.0,
        "req_s": round(len(ok) / wall_s, 2) if wall_s else 0.0,
        "ttft_ms": percentiles(ttfts),
        "tpot_ms": percentiles(tpots),
        "completion_tokens": toks,
    }


def _one(base_url: str, api_key: str, model: str, prompt: str, max_tokens: int,
         timeout: int = 300) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t0 = time.perf_counter()
    first = None
    tokens = 0
    usage = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.perf_counter() - t0
            if delta.get("content"):
                tokens += 1
        total = time.perf_counter() - t0
        if usage and usage.get("completion_tokens"):
            tokens = int(usage["completion_tokens"])
        tpot = (total - (first or total)) / tokens if tokens else 0.0
        return {"ok": True, "ttft_s": first or total, "tpot_s": tpot,
                "completion_tokens": tokens, "latency_s": total}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc), "latency_s": time.perf_counter() - t0}


def run_bench(cfg: dict, prompts: list[str], concurrency: int, max_tokens: int,
              api_key: str = "") -> tuple[list[dict], float]:
    def job(p):
        return _one(cfg["base_url"], api_key, cfg["model"], p, max_tokens)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        samples = list(ex.map(job, prompts))
    return samples, time.perf_counter() - t0
