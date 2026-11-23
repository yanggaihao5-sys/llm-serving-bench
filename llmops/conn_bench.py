"""CONN：客户端连接方式对照（A=urllib 每请求新连 / B=requests.Session 复用）。

只替换 transport，窗口边界、发射模式、统计口径全部复用 frozen_bench（FROZEN-v1）：
- 连续发射 32 worker，一空闲立即发下一条；
- 60 s 窗口，按 usage.completion_tokens 整请求求和；
- 冻结 prompt 集（sha256 51e1a484…）、c=32、max_tokens=64、temp=0、5 次。

用法：
  python -m llmops.conn_bench --client urllib  --repeats 5
  python -m llmops.conn_bench --client session --repeats 5
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

from . import frozen_bench as fb
from .protocol import (PROTOCOL, load_prompt_set, default_endpoint,
                         default_model)

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = {"urllib": "A", "session": "B"}
_tls = threading.local()


def session_once(endpoint: str, model: str, prompt_text: str, max_tokens: int,
                 api_key: str, timeout: float = 300.0) -> dict:
    """B 组：每个 worker 线程复用一条 requests.Session（连接池）。"""
    import requests  # 延迟导入：只在 B 组需要

    sess = getattr(_tls, "sess", None)
    if sess is None:
        sess = requests.Session()
        _tls.sess = sess
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": PROTOCOL["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    t_send = time.perf_counter()
    first = None
    chunks = 0
    usage = None
    try:
        resp = sess.post(endpoint, json=payload, headers=headers, stream=True,
                         timeout=timeout)
        if resp.status_code != 200:
            body = resp.text[:200]
            resp.close()
            return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                    "error": f"HTTP {resp.status_code}: {body}", "completion_tokens": 0}
        for raw in resp.iter_lines():
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
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
                first = time.perf_counter()
            if delta.get("content"):
                chunks += 1
        t_end = time.perf_counter()
        toks = int(usage.get("completion_tokens") or chunks) if usage else chunks
        details = (usage or {}).get("prompt_tokens_details") or {}
        return {
            "ok": True, "t_send": t_send, "t_first": first or t_end, "t_end": t_end,
            "completion_tokens": toks,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "chunks": chunks,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                "error": repr(exc), "completion_tokens": 0}


def transport_for(client: str):
    return fb.stream_once if client == "urllib" else session_once


def probe_env(client: str, endpoint: str, model: str, api_key: str) -> dict:
    """非流式探一次，换连接方式后仍走同一客户端。"""
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 4, "temperature": 0, "stream": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    try:
        if client == "urllib":
            import urllib.request
            req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                         method="POST", headers=headers)
            obj = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        else:
            import requests
            r = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            obj = r.json()
        return {"system_fingerprint": obj.get("system_fingerprint"),
                "usage": obj.get("usage"), "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CONN 客户端连接方式对照")
    ap.add_argument("--client", choices=["urllib", "session"], required=True)
    ap.add_argument("--repeats", type=int, default=PROTOCOL["repeats"])
    ap.add_argument("--window-s", type=float, default=PROTOCOL["window_s"])
    ap.add_argument("--concurrency", type=int, default=PROTOCOL["concurrency"])
    ap.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"])
    ap.add_argument("--warmup", type=int, default=PROTOCOL["warmup_requests"])
    ap.add_argument("--endpoint", default=default_endpoint())
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--out-root", default=os.environ.get(
        "LLM_LAB_RUNS", str(ROOT / "runs")))
    ap.add_argument("--run-id", default="")
    args = ap.parse_args(argv)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"[fatal] 环境变量 {args.api_key_env} 为空：网关 /v1/chat/completions 需鉴权")
        return 2

    prompt_set = load_prompt_set()
    prompts = prompt_set["prompts"]
    endpoint = args.endpoint
    model = args.model
    send = transport_for(args.client)
    group = CLIENTS[args.client]
    run_id = args.run_id or f"CONN-{time.strftime('%Y%m%d-%H%M%S')}-{group}"
    run_dir = Path(args.out_root) / run_id

    env = probe_env(args.client, endpoint, model, api_key)
    print(f"run_id={run_id} client={args.client}({group}) fingerprint={env.get('system_fingerprint')}")
    print(f"prompt_set={prompt_set['path']} sha256={prompt_set['sha256']} count={prompt_set['count']}")
    print(f"c={args.concurrency} max_tokens={args.max_tokens} window={args.window_s}s "
          f"temp=0 repeats={args.repeats} warmup={args.warmup}")

    run_dir.mkdir(parents=True, exist_ok=True)
    events, metrics, repeats, samples = [], [], [], []

    def ev(name: str, **extra) -> None:
        row = {"ts": round(time.time(), 3), "event": name, **extra}
        events.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {extra}")

    ev("start", run_id=run_id, client=args.client)
    for r in range(1, args.repeats + 1):
        warm_start = time.perf_counter()
        warm_ok = 0
        for p in fb.prompt_cycle(prompts, args.warmup):
            rec = send(endpoint, model, p["text"], args.max_tokens, api_key)
            warm_ok += int(bool(rec["ok"]))
        warm_s = round(time.perf_counter() - warm_start, 3)
        ev("warmup", repeat=r, requests=args.warmup, ok=warm_ok, seconds=warm_s)

        win = fb.run_window("continuous", prompts, endpoint, model, api_key,
                            args.concurrency, args.max_tokens, args.window_s,
                            transport=send)
        summary = fb.summarize_window(win["records"], args.window_s)
        row = {"repeat": r, "summary": summary, "wall_s": win["wall_s"],
               "peak_inflight": win["peak_inflight"],
               "interval": fb.interval_stats(win["records"], "continuous", None),
               "warmup": {"requests": args.warmup, "ok": warm_ok, "seconds": warm_s}}
        repeats.append(row)
        for rec in win["records"]:
            rec["repeat"] = r
            samples.append(rec)
        raw_path = run_dir / "raw" / f"samples-{args.client}-r{r}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in win["records"]),
                            encoding="utf-8")
        metrics += [
            {"ts": round(time.time(), 3), "step": r, "name": "throughput_tok_s",
             "value": summary["throughput_tok_s"]},
            {"ts": round(time.time(), 3), "step": r, "name": "requests_in_window",
             "value": summary["n"]},
            {"ts": round(time.time(), 3), "step": r, "name": "ttft_ms_p50",
             "value": summary["ttft_ms"]["p50"]},
            {"ts": round(time.time(), 3), "step": r, "name": "e2e_ms_p50",
             "value": summary["e2e_ms"]["p50"]},
        ]
        print(f"repeat {r}/{args.repeats}: tok/s={summary['throughput_tok_s']} n={summary['n']} "
              f"failed={summary['failed']} ttft_p50={summary['ttft_ms']['p50']}ms "
              f"e2e_p50={summary['e2e_ms']['p50']}ms drained={summary['drained']}")

    ev("stop", run_id=run_id)
    agg = fb.aggregate(repeats)
    run_meta = {
        "run_id": run_id, "target": "default", "kind": "llm-serving",
        "experiment": "CONN", "group": group,
        "created_at": time.time(), "status": "finished",
        "model": model,
        "engine": {"name": "vllm-ascend",
                   "system_fingerprint": env.get("system_fingerprint"),
                   "version_note": "指纹给出引擎版本与并行度；镜像 digest 未取证",
                   "parallelism": "tp8-ep-dp1"},
        "load": {"mode": "continuous", "client": args.client,
                 "client_note": {"urllib": "每请求新建 TCP 连接（本轮 129.93 用的）",
                                 "session": "每 worker 复用 requests.Session（历史 175 用的）"}[args.client],
                 "concurrency": args.concurrency, "max_tokens": args.max_tokens,
                 "window_s": args.window_s, "temperature": 0,
                 "repeats": args.repeats, "warmup": args.warmup,
                 "prompt_set": prompt_set["path"], "prompt_sha256": prompt_set["sha256"]},
        "protocol_id": "CONN",
        "endpoint": endpoint,
        "dispatch": "32 worker，一空闲立即发下一条（ASAP，无 sleep）",
        "env_probe": env,
        "finished_at": time.time(),
    }
    write_json(run_dir / "run.json", run_meta)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in events), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in metrics), encoding="utf-8")
    bench = {"protocol_id": "CONN", "client": args.client, "group": group,
             "repeats": repeats, "aggregate": agg,
             "samples": samples, "env_probe": env,
             "prompt_set": {"path": prompt_set["path"], "sha256": prompt_set["sha256"],
                            "count": prompt_set["count"]}}
    write_json(run_dir / "bench.json", bench)
    print("\n=== aggregate ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"\nrun -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())