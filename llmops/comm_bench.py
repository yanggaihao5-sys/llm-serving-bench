"""LONG-WAVE：社区可比的长输出基准（7K 输入 / 1K 输出 / c=32 / 3 轮）。

与 FROZEN-v1 的关系
-----------------------
复用同一端点、同一温度、同一「按 usage.completion_tokens 整请求求和」口径，
但**发射方式固定为「一波」**：每次重复同时发出 c=32 条、每条 max_tokens=1024，
等全部落定后再发下一轮，吞吐 = Σcompletion_tokens / 实测 wall。

为什么不按文档的 60 s 窗口当主指标
--------------------------------
c=32 实测聚合解码约 138 tok/s（CONN smoke），32×1024 token 一波保守需要
>200 s；60 s 窗口内完成请求数会接近 0，主指标直接失效。因此：
- 主指标 = 一波的真实耗时吞吐（本文件 `throughput_tok_s`）；
- 副指标 = 60 s 窗口口径（`window_60s`），照文档留痕，用来看「窗口口径为何不适用」。
两者都进 bench.json，不做事后择一。

互不打扰（本轮硬约束：CONN 在跑时不得叠加）
---------------------------------------------
开跑前过三道门：
1. 本机没有别的 llmops 压测进程（psutil 读 cmdline，读不到就按「忙」处理）；
2. `.locks/endpoint.lock.json` 没有活进程持有；
3. runs/ 下没有过去 `--idle-quiet-s` 秒内被写过的 events.jsonl。
三道全过才写租约；退出时删租约。`--no-gate` 只给确定独占时用。

用法：
  python -m llmops.comm_bench --repeats 3
  python -m llmops.comm_bench --fake            # 假数据自检，不碰网络，输出到 .tmp-comm-dryrun
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .protocol import default_endpoint, default_model

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BENCH = ROOT / "bench"
LOCK_DIR = ROOT / ".locks"
LOCK = LOCK_DIR / "endpoint.lock.json"

PROTOCOL = {
    "protocol_id": "LONG-WAVE",
    "endpoint": default_endpoint(),
    "model": default_model(),
    "concurrency": 32,
    "max_tokens": 1024,
    "repeats": 3,
    "temperature": 0,
    "dispatch": "wave：c=32 同时发出，全部落定后才发下一轮（无 sleep）",
    "throughput_rule": "Σ usage.completion_tokens / 实测 wall（min t_send → max t_end）",
    "window_secondary_s": 60.0,
    "prompt_set": "bench/prompts-frozen-v1.jsonl",
    "meta": "bench/prompts-frozen-v1.meta.json",
    "note": "bring your own long-input prompt set via --prompt-set/--meta",
}

BENCH_MODULES = ("llmops.conn_bench", "llmops.frozen_bench", "llmops.comm_bench",
                 "conn_bench.py", "frozen_bench.py", "comm_bench.py", "bench.py")


# ------------------------------------------------------------------ 门禁
def live_bench_pids() -> list[dict]:
    try:
        import psutil
    except ImportError:
        return [{"pid": -1, "reason": "psutil 缺失，无法确认独占"}]
    me = os.getpid()
    busy = []
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == me:
                continue
            if not (proc.info["name"] or "").lower().startswith("python"):
                continue
            cmd = " ".join(proc.info["cmdline"] or [])
            if not cmd:
                busy.append({"pid": proc.info["pid"], "reason": "cmdline 不可读，按忙处理"})
                continue
            if any(m in cmd for m in BENCH_MODULES):
                busy.append({"pid": proc.info["pid"], "reason": cmd[:160]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            busy.append({"pid": proc.info.get("pid"), "reason": "AccessDenied，按忙处理"})
    return busy


def lease_holder() -> dict | None:
    if not LOCK.exists():
        return None
    try:
        rec = json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"pid": -1, "reason": "租约文件损坏，按忙处理"}
    pid = rec.get("pid")
    try:
        import psutil
        if pid and psutil.pid_exists(int(pid)):
            return rec
    except ImportError:
        return rec
    try:
        LOCK.unlink()
    except OSError:
        pass
    return None


def recently_written_runs(quiet_s: float) -> list[str]:
    now = time.time()
    hot = []
    if not RUNS.exists():
        return hot
    for p in RUNS.glob("*/events.jsonl"):
        try:
            age = now - p.stat().st_mtime
        except OSError:
            continue
        if age < quiet_s:
            hot.append(f"{p.parent.name} ({age:.0f}s ago)")
    return hot


def wait_endpoint_idle(quiet_s: float, max_min: float, poll_s: float = 15.0) -> dict:
    """等端点真空闲；返回门禁快照。超时抛 RuntimeError。"""
    deadline = time.time() + max_min * 60
    log = []
    while True:
        busy = live_bench_pids()
        holder = lease_holder()
        hot = recently_written_runs(quiet_s)
        snap = {"ts": round(time.time(), 1), "busy_pids": busy, "lease": holder,
                "recent_runs": hot}
        if not busy and not holder and not hot:
            log.append(snap)
            return {"ok": True, "checks": log}
        log.append(snap)
        if time.time() > deadline:
            raise RuntimeError(
                f"等端点空闲超时（{max_min} min）：busy={busy} lease={holder} recent={hot}")
        print(f"[gate] 端点忙，等待 {poll_s:.0f}s · pids={len(busy)} "
              f"lease={bool(holder)} recent={hot}", flush=True)
        time.sleep(poll_s)


def acquire_lease(experiment: str, run_id: str) -> None:
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        raise RuntimeError(f"租约已被占用：{LOCK.read_text(encoding='utf-8')[:200]}")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "experiment": experiment,
                                "run_id": run_id, "acquired_at": round(time.time(), 3)},
                               ensure_ascii=False), encoding="utf-8")


def release_lease() -> None:
    try:
        rec = json.loads(LOCK.read_text(encoding="utf-8"))
        if rec.get("pid") == os.getpid():
            LOCK.unlink()
    except (OSError, json.JSONDecodeError):
        pass


# ------------------------------------------------------------------ 单请求
def stream_once(endpoint: str, model: str, prompt_text: str, max_tokens: int,
                api_key: str, timeout: float = 900.0) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": prompt_text}],
               "max_tokens": max_tokens, "temperature": PROTOCOL["temperature"],
               "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t_send = time.perf_counter()
    first = None
    chunks = 0
    usage = None
    finish = None
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
            choice = (obj.get("choices") or [{}])[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if first is None and (delta.get("content") or delta.get("reasoning_content")):
                first = time.perf_counter()
            if delta.get("content"):
                chunks += 1
        t_end = time.perf_counter()
        toks = int(usage.get("completion_tokens") or chunks) if usage else chunks
        details = (usage or {}).get("prompt_tokens_details") or {}
        return {"ok": True, "t_send": t_send, "t_first": first or t_end, "t_end": t_end,
                "completion_tokens": toks, "prompt_tokens": (usage or {}).get("prompt_tokens"),
                "cached_tokens": details.get("cached_tokens"), "chunks": chunks,
                "finish_reason": finish}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                "error": repr(exc), "completion_tokens": 0}


def fake_send(endpoint: str, model: str, text: str, max_tokens: int, api_key: str) -> dict:
    """假发送器：固定 TTFT 0.9 s + 每条 token 30 ms（c=32 → 约 1000 tok/s 聚合）。"""
    t0 = time.perf_counter()
    time.sleep(0.9)
    t_first = time.perf_counter()
    time.sleep(max_tokens * 0.030)
    t_end = time.perf_counter()
    return {"ok": True, "t_send": t0, "t_first": t_first, "t_end": t_end,
            "completion_tokens": max_tokens, "prompt_tokens": 7010, "cached_tokens": 0,
            "chunks": max_tokens // 4, "finish_reason": "length"}


# ------------------------------------------------------------------ 一波
def run_wave(prompts: list[dict], send, endpoint: str, model: str, api_key: str,
             concurrency: int, max_tokens: int) -> dict:
    records: list[dict] = []
    lock = threading.Lock()
    t_wave = time.perf_counter()

    def one(p: dict) -> None:
        rec = send(endpoint, model, p["text"], max_tokens, api_key)
        rec["prompt_id"] = p["id"]
        with lock:
            records.append(rec)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, prompts[:concurrency]))
    wall = time.perf_counter() - t_wave
    t0 = min(r["t_send"] for r in records)
    for rec in records:
        rec["t_start"] = round(rec["t_send"] - t0, 4)
        rec["t_end_rel"] = round(rec["t_end"] - t0, 4)
        rec["ttft_s"] = round(rec["t_first"] - rec["t_send"], 4)
        rec["e2e_s"] = round(rec["t_end"] - rec["t_send"], 4)
        if rec["ok"] and rec["completion_tokens"]:
            rec["tpot_s"] = round((rec["t_end"] - rec["t_first"]) / rec["completion_tokens"], 5)
            if rec.get("prompt_tokens"):
                rec["prefill_tok_s"] = round(rec["prompt_tokens"] / max(rec["ttft_s"], 1e-9), 1)
    return {"records": records, "wall_s": round(wall, 3)}


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[k], 4)


def summarize_wave(wave: dict, window_s: float) -> dict:
    recs = wave["records"]
    ok = [r for r in recs if r["ok"]]
    failed = [r for r in recs if not r["ok"]]
    toks = sum(r["completion_tokens"] for r in ok)
    wall = wave["wall_s"]
    in_win = [r for r in ok if r["t_end_rel"] <= window_s]
    win_toks = sum(r["completion_tokens"] for r in in_win)
    ttfts = [r["ttft_s"] * 1000 for r in ok]
    e2es = [r["e2e_s"] * 1000 for r in ok]
    tpots = [r["tpot_s"] * 1000 for r in ok if r.get("tpot_s")]
    pre = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    pre_rate = [r["prefill_tok_s"] for r in ok if r.get("prefill_tok_s")]
    outs = [r["completion_tokens"] for r in ok]
    finishes: dict[str, int] = {}
    for r in ok:
        finishes[str(r.get("finish_reason"))] = finishes.get(str(r.get("finish_reason")), 0) + 1
    return {
        "n": len(recs), "ok": len(ok), "failed": len(failed),
        "wall_s": wall,
        "throughput_tok_s": round(toks / wall, 2) if wall else 0.0,
        "req_s": round(len(ok) / wall, 3) if wall else 0.0,
        "completion_tokens": toks,
        "output_tokens": {"min": min(outs) if outs else 0,
                          "median": statistics.median(outs) if outs else 0,
                          "max": max(outs) if outs else 0},
        "prompt_tokens": {"min": min(pre) if pre else 0,
                          "median": statistics.median(pre) if pre else 0,
                          "max": max(pre) if pre else 0},
        "ttft_ms": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95)},
        "e2e_ms": {"p50": pct(e2es, 50), "p95": pct(e2es, 95)},
        "tpot_ms": {"p50": pct(tpots, 50), "p95": pct(tpots, 95)},
        "prefill_tok_s": {"p50": pct(pre_rate, 50)},
        "cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in ok),
        "finish_reason": finishes,
        "window_60s": {
            "window_s": window_s, "n": len(in_win),
            "completion_tokens": win_toks,
            "throughput_tok_s": round(win_toks / window_s, 2),
            "note": "文档 60 s 窗口口径，仅留痕：1K 输出一波 >200 s，窗口内完成数天然接近 0",
        },
        "acceptance_rate": None,
        "acceptance_note": "DSpark 接受率 API 不返回；需 T12-PROF 或服务端日志，本轮留空不估算",
    }


def aggregate(repeats: list[dict]) -> dict:
    def stats(vals: list[float]) -> dict:
        med = statistics.median(vals)
        return {"values": [round(v, 2) for v in vals], "median": round(med, 2),
                "min": round(min(vals), 2), "max": round(max(vals), 2),
                "range_pct": round((max(vals) - min(vals)) / med * 100, 2) if med else 0.0}

    def series(fn):
        return [fn(r["summary"]) for r in repeats]

    return {
        "repeats": len(repeats),
        "throughput_tok_s": stats(series(lambda s: s["throughput_tok_s"])),
        "wall_s": stats(series(lambda s: s["wall_s"])),
        "ttft_ms_p50": stats(series(lambda s: s["ttft_ms"]["p50"])),
        "ttft_ms_p95": stats(series(lambda s: s["ttft_ms"]["p95"])),
        "tpot_ms_p50": stats(series(lambda s: s["tpot_ms"]["p50"])),
        "e2e_ms_p50": stats(series(lambda s: s["e2e_ms"]["p50"])),
        "window_60s_tok_s": stats(series(lambda s: s["window_60s"]["throughput_tok_s"])),
        "window_60s_n": stats(series(lambda s: s["window_60s"]["n"])),
        "failed": stats(series(lambda s: s["failed"])),
        "stability_rule": "R=(max-min)/median；判据 R<10%，3 次重复如实报告，不事后放宽",
    }


# ------------------------------------------------------------------ 主流程
def load_prompts(set_path: Path, meta_path: Path) -> dict:
    import hashlib
    raw = set_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if meta.get("sha256") and meta["sha256"] != digest:
        raise ValueError(f"prompt 集哈希漂移：meta={meta['sha256'][:12]} file={digest[:12]}")
    rows = [json.loads(l) for l in raw.decode("utf-8").splitlines() if l.strip()]
    try:
        rel = str(set_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        rel = str(set_path)
    return {"path": rel, "sha256": digest,
            "count": len(rows), "prompts": rows, "meta": meta}


def probe_env(endpoint: str, model: str, api_key: str) -> dict:
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 4, "temperature": 0, "stream": False}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        obj = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        return {"ok": True, "system_fingerprint": obj.get("system_fingerprint"),
                "usage": obj.get("usage")}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LONG-WAVE 长输出基准（7K/1K/c=32）")
    ap.add_argument("--repeats", type=int, default=PROTOCOL["repeats"])
    ap.add_argument("--concurrency", type=int, default=PROTOCOL["concurrency"])
    ap.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"])
    ap.add_argument("--window-secondary", type=float, default=PROTOCOL["window_secondary_s"])
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--gap-s", type=float, default=15.0, help="两轮之间的排空间隔")
    ap.add_argument("--endpoint", default=default_endpoint())
    ap.add_argument("--model", default=default_model())
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--prompt-set", default=str(BENCH / "prompts-frozen-v1.jsonl"))
    ap.add_argument("--meta", default=str(BENCH / "prompts-frozen-v1.meta.json"))
    ap.add_argument("--out-root", default=str(RUNS))
    ap.add_argument("--run-id", default="")
    ap.add_argument("--idle-quiet-s", type=float, default=120.0)
    ap.add_argument("--idle-max-min", type=float, default=45.0)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--fake", action="store_true", help="假数据自检，不碰网络")
    args = ap.parse_args(argv)

    api_key = "" if args.fake else os.environ.get(args.api_key_env, "")
    if not args.fake and not api_key:
        print(f"[fatal] 环境变量 {args.api_key_env} 为空：网关需鉴权")
        return 2

    pset = load_prompts(Path(args.prompt_set), Path(args.meta))
    if pset["count"] < args.repeats * args.concurrency:
        print(f"[fatal] prompt 数 {pset['count']} < repeats×concurrency "
              f"{args.repeats * args.concurrency}：跨轮会复用文档，可能命中前缀缓存")
        return 2

    out_root = Path(args.out_root)
    run_id = args.run_id or (f"LONG-WAVE-{'FAKE' if args.fake else 'RUN'}-"
                             f"{time.strftime('%Y%m%d-%H%M%S')}")
    run_dir = out_root / run_id
    endpoint = args.endpoint if getattr(args, "endpoint", None) else PROTOCOL["endpoint"]
    model = args.model if getattr(args, "model", None) else PROTOCOL["model"]
    send = fake_send if args.fake else stream_once

    gate = {"ok": True, "skipped": "fake" if args.fake else ("--no-gate" if args.no_gate else False)}
    if not args.fake and not args.no_gate:
        print(f"[gate] 等端点空闲（quiet={args.idle_quiet_s}s, max={args.idle_max_min}min）",
              flush=True)
        gate = wait_endpoint_idle(args.idle_quiet_s, args.idle_max_min)
        print(f"[gate] 通过：{json.dumps(gate['checks'][-1], ensure_ascii=False)[:200]}", flush=True)
    if not args.fake:
        acquire_lease("LONG-WAVE", run_id)

    env = {"ok": True, "system_fingerprint": "fake", "usage": {}} if args.fake \
        else probe_env(endpoint, model, api_key)
    print(f"run_id={run_id} fingerprint={env.get('system_fingerprint')}", flush=True)
    print(f"prompt_set={pset['path']} sha256={pset['sha256'][:12]} count={pset['count']}", flush=True)
    print(f"protocol: c={args.concurrency} max_tokens={args.max_tokens} "
          f"repeats={args.repeats} wave/drain wall-measured", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict] = []
    metrics: list[dict] = []
    repeats: list[dict] = []
    samples: list[dict] = []

    def ev(name: str, **extra) -> None:
        row = {"ts": round(time.time(), 3), "event": name, **extra}
        events.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {extra}", flush=True)

    ev("start", run_id=run_id, gate=gate.get("skipped") or "checked")
    try:
        for r in range(1, args.repeats + 1):
            offset = (r - 1) * args.concurrency
            batch_prompts = pset["prompts"][offset:offset + args.concurrency]
            if r == 1 and args.warmup:
                warm_ok = 0
                for p in batch_prompts[:args.warmup]:
                    warm_ok += int(bool(send(endpoint, model, p["text"], 8, api_key)["ok"]))
                ev("warmup", repeat=r, requests=args.warmup, ok=warm_ok)

            wave = run_wave(batch_prompts, send, endpoint, model, api_key,
                            args.concurrency, args.max_tokens)
            summary = summarize_wave(wave, args.window_secondary)
            row = {"repeat": r, "summary": summary,
                   "prompt_ids": [p["id"] for p in batch_prompts]}
            repeats.append(row)
            for rec in wave["records"]:
                rec["repeat"] = r
                samples.append(rec)
            raw = run_dir / "raw" / f"samples-7k1k-r{r}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in wave["records"]),
                           encoding="utf-8")
            metrics += [
                {"ts": round(time.time(), 3), "step": r, "name": "throughput_tok_s",
                 "value": summary["throughput_tok_s"]},
                {"ts": round(time.time(), 3), "step": r, "name": "wall_s",
                 "value": summary["wall_s"]},
                {"ts": round(time.time(), 3), "step": r, "name": "ttft_ms_p50",
                 "value": summary["ttft_ms"]["p50"]},
                {"ts": round(time.time(), 3), "step": r, "name": "tpot_ms_p50",
                 "value": summary["tpot_ms"]["p50"]},
                {"ts": round(time.time(), 3), "step": r, "name": "prompt_tokens_median",
                 "value": summary["prompt_tokens"]["median"]},
            ]
            print(f"repeat {r}/{args.repeats}: tok/s={summary['throughput_tok_s']} "
                  f"wall={summary['wall_s']}s ok={summary['ok']} failed={summary['failed']} "
                  f"ttft_p50={summary['ttft_ms']['p50']}ms tpot_p50={summary['tpot_ms']['p50']}ms "
                  f"prompt_tok_med={summary['prompt_tokens']['median']} "
                  f"cached={summary['cached_tokens']}", flush=True)
            if r < args.repeats:
                time.sleep(args.gap_s)
        ev("stop", run_id=run_id)
    finally:
        if not args.fake:
            release_lease()

    agg = aggregate(repeats)
    run_meta = {
        "run_id": run_id, "target": "default", "kind": "llm-serving",
        "experiment": "LONG-WAVE", "created_at": time.time(),
        "status": "finished" if len(repeats) == args.repeats else "partial",
        "model": model,
        "engine": {"name": "vllm-ascend", "system_fingerprint": env.get("system_fingerprint"),
                   "version_note": "指纹给引擎版本与并行度；镜像 digest 未取证（无宿主机权限）",
                   "parallelism": "tp8-ep-dp1"},
        "load": {"mode": "wave", "concurrency": args.concurrency,
                 "max_tokens": args.max_tokens, "repeats": args.repeats,
                 "temperature": 0, "warmup": args.warmup,
                 "prompt_set": pset["path"], "prompt_sha256": pset["sha256"],
                 "prompt_target_tokens": pset["meta"].get("target_tokens"),
                 "prompt_tokens_measured_meta": (pset["meta"].get("tokens_measured") or {}).get("median")},
        "protocol_id": PROTOCOL["protocol_id"],
        "protocol": PROTOCOL,
        "protocol_deviation": {
            "primary_metric": "wave wall-clock throughput，不是文档的 60 s 窗口",
            "reason": "1K 输出 × c=32 一波 >200 s，60 s 窗口内完成数接近 0，主指标失效",
            "secondary_kept": "window_60s 字段照文档留痕，两个口径都在 bench.json",
            "approved_by": "用户 2026-11-10 20:45 选择「一波跑完算真实耗时（推荐）」",
        },
        "endpoint": endpoint,
        "env_probe": env,
        "gate": gate,
        "finished_at": time.time(),
    }
    (run_dir / "run.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in events), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in metrics), encoding="utf-8")
    bench = {"protocol_id": PROTOCOL["protocol_id"], "protocol": PROTOCOL,
             "repeats": repeats, "aggregate": agg, "samples": samples,
             "env_probe": env,
             "prompt_set": {"path": pset["path"], "sha256": pset["sha256"],
                            "count": pset["count"], "meta": pset["meta"]}}
    (run_dir / "bench.json").write_text(json.dumps(bench, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print("\n=== aggregate ===", flush=True)
    print(json.dumps({k: v for k, v in agg.items() if k != "stability_rule"},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"\nrun -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())