
"""CONN-PAIR：三组客户端 + TCP 连接计数 + ABBA 配对(protocol: see docs/METHODOLOGY.md)。

组：
- A `urllib` 每请求新建 TCP 连接（复现 129.93 / 131.00 那条路径）
- B 每 worker 独立 `requests.Session`，`HTTPAdapter(pool_connections=32, pool_maxsize=32,
  pool_block=True)`（32 线程共享一个 Session 是 v1 的缺陷，本版禁止）
- C `requests.api.request()`，每次调用新建会话（把「HTTP 库差异」与「连接复用」拆开）

顺序：5 轮 ABBA（轮 1 A→B，轮 2 B→A，…），每次切换后串行预热 8 条，正式采样 60 s。

复用成立性：包装 `socket.socket.connect` 计新建连接数；窗口内每 5 s 采样 `netstat -ano`
统计到端点 ESTABLISHED 数。A 组新连 ≈ 请求数；B 组 ≤ 32 且不随请求数增长；
否则该 attempt 标 `reuse_not_achieved` 作废。

判据：`d_i = X_B,i − X_A,i`，报中位 d、均值 95% CI（n=5 t=2.776；n=10 t=2.262），
CI 下界 ≥ +10%（相对 A 中位）认正收益，上界 ≤ −10% 认负收益，其余「证据不足」。

用法：
  python -m llmops.conn_v2 --phase abba --rounds 5
  python -m llmops.conn_v2 --phase c --rounds 5
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import threading
import time
from pathlib import Path

from . import frozen_bench as fb
from .protocol import (PROTOCOL, load_prompt_set, default_endpoint,
                         default_model)

ROOT = Path(__file__).resolve().parents[1]
T975 = {4: 2.776, 9: 2.262, 14: 2.145}

# ---------------------------------------------------------------- 连接计数

_lock = threading.Lock()
_new_conn = 0
_orig_connect = None


def _counted_connect(self, address):
    global _new_conn
    with _lock:
        _new_conn += 1
    return _orig_connect(self, address)


def install_connection_counter() -> None:
    """包装 socket.socket.connect；urllib / urllib3 最终都走它。"""
    global _orig_connect
    if _orig_connect is None:
        _orig_connect = socket.socket.connect
        socket.socket.connect = _counted_connect


def reset_connection_counter() -> None:
    global _new_conn
    with _lock:
        _new_conn = 0


def connection_count() -> int:
    with _lock:
        return _new_conn


# ---------------------------------------------------------------- ESTABLISHED 采样


def _netstat_estab(target: str = "") -> int:
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True,
                             timeout=20, errors="replace").stdout
    except Exception:  # noqa: BLE001
        return -1
    n = 0
    for line in out.splitlines():
        parts = line.split()
        hostport = target.split("//")[-1].split("/")[0] if target else ""
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "ESTABLISHED" \
                and (not hostport or parts[2].endswith(hostport)):
            n += 1
    return n


class _EstabSampler:
    """窗口期间每 interval 秒采一次 ESTABLISHED 数，记录峰值与非负样本。"""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.samples: list[int] = []
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            n = _netstat_estab()
            if n >= 0:
                self.samples.append(n)
            self._stop.wait(self.interval)

    def __enter__(self):
        n = _netstat_estab()
        if n >= 0:
            self.samples.append(n)
        self._th.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join(timeout=self.interval + 2)
        return False

    @property
    def peak(self) -> int:
        return max(self.samples) if self.samples else -1


# ---------------------------------------------------------------- 三组 transport

_tls = threading.local()


def urllib_once(endpoint, model, text, max_tokens, api_key, timeout: float = 300.0):
    """A 组：urllib 每请求新建连接（与 FROZEN 完全同路径）。"""
    return fb.stream_once(endpoint, model, text, max_tokens, api_key, timeout)


def _parse_sse(resp, t_send: float) -> dict:
    first = None
    chunks = 0
    usage = None
    # 注意：读到 [DONE] 后**不能 break**——提前中断会让 urllib3 认为响应未读尽，
    # 直接丢弃连接，keep-alive 失效（实测 new_conn≈请求数）。继续读到流自然结束。
    for raw in resp.iter_lines():
        line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
        if not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            continue
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
    return {"ok": True, "t_send": t_send, "t_first": first or t_end, "t_end": t_end,
            "completion_tokens": toks, "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"), "chunks": chunks}


def _new_pooled_session():
    import requests
    from requests.adapters import HTTPAdapter
    sess = requests.Session()
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32, pool_block=True,
                          max_retries=0)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def session_once(endpoint, model, text, max_tokens, api_key, timeout: float = 300.0):
    """B 组：每 worker 独立 Session + 池 32 阻塞（thread-local 实现「每 worker」）。"""
    sess = getattr(_tls, "sess", None)
    if sess is None:
        sess = _new_pooled_session()
        _tls.sess = sess
    payload = {"model": model, "messages": [{"role": "user", "content": text}],
               "max_tokens": max_tokens, "temperature": PROTOCOL["temperature"],
               "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    t_send = time.perf_counter()
    try:
        resp = sess.post(endpoint, json=payload, headers=headers, stream=True, timeout=timeout)
        if resp.status_code != 200:
            body = resp.text[:200]
            resp.close()
            return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                    "error": f"HTTP {resp.status_code}: {body}", "completion_tokens": 0}
        return _parse_sse(resp, t_send)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                "error": repr(exc), "completion_tokens": 0}


def noreuse_once(endpoint, model, text, max_tokens, api_key, timeout: float = 300.0):
    """C 组：requests 每次调用新建会话（不复用）。"""
    import requests
    payload = {"model": model, "messages": [{"role": "user", "content": text}],
               "max_tokens": max_tokens, "temperature": PROTOCOL["temperature"],
               "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    t_send = time.perf_counter()
    try:
        resp = requests.request("POST", endpoint, json=payload, headers=headers,
                                stream=True, timeout=timeout)
        if resp.status_code != 200:
            body = resp.text[:200]
            resp.close()
            return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                    "error": f"HTTP {resp.status_code}: {body}", "completion_tokens": 0}
        return _parse_sse(resp, t_send)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                "error": repr(exc), "completion_tokens": 0}


TRANSPORTS = {"A": urllib_once, "B": session_once, "C": noreuse_once}


# ---------------------------------------------------------------- 配对统计


def paired_stats(a: list[float], b: list[float], label_a="A", label_b="B") -> dict:
    """d_i = b_i − a_i；报中位、均值 95% CI（t 分布）。"""
    if len(a) != len(b) or not a:
        raise ValueError("配对长度不一致或为空")
    d = [bi - ai for ai, bi in zip(a, b)]
    n = len(d)
    med_a = statistics.median(a)
    med_b = statistics.median(b)
    med_d = statistics.median(d)
    mean_d = statistics.mean(d)
    sd = statistics.stdev(d) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    t = T975.get(n - 1, 2.0)
    lo, hi = mean_d - t * se, mean_d + t * se
    return {
        "pairs": n, "label_a": label_a, "label_b": label_b,
        "median_a": round(med_a, 2), "median_b": round(med_b, 2),
        "median_d": round(med_d, 2), "mean_d": round(mean_d, 2),
        "sd_d": round(sd, 2), "se_d": round(se, 2), "t": t,
        "ci95": [round(lo, 2), round(hi, 2)],
        "d_pct_of_a": round(med_d / med_a * 100, 2) if med_a else None,
        "ci_pct_of_a": [round(lo / med_a * 100, 2), round(hi / med_a * 100, 2)] if med_a else None,
        "d_values": [round(x, 2) for x in d],
        "r_a_pct": round((max(a) - min(a)) / med_a * 100, 2) if med_a else None,
        "r_b_pct": round((max(b) - min(b)) / med_b * 100, 2) if med_b else None,
    }


def verdict(stats: dict) -> str:
    """按 §4.1 判据给结论（相对 A 中位）。"""
    lo, hi = stats["ci_pct_of_a"]
    if lo >= 10:
        return "正收益（CI 下界 ≥ +10%）"
    if hi <= -10:
        return "负收益（CI 上界 ≤ −10%）"
    return "证据不足（CI 跨 0 或幅度不足 10%）"


# ---------------------------------------------------------------- 单组一个窗口


def run_group_window(group: str, prompts, endpoint, model, api_key, concurrency,
                     max_tokens, window_s, warmup, estab_interval: float = 5.0) -> dict:
    """预热 → 计数 → 采样 → 60 s 窗口。返回窗口记录 + 连接计数 + 峰值 ESTABLISHED。"""
    send = TRANSPORTS[group]
    warm_start = time.perf_counter()
    ok = 0
    for p in fb.prompt_cycle(prompts, warmup):
        rec = send(endpoint, model, p["text"], max_tokens, api_key)
        ok += int(bool(rec["ok"]))
    warm_s = round(time.perf_counter() - warm_start, 3)

    reset_connection_counter()
    with _EstabSampler(interval=estab_interval) as sampler:
        win = fb.run_window("continuous", prompts, endpoint, model, api_key,
                            concurrency, max_tokens, window_s, transport=send)
    return {
        "group": group,
        "summary": fb.summarize_window(win["records"], window_s),
        "wall_s": win["wall_s"],
        "new_connections": connection_count(),
        "peak_established": sampler.peak,
        "estab_samples": sampler.samples,
        "warmup": {"requests": warmup, "ok": ok, "seconds": warm_s},
        "records": win["records"],
    }


def pair_stats(groups: list[dict], pair: tuple[str, str]) -> dict | None:
    """从 groups 里按轮号取配对，返回配对统计；无合法配对返回 None。"""
    rounds = sorted({g["round"] for g in groups})
    a_vals, b_vals, used = [], [], []
    for r in rounds:
        rows = {row["group"]: row for row in groups if row["round"] == r}
        if pair[0] in rows and pair[1] in rows:
            ra, rb = rows[pair[0]], rows[pair[1]]
            if "invalid_reason" not in rb and "invalid_reason" not in ra:
                a_vals.append(ra["throughput_tok_s"])
                b_vals.append(rb["throughput_tok_s"])
                used.append(r)
    if not a_vals:
        return None
    out = paired_stats(a_vals, b_vals, pair[0], pair[1])
    out["used_rounds"] = used
    out["verdict"] = verdict(out)
    return out


def load_previous_groups(path: str | Path) -> tuple[list[dict], str]:
    """读取前一个 run 的 groups，轮号顺延，返回 (groups, 来源 run 目录)。"""
    p = Path(path)
    if p.is_dir():
        p = p / "bench.json"
    prev = json.loads(p.read_text(encoding="utf-8"))
    groups = [dict(g) for g in prev["groups"]]
    offset = max(g["round"] for g in groups)
    for g in groups:
        g["round"] += offset
    return groups, str(p.parent)


# ---------------------------------------------------------------- ABBA / C 主循环


def abba_order(round_index: int) -> tuple[str, str]:
    """轮 1 A→B，轮 2 B→A，交替。"""
    return ("A", "B") if round_index % 2 == 1 else ("B", "A")


def c_order(round_index: int) -> tuple[str, str]:
    """C 阶段：B/C 配对，同样交替。"""
    return ("B", "C") if round_index % 2 == 1 else ("C", "B")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CONN-PAIR 配对实验")
    ap.add_argument("--phase", choices=["abba", "c"], default="abba")
    ap.add_argument("--rounds", type=int, default=5)
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
    ap.add_argument("--merge-run", default="",
                    help="把该 run 的 groups 一并纳入配对统计（跨 attempt 合并，需在 summary 标注）")
    args = ap.parse_args(argv)

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"[fatal] 环境变量 {args.api_key_env} 为空")
        return 2
    install_connection_counter()

    prompt_set = load_prompt_set()
    prompts = prompt_set["prompts"]
    endpoint = args.endpoint
    model = args.model
    order_fn = abba_order if args.phase == "abba" else c_order
    pair = ("A", "B") if args.phase == "abba" else ("B", "C")
    tag = "ABBA" if args.phase == "abba" else "CVSB"
    run_id = args.run_id or f"CONN-PAIR-{time.strftime('%Y%m%d-%H%M%S')}-{tag}"
    run_dir = Path(args.out_root) / run_id

    print(f"run_id={run_id} phase={args.phase} pair={pair} rounds={args.rounds} "
          f"window={args.window_s}s c={args.concurrency}")

    events, groups, samples = [], [], []
    t_start = time.time()

    def ev(name: str, **extra) -> None:
        row = {"ts": round(time.time(), 3), "event": name, **extra}
        events.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {extra}", flush=True)

    ev("start", run_id=run_id, phase=args.phase)

    for r in range(1, args.rounds + 1):
        for g in order_fn(r):
            t_round = time.perf_counter()
            res = run_group_window(g, prompts, endpoint, model, api_key,
                                   args.concurrency, args.max_tokens, args.window_s,
                                   args.warmup)
            s = res["summary"]
            ok_rate = res["new_connections"] / s["n"] if s["n"] else 0
            row = {
                "round": r, "group": g, "throughput_tok_s": s["throughput_tok_s"],
                "n": s["n"], "failed": s["failed"], "issued": s["issued"],
                "ttft_p50_ms": s["ttft_ms"]["p50"], "ttft_p95_ms": s["ttft_ms"]["p95"],
                "e2e_p50_ms": s["e2e_ms"]["p50"], "e2e_p95_ms": s["e2e_ms"]["p95"],
                "tpot_p50_ms": s["tpot_ms"]["p50"],
                "new_connections": res["new_connections"],
                "connections_per_request": round(ok_rate, 3),
                "peak_established": res["peak_established"],
                "estab_samples": res["estab_samples"],
                "warmup_s": res["warmup"]["seconds"],
                "elapsed_s": round(time.perf_counter() - t_round, 1),
            }
            reuse_ok = True
            if g == "B" and res["new_connections"] > args.concurrency + 2:
                reuse_ok = False
                row["invalid_reason"] = "reuse_not_achieved"
            groups.append(row)
            for rec in res["records"]:
                rec["round"] = r
                rec["group"] = g
                samples.append(rec)
            raw = run_dir / "raw" / f"samples-{g}-r{r}.jsonl"
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in res["records"]),
                           encoding="utf-8")
            ev("window", round=r, group=g, tok_s=s["throughput_tok_s"], n=s["n"],
               failed=s["failed"], new_conn=res["new_connections"],
               peak_estab=res["peak_established"], reuse_ok=reuse_ok)

    # 配对：只取同一轮内两个组都有、且合法的配对；可用 --merge-run 合并前一个 attempt
    groups_all = list(groups)
    merged_from = ""
    if args.merge_run:
        prev_groups, merged_from = load_previous_groups(args.merge_run)
        groups_all = prev_groups + groups
    stats = pair_stats(groups_all, pair)
    if stats:
        stats["merged_from"] = merged_from
        if stats["pairs"] < 5:
            stats["note"] = "有效配对数 <5，CI 仅参考，需补跑到 5 轮"

    run_meta = {
        "run_id": run_id, "target": "default", "kind": "llm-serving",
        "experiment": "CONN-PAIR", "phase": args.phase, "pair": list(pair),
        "protocol_id": "CONN-PAIR", "created_at": t_start, "status": "finished",
        "model": model, "endpoint": endpoint,
        "load": {"mode": "continuous", "concurrency": args.concurrency,
                 "max_tokens": args.max_tokens, "window_s": args.window_s,
                 "temperature": 0, "rounds": args.rounds, "warmup": args.warmup,
                 "schedule": [list(order_fn(r)) for r in range(1, args.rounds + 1)],
                 "prompt_set": prompt_set["path"], "prompt_sha256": prompt_set["sha256"]},
        "client_notes": {
            "A": "urllib.request，每请求新建 TCP 连接",
            "B": "每 worker 独立 requests.Session + HTTPAdapter(pool_connections=32, "
                 "pool_maxsize=32, pool_block=True)",
            "C": "requests.api.request()，每次调用新建会话（不复用）",
        },
        "finished_at": time.time(),
    }
    write_json(run_dir / "run.json", run_meta)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in events), encoding="utf-8")
    bench = {"protocol_id": "CONN-PAIR", "phase": args.phase,
             "groups": groups, "groups_all": groups_all, "merged_from": merged_from,
             "paired": stats,
             "env": {"system_fingerprint": None}, "samples": len(samples)}
    write_json(run_dir / "bench.json", bench)

    print("\n=== groups ===")
    for row in groups:
        print(f"r{row['round']} {row['group']}: {row['throughput_tok_s']:.2f} t/s "
              f"n={row['n']} new_conn={row['new_connections']} "
              f"({row['connections_per_request']}/req) peak_estab={row['peak_established']} "
              f"{row.get('invalid_reason', '')}")
    print("\n=== paired ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2) if stats else "no valid pair")
    print(f"\nrun -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
