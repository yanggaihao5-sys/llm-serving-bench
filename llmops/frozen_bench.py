
"""FROZEN-v1 冻结协议压测器：连续发射 / 批次屏障，严格 60 s 窗口边界。

窗口边界（与《初始性能与目标》§2.1 一致）：
- t0 = 预热结束、开始发送的那一刻；分母固定 window_s；
- 分子 = finish 时刻 <= t0 + window_s 的请求，按 usage.completion_tokens 整请求求和；
- 跨窗请求整条不计；失败请求不计入分子，单独统计；
- t0 + window_s 停止发新请求，在飞请求自然结束（排空不计入本窗）。

发射模式：
- continuous：concurrency 个 worker，各自一结束立即发下一条（ASAP，无 sleep）；
- barrier：每批 concurrency 条同时发出，全部返回后才发下一批。

用法：
  python -m llmops.frozen_bench --mode continuous --repeats 5
  python -m llmops.frozen_bench --mode barrier --repeats 5
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:  # 包内导入（python -m llmops.frozen_bench）
    from .protocol import (PROTOCOL, load_prompt_set, prompt_cycle,
                           default_endpoint, default_model)
except ImportError:  # 直接执行脚本（python llmops\frozen_bench.py，此时 __package__ 为 None）
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from llmops.protocol import (PROTOCOL, load_prompt_set, prompt_cycle,
                                 default_endpoint, default_model)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- 单请求

def stream_once(endpoint: str, model: str, prompt_text: str, max_tokens: int,
                api_key: str, timeout: float = 300.0,
                priority: int | None = None) -> dict:
    """发一条流式请求，返回时间戳与 usage 统计（不依赖 chunk 数当 token）。

    priority：**仅当调用方显式给出时才写进 body**。默认 None = 不传字段，
    与历史基线（c=40 176.92 / BS32 163.48 等）完全等价。必须如此，因为
    vLLM `chat_completion/protocol.py:357` 明写「Any priority other than 0
    will raise an error if the served model does not use priority scheduling」
    ⇒ fcfs 模式下传非 0 会被协议层拒绝；而传 0 虽合法，但会让 body 与历史
    请求不一致，破坏可比性。
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": PROTOCOL["temperature"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if priority is not None:
        payload["priority"] = int(priority)
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    t_send = time.perf_counter()
    first = None
    content_chunks = 0
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
                first = time.perf_counter()
            if delta.get("content"):
                content_chunks += 1
        t_end = time.perf_counter()
        toks = int(usage.get("completion_tokens") or content_chunks) if usage else content_chunks
        details = (usage or {}).get("prompt_tokens_details") or {}
        return {
            "ok": True, "t_send": t_send, "t_first": first or t_end, "t_end": t_end,
            "completion_tokens": toks,
            "prompt_tokens": (usage or {}).get("prompt_tokens"),
            "cached_tokens": details.get("cached_tokens"),
            "chunks": content_chunks,
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "t_send": t_send, "t_end": time.perf_counter(),
                "error": repr(exc), "completion_tokens": 0}


# ---------------------------------------------------------------- 假发送器

def fake_stream(endpoint: str, model: str, prompt_text: str, max_tokens: int,
                api_key: str = "", **kwargs) -> dict:
    """CI / 冒烟用的假发送器：不碰网络，返回结构同 stream_once。"""
    t_send = time.perf_counter()
    time.sleep(0.002)
    t_end = time.perf_counter()
    return {
        "ok": True, "t_send": t_send, "t_first": t_end, "t_end": t_end,
        "completion_tokens": max_tokens, "prompt_tokens": 12,
        "cached_tokens": 0, "chunks": max_tokens,
    }


# ---------------------------------------------------------------- 单次窗口

def _accepts_priority(fn) -> bool:
    """transport 是否接受 priority 关键字。

    CI 的假发送器（tests/test_frozen_bench.py::fake_transport）只有 5 个位置
    参数，不能因新增字段而破坏它；这里按签名探测，只有真发送器
    （stream_once）或显式声明 **kwargs 的注入器才会收到 priority。
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if "priority" in params:
        return True
    return any(p.kind is p.VAR_KEYWORD for p in params.values())


def mix_prompt_sets(sets: list[dict], ratio: str, cycles: int = 4) -> list[dict]:
    """按 ratio 把多个 prompt 集交错合并（顺序对应 ratio 各段）。

    `--mix-ratio 8:1 --mix-cycles 4` ⇒ 32 短 + 4 长 = 36 条，长条目落在第
    9/18/27/36 位 —— 与手写混合集等价，但**不需要新建文件**（沙箱禁止脚本
    写工作区，2026-11-18 实测 PermissionError）。

    必须交错而非分块拼接：continuous worker 按 `i % len(prompts)` 轮转取，
    若长请求集中在尾部，`--concurrency 40` 下各 worker 的相位会让某些窗
    几乎碰不到长请求，实际比例失控。
    """
    if not ratio:
        if len(sets) != 1:
            raise SystemExit(f"给了 {len(sets)} 个 --prompt-set，必须同时给 --mix-ratio")
        return list(sets[0]["prompts"])
    try:
        parts = [int(x) for x in ratio.split(":")]
    except ValueError:
        raise SystemExit(f"--mix-ratio 需形如 9:1，实得 {ratio!r}")
    if any(n <= 0 for n in parts):
        raise SystemExit(f"--mix-ratio 各段必须为正整数，实得 {ratio!r}")
    if len(parts) != len(sets):
        raise SystemExit(f"--mix-ratio {ratio} 有 {len(parts)} 段，"
                         f"但给了 {len(sets)} 个 --prompt-set")
    out: list[dict] = []
    for _ in range(max(1, cycles)):
        for i, n in enumerate(parts):
            src = sets[i]["prompts"]
            if not src:
                raise SystemExit(f"第 {i} 个 prompt 集为空，无法交错")
            for k in range(n):
                out.append(src[k % len(src)])
    return out


def attach_prompt_tokens(prompts: list[dict], meta: dict) -> int:
    """把 meta 里实测的 prompt token 数注入每条 prompt，返回注入条数。

    🔴 两个冻结集的 meta 字段名不一致（2026-11-18 实测）：
      - `prompts-frozen-v1.meta.json`   → `prompt_tokens.per_id`（32/32 全覆盖）
      - `prompts-comm-7k-v1.meta.json`  → `tokens_measured.per_id`（仅 6/96 采样）
    只认前者会让 7K 集整组退回字符估算，故两处都取。无元数据的条目保持原样，
    `make_classifier` 会回退到 `len(text) * chars_per_token`。
    """
    meta = meta or {}
    # 不可用 or 短路：两个键可能同时存在（合并集就是这种情况），必须都合并。
    per_id: dict = {}
    for key in ("prompt_tokens", "tokens_measured"):
        per_id.update((meta.get(key) or {}).get("per_id") or {})
    hit = 0
    for p in prompts:
        v = per_id.get(p.get("id"))
        if isinstance(v, (int, float)) and v > 0:
            p["n_tokens"] = v
            hit += 1
    return hit


def make_classifier(args):
    """按 --priority-mode 生成 classify(prompt) -> (priority|None, group|None)。

    group 恒按长度判定（short / long），即使 priority 是固定值也要有，否则
    无法按组统计「短请求 TTFT 是否改善 / 长请求是否被饿死」。
    """
    mode = getattr(args, "priority_mode", "off")

    def n_tokens_of(prompt: dict) -> float:
        for key in ("n_tokens", "prompt_tokens", "tokens"):
            v = prompt.get(key)
            if isinstance(v, (int, float)) and v > 0:
                return float(v)
        # 回退：字符数 × 系数（无 token 元数据的语料才会走到这里）
        return len(prompt.get("text") or "") * args.chars_per_token

    threshold = float(getattr(args, "short_threshold", 4096.0))

    if mode == "off":
        def clf(prompt: dict):
            # 不传 priority（None），但**仍按长度标组**：否则对照组没有分组
            # 数据，无法回答「短请求 TTFT 是否改善 / 长请求是否被饿死」这两个
            # 核心问题。2026-11-18 首次 T-SRF-B 就踩了这个坑，只能事后用
            # prompt_id 前缀反推。
            return None, ("short" if n_tokens_of(prompt) < threshold else "long")
        return clf

    if mode == "fixed":
        fixed = int(getattr(args, "priority_fixed", 0))

        def clf(prompt: dict):
            return fixed, ("short" if n_tokens_of(prompt) < threshold else "long")
        return clf

    if mode == "by_prompt_length":
        sp = int(getattr(args, "short_priority", 0))
        lp = int(getattr(args, "long_priority", 100))

        def clf(prompt: dict):
            return (sp, "short") if n_tokens_of(prompt) < threshold else (lp, "long")
        return clf

    raise ValueError(f"unknown priority mode: {mode}")


def _ids_hash(ids: list) -> str:
    """对 id 序列取 sha256 前 16 位，用于判定两次运行的输入是否逐位一致。"""
    return hashlib.sha256("\n".join(str(x) for x in ids).encode()).hexdigest()[:16]


def _peak_concurrent(records: list[dict]) -> int:
    """在飞峰值：按 t_start 进、t_end_rel 出扫描事件。

    openloop 下这个值**不能**用 `concurrency` 顶替 —— 实际并发由
    「到达速率 × 服务时长」涌现；2026-11-18 的第一版就是把它写成
    concurrency，于是「客户端线程池把峰值钉在 80」这件事全程不可见。
    """
    events = []
    for r in records:
        st = r.get("t_start")
        en = r.get("t_end_rel")
        if st is None or en is None:
            continue
        events.append((st, 1))
        events.append((en, -1))
    events.sort()
    cur = peak = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


# 开环「是否真的形成等待队列」的下限：在飞越过服务端 max_num_seqs 的时间
# 至少要占窗口的 5%，否则只算瞬时抖动（EXP 600 ms 实测峰值 41 条、只持续一瞬）。
OPENLOOP_QUEUE_MIN_FRAC = 0.05


def _inflight_above_frac(records: list[dict], threshold: int) -> float:
    """在飞**严格大于** threshold 的时间占比（按事件区间加权）。

    为什么需要它：priority 只对**等待队列**排序，而等待队列只有在 running
    槽位（= 服务端 `max_num_seqs`，当前部署 32，2026-11-19 定格）被占满时
    才存在。若整窗
    in_flight <= 32，调度器从来没有两条以上候选可比，**按构造就测不出
    调度策略的差别**。2026-11-18 实测：600 ms 档占比 0%（峰值 37–41）、
    200 ms 档占比 90%（但那是客户端线程池顶在 80，不是服务端排队）。
    """
    events = []
    for r in records:
        st = r.get("t_start")
        en = r.get("t_end_rel")
        if st is None or en is None:
            continue
        events.append((st, 1))
        events.append((en, -1))
    events.sort()
    if len(events) < 2:
        return 0.0
    cur = 0
    above_s = total_s = 0.0
    prev_t = events[0][0]
    for t, delta in events:
        dt = t - prev_t
        if dt > 0:
            total_s += dt
            if cur > threshold:
                above_s += dt
        prev_t = t
        cur += delta
    return (above_s / total_s) if total_s > 0 else 0.0


def window_validity(win: dict, summary: dict, *, mode: str, window_s: float,
                    interval_ms: float | None, max_lag_ms: float,
                    server_seqs: int) -> dict:
    """单窗可用性判定 —— 跑中门禁与跑后 `data_valid` 共用同一套判据。

    三条判据，各有明确的反例（都是 2026-11-18 实测过的失败形态）：

    ① `arrival`：开环下计划到达必须被遵守 —— `lag p95 <= max_lag_ms`，且窗内
       **按时发出**的条数 == 计划条数。反例：200 ms 档 lag p95 = 20.2 s、
       窗内实发 185/300，而 `stability.pass` 仍是 true（MAD 只看吞吐离散度）。
    ② `queue`：是否真的形成等待队列（见 `_inflight_above_frac`）。反例：
       600 ms 档全程 <=40 ⇒ 无队列，priority 无处可排。
    ③ `drain`：窗末未完成条数占计划条数的比例。**只报告不否决** —— 固定速率
       稳态下它 ≈ e2e/窗口（600 ms 档实测 0.31–0.35、五窗几乎不变），本身
       不是越载证据；跨窗增长才是。
    """
    planned_n = int(round(window_s * 1000.0 / interval_ms)) if interval_ms else None
    lag = (win.get("arrival_lag_ms") or {}).get("p95")
    arrival_ok = True
    reasons: list[str] = []
    if mode == "openloop":
        if lag is not None and max_lag_ms > 0 and lag > max_lag_ms:
            arrival_ok = False
            reasons.append(f"arrival_lag_p95={lag}ms > {max_lag_ms}ms")
        if planned_n is not None and win["arrival_n"] < planned_n:
            arrival_ok = False
            reasons.append(f"窗内按时发出 {win['arrival_n']}/{planned_n}"
                           "（客户端排队，到达轨迹不完整）")
    frac = _inflight_above_frac(win["records"], server_seqs)
    # 阈值不能是 `frac > 0`：EXP 600 ms 实测峰值 41，只在一瞬越过 40，
    # frac ≈ 0.001，若按 >0 判定就会把「按构造空测」判成「排过队」。
    # 5% 窗口（60 s 里 3 s）才算真的形成过队列。
    queue_formed = frac >= OPENLOOP_QUEUE_MIN_FRAC
    if not queue_formed:
        reasons.append(f"inflight 超过 {server_seqs} 的时间只占 {frac:.1%}"
                       f"（< {OPENLOOP_QUEUE_MIN_FRAC:.0%}，无等待队列，"
                       "priority 无处可排）")
    drained = summary.get("drained", 0)
    return {
        "planned_n": planned_n,
        "arrival_n": win["arrival_n"],
        "lag_p95_ms": lag,
        # ⚠️ 两个有效性必须分开：arrival = 客户端是否照计划发出；
        # queue_exposure = 服务端是否出现 priority 可以发挥作用的竞争场景。
        # 600 ms EXP 就是「前者过、后者不过」。
        "arrival_valid": arrival_ok,
        "queue_exposure_valid": queue_formed,
        "inflight_above_server_frac": round(frac, 3),
        "inflight_peak": win["peak_inflight"],
        "drain_frac": round(drained / planned_n, 3) if planned_n else None,
        "drained": drained,
        "reasons": reasons,
        # 单窗层面只能判到达有效性；队列暴露与跨组可比性归 run 级判据。
        "ok": arrival_ok,
    }


def backlog_growth_frac(prev_drained: list[int], now_drained: int) -> float | None:
    """窗末未完成条数相对**本 run 前序最小值的涨幅**（越载的增长信号）。

    为什么要它：固定速率稳态下 `drained` 应≈常数（600 ms 档 31–35）；一旦
    每窗持续变大，说明积压随时窗增长、窗与窗不再独立 —— 这类 run 里
    CTLR/EXP 各自继承的积压不同，就算 p95 有差也解释不了。
    阈值 40% 远高于窗间离散度（MAD 1–3%，最大极差 15%）。
    """
    if not prev_drained:
        return None
    base = min(prev_drained)
    if base <= 0:
        return None
    return round((now_drained - base) / base, 3)


# ---------------------------------------------------------------- 脉冲排空核算（T-SRF-D）
# 用户 2026-11-18 要求：提交 / 完成 / HTTP 失败 / 超时 / 未提交**分别记录**，
# 未完成样本不得从分位数里静默删除；并新增「长请求提交→首 token」「提交→完成」
# 的最长墙钟时间（TTFT max / E2E max 不足以说明是否被持续推迟）。

def _is_timeout(error) -> bool:
    if not error:
        return False
    e = str(error).lower()
    return "timeout" in e or "timed out" in e


def analyze_drain(records: list[dict], *, inject_s: float, drain_max_s: float,
                  planned_n: int | None = None) -> dict:
    """一轮脉冲的排空核算（纯函数，可单测）。

    `completion_rate` 按用户定义：**规定排空期限内成功完成 / 成功提交**。
    未提交（计划里有、实际没发出去）与未完成（发了但没在期限内完成）分开计。
    """
    submitted = list(records)
    ok = [r for r in submitted if r.get("ok")]
    failed = [r for r in submitted if not r.get("ok")]
    timeouts = [r for r in failed if _is_timeout(r.get("error"))]
    http_fail = [r for r in failed if not _is_timeout(r.get("error"))]
    deadline = inject_s + drain_max_s
    completed = [r for r in ok if (r.get("t_end_rel") or 0.0) <= deadline]
    late = [r for r in ok if (r.get("t_end_rel") or 0.0) > deadline]
    ends = [r.get("t_end_rel") or 0.0 for r in submitted]

    def _wait_ft(group: str) -> float | None:
        vals = [(r["t_first_rel"] - r["t_start"]) * 1000.0
                for r in ok
                if r.get("prompt_group") == group
                and r.get("t_first_rel") is not None]
        return round(max(vals), 1) if vals else None

    def _wall(group: str) -> float | None:
        vals = [(r["t_end_rel"] - r["t_start"]) * 1000.0
                for r in ok if r.get("prompt_group") == group]
        return round(max(vals), 1) if vals else None

    rate = round(len(completed) / len(submitted), 4) if submitted else None
    return {
        "planned_n": planned_n,
        "submitted_n": len(submitted),
        "completed_n": len(completed),
        "completed_late_n": len(late),
        "http_failed_n": len(http_fail),
        "timed_out_n": len(timeouts),
        "unsubmitted_n": (planned_n - len(submitted)
                          if planned_n is not None else None),
        "completion_rate": rate,
        "drain_s": round(max(ends) - inject_s, 3) if ends else None,
        "drain_deadline_s": drain_max_s,
        "inject_s": inject_s,
        # 本轮已 join 所有线程 ⇒ 客户端侧遗留为 0；但超时被放弃的请求
        # 服务端可能仍在跑，必须单独标注，不能算「已排空」。
        "cross_round_n": 0,
        "server_may_carry_n": len(timeouts),
        "long_submitted_n": sum(1 for r in submitted
                                if r.get("prompt_group") == "long"),
        "long_completed_n": sum(1 for r in ok
                                if r.get("prompt_group") == "long"),
        "long_wait_to_first_token_max_ms": _wait_ft("long"),
        "long_wall_max_ms": _wall("long"),
        "short_wait_to_first_token_max_ms": _wait_ft("short"),
        "short_wall_max_ms": _wall("short"),
        "ok": (not timeouts and not late
               and rate is not None and rate >= 1.0),
    }


def quiet_check(*, inflight_now: int, pending_tasks: int,
                probe_ttft_ms: float | None = None,
                baseline_ttft_ms: float | None = None,
                drift_max_frac: float = 0.10,
                drift_abs_tol_ms: float = 200.0,
                server_waiting: int | None = None,
                hbm_gb: float | None = None) -> dict:
    """静默段验证（不是「等 60 s」）：客户端为主，服务端指标缺证据时明确标注。

    thread-join 成功 ≠ 请求完成 ≠ 已排空 —— 三个概念分开，不合并成一个布尔。
    探针漂移：**相对 10% 或绝对 200 ms 任一满足即算恢复**。绝对容差是必要的：
    空载 TTFT ≈ 500–600 ms，单次探针的冷启动/抖动就能到 ±15%，只靠相对阈值
    会把健康状态误判成「未恢复」（2026-11-18 冒烟实测：584.8 → 497.0 被判失败）。
    因此探针取多次的**最小值**（最接近无竞争下限），再叠加绝对容差。
    """
    checks = {
        "client_inflight_zero": inflight_now == 0,
        "client_pending_zero": pending_tasks == 0,
        # 只有真观测到 0 才算通过；None（取不到）记 False 并在 reasons 说明
        "server_waiting_zero": server_waiting == 0,
    }
    drift = None
    delta = None
    if probe_ttft_ms is not None and baseline_ttft_ms:
        drift = round((probe_ttft_ms - baseline_ttft_ms) / baseline_ttft_ms, 4)
        delta = round(probe_ttft_ms - baseline_ttft_ms, 1)
        checks["probe_ttft_drift_ok"] = (abs(drift) <= drift_max_frac
                                         or abs(delta) <= drift_abs_tol_ms)
    reasons = [k for k, v in checks.items() if not v]
    client_ok = (checks["client_inflight_zero"]
                 and checks["client_pending_zero"]
                 and checks.get("probe_ttft_drift_ok", True))
    return {
        "checks": checks,
        "client_ok": client_ok,
        "strict_ok": client_ok and checks["server_waiting_zero"],
        "evidence": ("full" if server_waiting is not None and hbm_gb is not None
                     else "client_only"),
        "probe_ttft_ms": probe_ttft_ms,
        "baseline_ttft_ms": baseline_ttft_ms,
        "probe_ttft_drift_frac": drift,
        "probe_ttft_delta_ms": delta,
        "drift_abs_tol_ms": drift_abs_tol_ms,
        "server_waiting": server_waiting,
        "hbm_gb": hbm_gb,
        "reasons": reasons,
        "note": ("server_waiting/HBM 客户端取不到，需容器内取证；"
                 "None 视为未取证，不计为通过"),
    }


def run_validity(repeats: list[dict], agg: dict, *, priority_mode: str,
                 max_lag_ms: float, stopped_early: dict | None = None) -> dict:
    """整轮可用性 —— 四项分开，**不得用单个布尔值代替**。

    用户 2026-11-18 要求拆开：600 ms EXP 就是「到达有效、处理有效」两条结论
    相反，一个 `data_valid` 必然误判其中一条。

      arrival_valid        客户端是否按计划发出（lag + 窗内按时条数）
      queue_exposure_valid 服务端是否出现过 priority 可发挥作用的竞争（代理指标，
                           客户端 in_flight 计数 ≠ 服务端 waiting，只是代理）
      comparison_valid     本 run 内部是否可比（plan hash 窗间一致、未提前中止）
                           ⚠️ **跨组可比性（两组 plan hash 相等）必须在对比时另判**，
                           本函数只能保证「本 run 自己没漂」
      overall_valid        以上三条成立；priority 组额外要求 queue_exposure_valid
    """
    reasons: list[str] = []
    windows = [r.get("validity") or {} for r in repeats]
    bad = [i + 1 for i, w in enumerate(windows) if not w.get("arrival_valid", True)]
    arrival_valid = not bad
    if bad:
        reasons.append(f"[arrival] 窗 {bad} 到达轨迹未达标")

    plans = {r.get("arrival_plan_hash_v2") or r.get("arrival_plan_hash")
             for r in repeats
             if r.get("arrival_plan_hash_v2") or r.get("arrival_plan_hash")}
    comparison_valid = len(plans) <= 1 and not stopped_early
    if len(plans) > 1:
        reasons.append(f"[comparison] plan_hash 窗间不一致：{sorted(plans)}")
    if stopped_early:
        reasons.append(f"[comparison] 提前中止于第 {stopped_early.get('repeat')} 窗"
                       f"（{stopped_early.get('reason')}）")

    fracs = [w.get("inflight_above_server_frac") or 0.0 for w in windows]
    queue_exposure_valid = any(w.get("queue_exposure_valid") for w in windows)
    if priority_mode != "off" and windows and not queue_exposure_valid:
        reasons.append(f"[queue] 未出现足以检验 priority 的等待竞争（在飞超阈值"
                       f"时间占比最大 {max(fracs):.1%} < "
                       f"{OPENLOOP_QUEUE_MIN_FRAC:.0%}）")

    stab = agg.get("stability") or {}
    if stab and not stab.get("pass", True):
        reasons.append(f"[comparison] 窗间不稳定 MAD%={stab.get('MAD_pct')}")

    treatment_ok = queue_exposure_valid or priority_mode == "off"
    overall_valid = bool(arrival_valid and comparison_valid and treatment_ok)
    return {
        "arrival_valid": arrival_valid,
        "queue_exposure_valid": queue_exposure_valid,
        "comparison_valid": bool(comparison_valid),
        "overall_valid": overall_valid,
        "ok": overall_valid,          # 兼容：旧脚本读 `ok`
        "reasons": reasons,
        "checks": {
            "windows": len(repeats),
            "plan_hashes": sorted(plans),
            "max_lag_ms": max_lag_ms,
            "priority_mode": priority_mode,
            "queue_formed_windows": sum(1 for w in windows
                                        if w.get("queue_exposure_valid")),
            "queue_above_frac_max": round(max(fracs), 3) if fracs else None,
            "queue_min_frac": OPENLOOP_QUEUE_MIN_FRAC,
            "queue_metric_note": "客户端 in_flight 只是服务端 waiting 的代理；"
                                 "严格证据需容器内 vllm:num_requests_waiting",
            # 跨组可比性不在本 run 内可判，必须在对比表里逐条确认
            "cross_run_plan_hash_equality": "must_check_at_comparison",
        },
    }


def run_window(mode: str, prompts: list[dict], endpoint: str, model: str,
               api_key: str, concurrency: int, max_tokens: int,
               window_s: float, transport=None, classify=None,
               openloop_interval_ms: float = 200.0) -> dict:
    """跑一个计时窗口，返回原始记录、批次边界与时间线元数据。

    transport 可注入（CI 用假发送器，不碰网络）。
    classify 可注入：classify(prompt) -> (priority|None, group|None)；
      为 None 时等价于 --priority-mode off（不传 priority 字段）。
    """
    send = transport or stream_once
    send_takes_priority = _accepts_priority(send)
    spec = PROTOCOL["modes"][mode]
    t0 = time.perf_counter()
    deadline = t0 + window_s
    records: list[dict] = []
    lock = threading.Lock()
    inflight = {"now": 0, "peak": 0}
    # prompt 内容指纹：计划 hash v1 只含 id/chars/计划时刻，理论上两份**同 id 同长度但内容不同**
    # 的 prompt 会算出同一个 hash（用户 2026-11-18 追问「hash 覆盖的是完整计划吗」）。
    # v2 把内容指纹并进去，v1 保留以便与历史 run 仍可对表。
    prompt_sha = {p["id"]: hashlib.sha256(
        (p.get("text") or "").encode("utf-8")).hexdigest()[:16] for p in prompts}

    def timed(prompt: dict, sched_rel: float | None = None) -> dict:
        prio, group = classify(prompt) if classify else (None, None)
        if send_takes_priority:
            rec = send(endpoint, model, prompt["text"], max_tokens, api_key,
                       priority=prio)
        else:
            rec = send(endpoint, model, prompt["text"], max_tokens, api_key)
        rec["prompt_id"] = prompt["id"]
        rec["priority"] = prio
        rec["prompt_group"] = group
        rec["prompt_chars"] = len(prompt.get("text") or "")
        rec["prompt_sha"] = prompt_sha.get(prompt["id"])
        # 计划到达时刻（相对 t0）。闭环模式下为 None：此时到达轨迹由完成时刻决定。
        rec["t_sched_rel"] = sched_rel
        return rec

    if mode == "continuous":
        def worker(idx: int) -> None:
            i = idx
            local = []
            while time.perf_counter() < deadline:
                p = prompts[i % len(prompts)]
                rec = timed(p)
                local.append(rec)
                i += 1
            with lock:
                records.extend(local)

        threads = [threading.Thread(target=worker, args=(i,), daemon=True)
                   for i in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        batches = None
    elif mode == "barrier":
        batches = []
        offset = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            while time.perf_counter() < deadline:
                batch = prompt_cycle(prompts, concurrency, offset)
                b_start = time.perf_counter()
                futs = [pool.submit(timed, p) for p in batch]
                got = [f.result() for f in futs]
                b_end = time.perf_counter()
                for p, rec in zip(batch, got):
                    rec["prompt_id"] = p["id"]
                records.extend(got)
                inflight["peak"] = max(inflight["peak"], concurrency)
                batches.append({
                    "index": len(batches),
                    "t_start": round(b_start - t0, 4),
                    "t_end": round(b_end - t0, 4),
                    "n": len(got),
                    "n_ok": sum(1 for r in got if r["ok"]),
                    "dispatch_spread_s": round(
                        max(r["t_send"] for r in got) - min(r["t_send"] for r in got), 4),
                })
                offset += concurrency
    elif mode == "openloop":
        # 开环：按 k*interval 固定时间表发请求，**不看前一条是否返回**。
        # 这样到达轨迹与完成时刻解耦，两组可逐条对齐；闭环(continuous)下
        # 短请求先完成会让对应 worker 更快发下一条，短请求占比自我放大。
        interval_s = max(float(openloop_interval_ms), 1.0) / 1000.0
        # 池子必须容得下**整个窗口**的计划条数，而不是「并发的 2 倍」。
        # 2026-11-18 实测：interval=200ms 时 max(concurrency*2,32)=80，窗口内
        # 计划 300 条 ⇒ 同时在飞峰值被钉死在 80，第 81 条起在**客户端线程池**
        # 排队，实际 t_send 由客户端排队决定：lag p95 = 42.7 s、t_start 最大
        # 108.2 s（窗口只有 60 s），固定的到达轨迹已经完全失效。
        # 池子给足后，滞后只剩调度线程 sleep 精度（实测 p50 ≈ 1 ms）。
        planned_max = int(window_s / interval_s) + 2
        max_workers = max(concurrency * 2, planned_max)
        futs = []
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            k = 0
            while True:
                target = t0 + k * interval_s
                now = time.perf_counter()
                if target > now:
                    time.sleep(target - now)
                if time.perf_counter() >= deadline:
                    break
                futs.append(pool.submit(timed, prompts[k % len(prompts)],
                                        k * interval_s))
                k += 1
            records.extend(f.result() for f in futs)
        batches = None
    else:
        raise ValueError(f"unknown mode: {mode}")

    t_join = time.perf_counter()
    for rec in records:
        rec["t_start"] = round(rec["t_send"] - t0, 4)
        rec["t_end_rel"] = round(rec["t_end"] - t0, 4)
        # 失败请求走 stream_once 的 except 分支，没有 t_first；此处不得无条件取。
        if rec["ok"]:
            rec["t_first_rel"] = round(rec["t_first"] - t0, 4)
            rec["ttft_s"] = round(rec["t_first"] - rec["t_send"], 4)
            rec["e2e_s"] = round(rec["t_end"] - rec["t_send"], 4)
        else:
            rec["t_first_rel"] = None
            rec["ttft_s"] = None
            rec["e2e_s"] = None
        if rec["ok"]:
            gen = max(rec["t_end"] - rec["t_first"], 1e-9)
            rec["tpot_s"] = round(gen / rec["completion_tokens"], 4) \
                if rec["completion_tokens"] else None

    arrive = sorted((x for x in records if x.get("t_start", 0.0) <= window_s),
                    key=lambda r: r["t_start"])

    def _pct(xs, q):
        if not xs:
            return 0.0
        ys = sorted(xs)
        kk = (len(ys) - 1) * q / 100.0
        lo = int(kk)
        hi = min(lo + 1, len(ys) - 1)
        return ys[lo] + (ys[hi] - ys[lo]) * (kk - lo)

    # 计划轨迹：按「计划时刻」排序，只含与调度策略无关的字段（id/字符数/计划时刻）。
    # 开环下两组必须完全相同 ⇒ 才能说"两组到达轨迹一致"；闭环无计划时刻 ⇒ None。
    sched = [r for r in arrive if r.get("t_sched_rel") is not None]
    plan_hash = None
    plan_hash_v2 = None
    priority_plan_hash = None
    lag_ms = None
    if sched:
        ordered = sorted(sched, key=lambda r: r["t_sched_rel"])
        plan_hash = _ids_hash([
            "%s|%s|%s" % (r.get("prompt_id"), r.get("prompt_chars"),
                          r["t_sched_rel"])
            for r in ordered])
        # v2 = 完整计划（含 prompt 内容指纹）；两组**必须**相等才有可比性
        plan_hash_v2 = _ids_hash([
            "%s|%s|%s|%s" % (r.get("prompt_id"), r.get("prompt_sha"),
                             r.get("prompt_chars"), r["t_sched_rel"])
            for r in ordered])
        # 处理变量本身（priority 分配）单独成 hash：两组**允许**不同，用于记录「发了什么」
        priority_plan_hash = _ids_hash([
            "%s|%s" % (r.get("prompt_id"), r.get("priority")) for r in ordered])
        lags = [(r["t_start"] - r["t_sched_rel"]) * 1000.0 for r in sched]
        lag_ms = {"p50": round(_pct(lags, 50), 1),
                  "p95": round(_pct(lags, 95), 1),
                  "max": round(max(lags), 1)}

    return {
        "records": records,
        "batches": batches,
        "t0": t0,
        "wall_s": round(t_join - t0, 3),
        "peak_inflight": (inflight["peak"] if mode == "barrier"
                          else _peak_concurrent(records)),
        # 客户端线程池上限：openloop 下若 peak_inflight 恰好等于该值，
        # 说明「客户端排队」而非「服务端排队」决定了实际到达时刻。
        "openloop_max_workers": max_workers if mode == "openloop" else None,
        "mode_spec": spec,
        # 计划到达轨迹 hash：开环两组相等 ⇒ 到达轨迹一致（用户 2026-11-18 要求）
        "arrival_plan_hash": plan_hash,
        # v2：含 prompt 内容指纹的完整计划 hash（跨组可比性的**唯一**依据）
        "arrival_plan_hash_v2": plan_hash_v2,
        # 处理变量指纹：两组允许不同，记录 priority 分配
        "priority_plan_hash": priority_plan_hash,
        # 实际到达（按 t_start 排序）的 id 序列 hash：闭环下这是唯一可比项
        "arrival_ids_hash": _ids_hash([r.get("prompt_id") for r in arrive]),
        "arrival_n": len(arrive),
        # 实际提交时刻相对计划时刻的偏差 ⇒ 调度线程是否准时（开环专有）
        "arrival_lag_ms": lag_ms,
    }


# ---------------------------------------------------------------- 汇总

def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[k], 4)


def stability_stats(vals: list[float]) -> dict:
    """窗间 / 重复间稳定性：主判据 MAD/中位<10%，|x−中位|>3×MAD 记孤点。

    口径来源（2026-11-16 20:15，`python -m llmops.reanalyze` 对 18 个 run /
    全部 raw 样本重算；本函数是**唯一**实现，reanalyze 也调用它）：
      R=(max−min)/中位<10%  达标  2/18（中位 19.06%）
      stdev/中位<10%        达标 12/18（中位 6.79%）
      IQR/中位<10%          达标 11/18（中位 7.93%）
      MAD/中位<10%          达标 17/18（中位 4.47%）
    为什么不用 R：R 是极差统计量，单个孤点即可让整组判「不可复现」，把真实状态切换
    与测量噪声混成一个数（POSTSTOP 反例：R=15.33% 但 IQR=1.91%、MAD=1.20%）。
    MAD 对孤点稳健，但**不等于**引擎稳定：MAD 17/18 说明典型抖动 4.47%。
    分位数按索引取（与 reanalyze 历史口径一致），窗数<4 时 IQR 退化为 (max−min)/中位。
    """
    spec = PROTOCOL.get("stability_rule_v3") or {}
    threshold = float(spec.get("threshold_pct", 10.0))
    k = float(spec.get("outlier_k", 3.0))
    rule = (f"主判据 MAD/中位<{threshold:g}%；|x-median|>{k:g}*MAD 记孤点必须归因；"
            f"R 仅供参考不作判据")
    n = len(vals)
    if not vals:
        return {"median": 0.0, "MAD_pct": 0.0, "IQR_pct": 0.0, "stdev_pct": 0.0,
                "R_pct": 0.0, "outliers": [], "rule": rule, "pass": False, "n": 0}

    med = statistics.median(vals)
    s = sorted(vals)
    if n >= 4:
        q1 = s[int(round(0.25 * (n - 1)))]
        q3 = s[int(round(0.75 * (n - 1)))]
        iqr = q3 - q1
    else:
        iqr = s[-1] - s[0]
        rule += "；窗数<4，IQR 退化=(max-min)/median"
    mad = statistics.median([abs(x - med) for x in s])
    mad_pct = (mad / med * 100) if med else 0.0
    iqr_pct = (iqr / med * 100) if med else 0.0
    stdev_pct = (statistics.stdev(vals) / med * 100) if (n >= 2 and med) else 0.0
    r_pct = ((s[-1] - s[0]) / med * 100) if med else 0.0

    outliers = []
    for i, v in enumerate(vals):
        dev = abs(v - med)
        # MAD=0（多数窗完全相同）时任何偏离都算孤点，但无法给倍数
        if (dev > k * mad) if mad > 0 else (dev > 0):
            outliers.append({"step": i + 1, "value": round(v, 2),
                             "dev_MAD": round(dev / mad, 2) if mad > 0 else None})
    return {"median": round(med, 2), "MAD_pct": round(mad_pct, 2),
            "IQR_pct": round(iqr_pct, 2), "stdev_pct": round(stdev_pct, 2),
            "R_pct": round(r_pct, 2), "outliers": outliers, "rule": rule,
            "pass": bool(mad_pct < threshold), "n": n}


def stability_line(st: dict) -> str:
    """一行汇总，形如：
    stability: median=167.93 MAD=1.20% IQR=1.91% R=15.33% outliers=1 -> PASS"""
    return (f"stability: median={st['median']:.2f} MAD={st['MAD_pct']:.2f}% "
            f"IQR={st['IQR_pct']:.2f}% R={st['R_pct']:.2f}% "
            f"outliers={len(st['outliers'])} -> {'PASS' if st['pass'] else 'FAIL'}")


def summarize_groups(records: list[dict], window_s: float) -> dict:
    """按 prompt_group（short / long）拆开统计，供 T-SRF-B 逐组判据使用。

    用户 2026-11-18 要求「必须记录长请求的等待时间 p50/p95/最大值，以及是否
    出现长时间未完成」——本函数给这两项：
      - ttft_ms.max = 该组最坏一次 TTFT（= 队列等待 + prefill，客户端侧无法分离）
      - drained     = 窗末仍未返回、跨窗排空的条数（long 组 >0 即「被饿住」的硬证据）
    只统计 t_end_rel <= window_s 的成功请求，与 summarize_window 同口径；
    不重复给吞吐，因为混合负载下分组吞吐没有可与单一长度负载对比的意义。
    """
    in_win = [r for r in records if r["ok"] and r["t_end_rel"] <= window_s]
    out: dict = {}
    for group in ("short", "long"):
        # 组的存在性按「该组的全部记录」判断，不按「窗内完成数」：某组若一个都
        # 没在窗内完成（全被饿住），按完成数判断会让整个组从结果里消失，恰好
        # 丢掉最需要看的那一格。
        if not any(r.get("prompt_group") == group for r in records):
            continue
        sel = [r for r in in_win if r.get("prompt_group") == group]
        # 窗末仍在飞的该组请求（已提交、未完成）。drained_wait_ms 记录它们
        # 已等多久 —— 光看条数持平无法区分「仍在排队 / 已进 prefill /
        # 已出首 token 只差 decode」这三种状态。
        dr = [r for r in records
              if r["ok"] and r.get("prompt_group") == group
              and r["t_start"] <= window_s and r["t_end_rel"] > window_s]
        wait_ms = [(window_s - r["t_start"]) * 1000.0 for r in dr]
        inflight_n = sum(1 for r in records
                         if r.get("prompt_group") == group
                         and r["t_start"] <= window_s and r["t_end_rel"] > window_s)
        ttfts = [r["ttft_s"] * 1000 for r in sel]
        e2es = [r["e2e_s"] * 1000 for r in sel]
        tpots = [r["tpot_s"] * 1000 for r in sel if r.get("tpot_s")]
        pts = [r.get("prompt_tokens") or 0 for r in sel]
        out[group] = {
            "n": len(sel),
            "priorities": sorted({r.get("priority") for r in sel
                                  if r.get("priority") is not None}),
            "prompt_tokens_p50": pct(pts, 50),
            "ttft_ms": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95),
                        "max": round(max(ttfts), 2) if ttfts else 0.0},
            "e2e_ms": {"p50": pct(e2es, 50), "p95": pct(e2es, 95),
                       "max": round(max(e2es), 2) if e2es else 0.0},
            "tpot_ms": {"p50": pct(tpots, 50), "p95": pct(tpots, 95)},
            # 🔴 口径（用户 2026-11-18 指出）：drained 必须是「窗口结束前已提交、
            # 但窗末仍未完成」，因此必须带 `t_start <= window_s`。不加这一条时，
            # continuous worker 的 `while perf_counter() < deadline` 与真正 t_send
            # 之间存在几微秒间隙，会把「窗口后才发出的请求」误算成被饿住。
            "drained": len(dr),
            # ⚠️ 条数持平 ≠ 等待未恶化。补两项以区分三种状态：
            #   drained_wait_ms            = 这些请求到窗末已经等了多久
            #   drained_with_first_token   = 其中已出首 token 的条数
            #     → 0 条出首 token ⇒ 都还堵在排队/预填；有出首 token 的 ⇒ 只差 decode
            "drained_wait_ms": {
                "p50": round(pct(wait_ms, 50), 1),
                "max": round(max(wait_ms), 1) if wait_ms else 0.0,
            },
            "drained_with_first_token": sum(
                1 for r in dr
                if r.get("t_first_rel") is not None
                and r["t_first_rel"] <= window_s),
            # 同一口径但**含失败请求**：失败的在飞请求若因超时迟迟不返回，
            # 不该被 ok 过滤掉。
            "inflight_at_window_end": inflight_n,
        }
    if out:
        out["_note"] = ("drained>0 = 该组有请求跨窗未返回（被饿住或超窗）；"
                        "TTFT 含排队+prefill，客户端侧无法分离")
    return out


def summarize_window(records: list[dict], window_s: float,
                     steady_skip: float = 0.0) -> dict:
    """按 §2.1 边界算吞吐与延迟分位数；另给 v3 稳态子窗口。

    稳态子窗口口径（v3）：
    - 只取**结束时刻落在 (steady_skip, window_s]** 的成功请求；
    - 分母 = window_s - steady_skip（不是 60 s）；
    - 已知偏差：仍受「排空丢弃」影响；跨 skip 边界的请求会把**全部** token 计入本段
      ⇒ 绝对值偏乐观，**仅用于比较窗间稳定性 R，不与全窗吞吐比大小**。
    """
    in_win = [r for r in records if r["ok"] and r["t_end_rel"] <= window_s]
    drained = [r for r in records if r["ok"] and r["t_end_rel"] > window_s]
    failed = [r for r in records if not r["ok"]]
    toks = sum(r["completion_tokens"] for r in in_win)
    ttfts = [r["ttft_s"] * 1000 for r in in_win]
    e2es = [r["e2e_s"] * 1000 for r in in_win]
    tpots = [r["tpot_s"] * 1000 for r in in_win if r.get("tpot_s")]

    steady = None
    if 0.0 < steady_skip < window_s:
        span = window_s - steady_skip
        sel = [r for r in in_win if r["t_end_rel"] > steady_skip]
        s_toks = sum(r["completion_tokens"] for r in sel)
        s_tpot = [r["tpot_s"] * 1000 for r in sel if r.get("tpot_s")]
        s_ttft = [r["ttft_s"] * 1000 for r in sel]
        steady = {
            "span_s": round(span, 2), "n": len(sel),
            "throughput_tok_s": round(s_toks / span, 2) if span else 0.0,
            "tpot_ms": {"p50": pct(s_tpot, 50), "p95": pct(s_tpot, 95)},
            "ttft_ms": {"p50": pct(s_ttft, 50), "p95": pct(s_ttft, 95)},
            "completion_tokens": s_toks,
        }

    return {
        "n": len(in_win), "ok": len(in_win),
        "error_rate": round(len(failed) / len(records), 4) if records else 0.0,
        "wall_s": window_s,
        "throughput_tok_s": round(toks / window_s, 2) if window_s else 0.0,
        "req_s": round(len(in_win) / window_s, 2) if window_s else 0.0,
        "ttft_ms": {"p50": pct(ttfts, 50), "p95": pct(ttfts, 95)},
        "tpot_ms": {"p50": pct(tpots, 50), "p95": pct(tpots, 95)},
        "e2e_ms": {"p50": pct(e2es, 50), "p95": pct(e2es, 95)},
        "completion_tokens": toks,
        "issued": len(records),
        "drained": len(drained),
        "failed": len(failed),
        "cached_tokens": sum(int(r.get("cached_tokens") or 0) for r in in_win),
        "groups": summarize_groups(records, window_s),
        "steady": steady,
    }


def interval_stats(records: list[dict], mode: str, batches: list | None) -> dict:
    """请求间隔与批次边界统计。"""
    if mode == "barrier" and batches:
        gaps = [round(batches[i]["t_start"] - batches[i - 1]["t_end"], 4)
                for i in range(1, len(batches))]
        spreads = [b["dispatch_spread_s"] for b in batches]
        return {
            "kind": "barrier", "batches": len(batches),
            "batch_size": PROTOCOL["modes"]["barrier"]["batch_size"],
            "inter_batch_gap_s": {"p50": pct(gaps, 50), "p95": pct(gaps, 95),
                                  "min": min(gaps) if gaps else None,
                                  "max": max(gaps) if gaps else None},
            "intra_batch_dispatch_spread_s": {"p50": pct(spreads, 50),
                                              "p95": pct(spreads, 95)},
        }
    if mode == "openloop":
        ok = sorted([r for r in records if r["ok"]], key=lambda r: r["t_start"])
        lags = [(r["t_start"] - r["t_sched_rel"]) * 1000.0 for r in ok
                if r.get("t_sched_rel") is not None]
        return {
            "kind": "openloop", "requests": len(ok),
            "dispatch": "按固定时间表发，不看前一条是否返回",
            "arrival_lag_ms": {"p50": round(pct(lags, 50), 1),
                               "p95": round(pct(lags, 95), 1),
                               "max": round(max(lags), 1) if lags else None},
            "note": "lag = 实际 t_send - 计划 k*interval；lag 大说明客户端"
                    "发不出去（线程池满或网络排队），到达轨迹已被污染。",
        }
    ok = sorted([r for r in records if r["ok"]], key=lambda r: r["t_start"])
    starts = [r["t_start"] for r in ok]
    gaps = [round(starts[i] - starts[i - 1], 4) for i in range(1, len(starts))]
    return {
        "kind": "continuous", "requests": len(ok),
        "worker_gap_s": "worker 一结束立即发下一条，间隔由服务返回时间决定",
        "global_start_gap_s": {"p50": pct(gaps, 50), "p95": pct(gaps, 95)},
        "note": "global_start_gap 是 32 个 worker 全局起发间隔，不是单 worker 间隔；"
                "单 worker 间隔 ≈ 单请求 e2e。",
    }


def aggregate(repeats: list[dict]) -> dict:
    """5 次重复的吞吐 / 请求数 / TTFT / E2E 中位数与极差。"""
    def series(fn):
        return [fn(r) for r in repeats]

    thr = series(lambda r: r["summary"]["throughput_tok_s"])
    req = series(lambda r: r["summary"]["n"])
    ttft50 = series(lambda r: r["summary"]["ttft_ms"]["p50"])
    ttft95 = series(lambda r: r["summary"]["ttft_ms"]["p95"])
    e2e50 = series(lambda r: r["summary"]["e2e_ms"]["p50"])
    e2e95 = series(lambda r: r["summary"]["e2e_ms"]["p95"])
    tpot50 = series(lambda r: r["summary"]["tpot_ms"]["p50"])
    fail = series(lambda r: r["summary"]["failed"])
    # v3：稳态子窗口序列（老数据无此字段时为 None，过滤掉）
    steady = [r["summary"].get("steady") for r in repeats]
    steady = [s for s in steady if s]
    steady_thr = [s["throughput_tok_s"] for s in steady]
    steady_tpot = [s["tpot_ms"]["p50"] for s in steady]

    def stats(vals: list[float]) -> dict:
        med = statistics.median(vals)
        return {
            "values": vals, "median": round(med, 2),
            "min": round(min(vals), 2), "max": round(max(vals), 2),
            "range_pct": round((max(vals) - min(vals)) / med * 100, 2) if med else 0.0,
        }

    return {
        "repeats": len(repeats),
        "throughput_tok_s": stats(thr),
        "requests_in_window": stats(req),
        "ttft_ms_p50": stats(ttft50),
        "ttft_ms_p95": stats(ttft95),
        "e2e_ms_p50": stats(e2e50),
        "e2e_ms_p95": stats(e2e95),
        "tpot_ms_p50": stats(tpot50),
        "failed": stats(fail),
        "steady": ({"windows": len(steady),
                    "throughput_tok_s": stats(steady_thr),
                    "tpot_ms_p50": stats(steady_tpot)} if steady else None),
        # v3 判据：主判据 MAD/中位<10%，孤点必须单独归因（R 仅参考）
        "stability": stability_stats(thr),
        "stability_rule": "判据 v3：主判据 MAD/中位<10%（见 stability 块），"
                          "R=(max-min)/median 仅作参考不作判据，孤点必须单独归因；"
                          "「全窗」与「稳态子窗口」两套口径不得混比大小",
    }


# ---------------------------------------------------------------- 落盘

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_env(endpoint: str, model: str, api_key: str) -> dict:
    """探一次非流式请求，记录 system_fingerprint 与 usage 字段。"""
    payload = {"model": model, "messages": [{"role": "user", "content": "ping"}],
               "max_tokens": 4, "temperature": 0, "stream": False}
    req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", "Bearer " + api_key)
    try:
        obj = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        return {"system_fingerprint": obj.get("system_fingerprint"),
                "usage": obj.get("usage"), "ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": repr(exc)}


def main(argv: list[str] | None = None) -> int:
    # Windows 终端是 GBK：打印里一旦混进 ⇒ / ⚠ 这类非 GBK 字符，print 就抛
    # UnicodeEncodeError 把整轮实验打断（2026-11-18 实测复现）。这里只把
    # **错误策略**改成替换，不动编码本身（改 UTF-8 会让终端中文变乱码）。
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="FROZEN-v1 压测")
    ap.add_argument("--mode", choices=["continuous", "barrier", "openloop"],
                    required=True,
                    help="openloop = 开环固定到达轨迹（每 interval 一条，"
                         "不看前一条是否返回）；两组到达轨迹可逐条对齐")
    ap.add_argument("--openloop-interval-ms", type=float, default=200.0,
                    help="openloop 模式的到达间隔（毫秒，默认 200，即 5 条/秒）")
    ap.add_argument("--repeats", type=int, default=PROTOCOL["repeats"])
    ap.add_argument("--window-s", type=float, default=PROTOCOL["window_s"])
    ap.add_argument("--concurrency", type=int, default=PROTOCOL["concurrency"])
    ap.add_argument("--max-tokens", type=int, default=PROTOCOL["max_tokens"])
    ap.add_argument("--warmup", type=int, default=PROTOCOL["warmup_requests"])
    ap.add_argument("--warmup-scope", choices=["each", "first"],
                    default=PROTOCOL.get("warmup_scope", "first"),
                    help="each = v1/v2 每窗前都预热；first = v3 只在第一窗前预热一次")
    ap.add_argument("--steady-skip", type=float,
                    default=PROTOCOL.get("steady_skip_s", 0.0),
                    help="v3 稳态子窗口丢弃的每窗开头秒数；0 表示不算稳态口径")
    ap.add_argument("--priority-mode", choices=["off", "fixed", "by_prompt_length"],
                    default="off",
                    help="off=不传 priority 字段（与历史基线完全等价，默认）；"
                         "fixed=所有请求同一 priority；"
                         "by_prompt_length=按 prompt 长度分配短/长两档")
    ap.add_argument("--priority-fixed", type=int, default=0,
                    help="--priority-mode=fixed 时的取值（默认 0）")
    ap.add_argument("--short-threshold", type=float, default=4096.0,
                    help="by_prompt_length 的短/长分界，单位 prompt token"
                         "（默认 4096 = BS32 下的前缀缓存块粒度）")
    ap.add_argument("--short-priority", type=int, default=0,
                    help="短请求 priority（vLLM 语义：数值小 = 先处理）")
    ap.add_argument("--long-priority", type=int, default=100,
                    help="长请求 priority")
    ap.add_argument("--chars-per-token", type=float, default=0.628,
                    help="prompt 无 token 元数据时按 len(text)*该系数估算 token 数")
    ap.add_argument("--prompt-set", action="append", default=[], metavar="PATH",
                    help="prompt jsonl 路径，可重复（默认 protocol.PROMPT_SET）；"
                         "同名 .meta.json 自动读取，用于 sha256 校验与 token 数注入")
    ap.add_argument("--mix-ratio", default="",
                    help="多个 --prompt-set 时的交错比例，如 9:1（前段:后段）；"
                         "留空则只允许一个 prompt 集")
    ap.add_argument("--mix-cycles", type=int, default=4,
                    help="交错周期数（默认 4）：8:1 乘 4 得 36 条，正好用满 32 条短集")
    ap.add_argument("--endpoint", default=default_endpoint(),
                    help="chat completions URL (default: $LLM_BENCH_BASE_URL or 127.0.0.1:8000)")
    ap.add_argument("--model", default=default_model(),
                    help="model id (default: $LLM_BENCH_MODEL; required for real runs)")
    ap.add_argument("--fake", action="store_true",
                    help="假数据自检：不碰网络，用于 CI 与冒烟")
    ap.add_argument("--api-key-env", default="VLLM_API_KEY")
    ap.add_argument("--probe", default="",
                    help="跑前能力探测：逗号分隔的到达间隔(ms)，如 1000,800,600；"
                         "每档跑 1 窗，报告 lag / 是否形成排队 / 可跑性后退出，"
                         "**不产生正式 repeats**")
    ap.add_argument("--probe-window-s", type=float, default=30.0,
                    help="probe 每档的窗口长度（秒，默认 30；probe 只判形态，"
                         "不追求统计量）")
    ap.add_argument("--max-lag-p95-ms", type=float, default=500.0,
                    help="跑中硬门禁：任一轮 arrival_lag p95 超过该值即写 "
                         "abort 并停止后续窗（默认 500；0 = 关闭门禁）")
    ap.add_argument("--server-max-num-seqs", type=int, default=32,
                    help="服务端 max_num_seqs，用于判定「是否真的形成等待队列」"
                         "（默认 32 = 当前部署，2026-11-19 用户定格；"
                         "in_flight 全程 <= 它 = 空测）")
    ap.add_argument("--max-backlog-growth", type=float, default=0.4,
                    help="跑中硬门禁：窗末未完成条数相对前序最小值的涨幅超过该值"
                         "即判越载并停止后续窗（默认 0.40；0 = 关闭）")
    ap.add_argument("--out-root", default=os.environ.get(
        "LLM_LAB_RUNS", str(ROOT / "runs")))
    ap.add_argument("--run-id", default="")
    args = ap.parse_args(argv)

    api_key = os.environ.get(args.api_key_env, "")
    if args.prompt_set:
        sets = []
        for p in args.prompt_set:
            ps_path = Path(p).resolve()
            s = load_prompt_set(ps_path, ps_path.with_name(ps_path.stem + ".meta.json"))
            n_inj = attach_prompt_tokens(s["prompts"], s.get("meta") or {})
            sets.append(s)
            print(f"prompt_set[{len(sets) - 1}]={s['path']} "
                  f"sha256={s['sha256'][:12]} count={s['count']} "
                  f"token注入={n_inj}/{s['count']}")
        prompts = mix_prompt_sets(sets, args.mix_ratio, args.mix_cycles)
        prompt_set = sets[0]          # run.json 记录首个集；混合信息另记
        n_injected = sum(1 for p in prompts if "n_tokens" in p)
        if args.mix_ratio:
            print(f"mix: {args.mix_ratio} x {args.mix_cycles} 周期 "
                  f"-> {len(prompts)} 条（源集 {len(sets)} 个）")
    else:
        prompt_set = load_prompt_set()
        prompts = prompt_set["prompts"]
        n_injected = attach_prompt_tokens(prompts, prompt_set.get("meta") or {})
    classify = make_classifier(args)
    # 输入序列指纹：两步必须完全一致才算同输入，否则比的是 prompt 循环差异。
    prompt_seq_hash = _ids_hash([p["id"] for p in prompts])
    endpoint = args.endpoint
    model = args.model
    tag = {"continuous": "CONT", "barrier": "BARR",
           "openloop": "OPEN"}.get(args.mode, args.mode[:4].upper())
    run_id = args.run_id or f"FROZEN-{tag}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_root = Path(args.out_root)
    run_dir = out_root / run_id

    send_fn = fake_stream if args.fake else stream_once
    env = ({"system_fingerprint": "fake"} if args.fake
           else probe_env(endpoint, model, api_key))
    print(f"run_id={run_id} mode={args.mode} fingerprint={env.get('system_fingerprint')}")
    print(f"prompt_set={prompt_set['path']} sha256={prompt_set['sha256']} "
          f"count={prompt_set['count']}")
    print(f"protocol: c={args.concurrency} max_tokens={args.max_tokens} "
          f"window={args.window_s}s temp=0 repeats={args.repeats} warmup={args.warmup} "
          f"warmup_scope={args.warmup_scope} steady_skip={args.steady_skip}s "
          f"seq={prompt_seq_hash} prompts={len(prompts)}")
    if args.priority_mode != "off":
        print(f"priority: mode={args.priority_mode} "
              + (f"fixed={args.priority_fixed}" if args.priority_mode == "fixed"
                 else f"<{args.short_threshold:g}tok -> {args.short_priority}, "
                      f">={args.short_threshold:g}tok -> {args.long_priority}"))
        if args.priority_mode == "by_prompt_length":
            print(f"          token 元数据注入 {n_injected}/{len(prompts)} 条，"
                  f"其余按 len(text)*{args.chars_per_token} 估算")

    run_dir.mkdir(parents=True, exist_ok=True)
    events = []
    metrics = []
    repeats: list[dict] = []
    all_samples: list[dict] = []
    stopped_early = None

    def ev(name: str, **extra) -> None:
        row = {"ts": round(time.time(), 3), "event": name, **extra}
        events.append(row)
        print(f"[{time.strftime('%H:%M:%S')}] {name} {extra}" if extra else
              f"[{time.strftime('%H:%M:%S')}] {name}")

    ev("start", mode=args.mode, run_id=run_id)

    # ---------------------------------------------------------------- 跑前探测
    # 200 ms 那轮的直接教训：没有任何代码在跑之前问一句「这个到达率服务端接得住
    # 吗」。--probe 就是那句问话：每档 1 窗，只看形态（lag / 是否形成排队），
    # 不产生正式 repeats，也不写 bench.json。
    if args.probe:
        rates = [float(x) for x in args.probe.split(",") if x.strip()]
        if not rates:
            raise SystemExit("--probe 需要逗号分隔的到达间隔（毫秒），如 1000,800,600")
        probe_win = min(args.window_s, args.probe_window_s)
        warm_ok = 0
        for p in prompt_cycle(prompts, args.warmup):
            warm_ok += int(bool(send_fn(endpoint, model, p["text"],
                                        args.max_tokens, api_key)["ok"]))
        ev("probe_start", rates_ms=rates, window_s=probe_win, warmup_ok=warm_ok)
        rows = []
        for iv in rates:
            print(f"\n[probe] interval={iv:g} ms（{1000.0 / iv:.2f} 条/秒）"
                  f"  窗 {probe_win:g}s")
            win = run_window(args.mode, prompts, endpoint, model, api_key,
                             args.concurrency, args.max_tokens, probe_win,
                             transport=send_fn, classify=classify,
                             openloop_interval_ms=iv)
            s = summarize_window(win["records"], probe_win)
            v = window_validity(win, s, mode=args.mode, window_s=probe_win,
                                interval_ms=(iv if args.mode == "openloop" else None),
                                max_lag_ms=args.max_lag_p95_ms,
                                server_seqs=args.server_max_num_seqs)
            usable = bool(v["arrival_valid"]
                          and (v["queue_exposure_valid"]
                               or args.priority_mode == "off"))
            row = {
                "interval_ms": iv, "offered_per_s": round(1000.0 / iv, 3),
                "window_s": probe_win, "planned_n": v["planned_n"],
                "arrival_n": win["arrival_n"], "n": s["n"],
                "lag_p95_ms": v["lag_p95_ms"], "drained": s["drained"],
                "drain_frac": v["drain_frac"],
                "inflight_peak": v["inflight_peak"],
                "inflight_above_server_frac": v["inflight_above_server_frac"],
                "throughput_tok_s": s["throughput_tok_s"],
                "failed": s["failed"],
                "arrival_valid": v["arrival_valid"],
                "queue_exposure_valid": v["queue_exposure_valid"],
                "usable": usable, "reasons": v["reasons"],
            }
            rows.append(row)
            print(f"        arrival_n={row['arrival_n']}/{row['planned_n']} "
                  f"lag_p95={row['lag_p95_ms']} ms drained={row['drained']} "
                  f"({row['drain_frac']}) inflight_peak={row['inflight_peak']} "
                  f">seqs={row['inflight_above_server_frac']:.0%} "
                  f"tok/s={row['throughput_tok_s']} usable={row['usable']}")
        write_json(run_dir / "probe.json", {
            "mode": args.mode, "window_s": probe_win, "warmup": args.warmup,
            "warmup_ok": warm_ok, "max_lag_ms": args.max_lag_p95_ms,
            "server_max_num_seqs": args.server_max_num_seqs,
            "priority_mode": args.priority_mode,
            "usability_rule": "arrival_valid 且（priority=off 或 queue_exposure_valid）",
            "rates": rows,
        })
        ok_rates = [x["interval_ms"] for x in rows if x["usable"]]
        print("\n=== probe 结论 ===")
        if ok_rates:
            print(f"可用到达间隔（毫秒）：{ok_rates}")
            print(f"其中速率最低、最稳的一档：{max(ok_rates):g} ms")
            if args.priority_mode != "off":
                print("提示：要测 priority，还要求 inflight 在大部分时间 > "
                      f"{args.server_max_num_seqs}；不满足则空测。")
        else:
            print("没有一档可用：要么到达轨迹被客户端拖垮（lag / 发送不足），"
                  "要么全程没形成等待队列（priority 无处可排）。")
        print(f"probe -> {run_dir / 'probe.json'}")
        return 0

    for r in range(1, args.repeats + 1):
        # 预热：固定条数，串行，不计入窗口。
        # v3：warmup_scope=first 时只在第一窗前做，之后窗口**背靠背**，
        #     避免每窗之间插 13–15 s 近空载（那会让每窗重新从并发 1 爬坡）。
        do_warm = (r == 1) or (args.warmup_scope == "each")
        warm_start = time.perf_counter()
        warm_ok = 0
        warm_err = None
        if do_warm:
            for p in prompt_cycle(prompts, args.warmup):
                rec = send_fn(endpoint, model, p["text"], args.max_tokens, api_key)
                warm_ok += int(bool(rec["ok"]))
                if not rec["ok"]:
                    warm_err = rec.get("error")
            warm_s = round(time.perf_counter() - warm_start, 3)
            ev("warmup", repeat=r, requests=args.warmup, ok=warm_ok, seconds=warm_s,
               error=warm_err)
        else:
            warm_s = 0.0
            ev("warmup_skipped", repeat=r, reason="warmup_scope=first")
        if do_warm and warm_ok == 0:
            ev("abort", repeat=r, reason="warmup_all_failed", error=warm_err)
            # 中止也要留痕：否则 run 目录全空，事后分不清「没启动」与「起了就死」
            (run_dir / "events.jsonl").write_text(
                "\n".join(json.dumps(x, ensure_ascii=False) for x in events),
                encoding="utf-8")
            raise SystemExit(
                f"预热 0/{args.warmup} 成功，未进入计时窗口，本 run 作废。\n"
                f"首个错误：{warm_err}\n"
                f"检查 VLLM_API_KEY 是否传入该终端、端点 {endpoint} 是否可达。")

        win = run_window(args.mode, prompts, endpoint, model, api_key,
                         args.concurrency, args.max_tokens, args.window_s,
                         classify=classify,
                         openloop_interval_ms=args.openloop_interval_ms)
        summary = summarize_window(win["records"], args.window_s,
                                   steady_skip=args.steady_skip)
        intervals = interval_stats(win["records"], args.mode, win["batches"])
        validity = window_validity(
            win, summary, mode=args.mode, window_s=args.window_s,
            interval_ms=(args.openloop_interval_ms
                         if args.mode == "openloop" else None),
            max_lag_ms=args.max_lag_p95_ms,
            server_seqs=args.server_max_num_seqs)
        # 拆成两项落到 summary：单窗只能判「到达」与「队列暴露」（代理指标），
# 跨组可比性（comparison）只有在对比两条 run 时才成立。
        summary["arrival_valid"] = validity["arrival_valid"]
        summary["queue_exposure_valid"] = validity["queue_exposure_valid"]
        # 积压增长：与**本 run 已跑窗口里最小的 drained**比，而不是与上一窗比
        growth = backlog_growth_frac(
            [x["summary"].get("drained", 0) for x in repeats], summary["drained"])
        validity["backlog_growth"] = growth
        row = {
            "repeat": r,
            "summary": summary,
            "validity": validity,
            "planned_n": validity["planned_n"],
            "wall_s": win["wall_s"],
            "peak_inflight": win["peak_inflight"],
            "openloop_max_workers": win.get("openloop_max_workers"),
            "batches": win["batches"],
            "interval": intervals,
            "arrival_ids_hash": win["arrival_ids_hash"],
            "arrival_plan_hash": win["arrival_plan_hash"],
            "arrival_plan_hash_v2": win["arrival_plan_hash_v2"],
            "priority_plan_hash": win["priority_plan_hash"],
            "arrival_lag_ms": win["arrival_lag_ms"],
            "arrival_n": win["arrival_n"],
            "warmup": {"requests": args.warmup, "ok": warm_ok, "seconds": warm_s},
        }
        repeats.append(row)
        for rec in win["records"]:
            rec["repeat"] = r
            all_samples.append(rec)
        raw_path = run_dir / "raw" / f"samples-{args.mode}-r{r}.jsonl"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in win["records"]),
            encoding="utf-8")
        # 增量落 events：被 Ctrl-C / 终端断开打断的 run 原本一个字都不留
        # （events.jsonl 只在跑完时写），事后分不清「跑到第几窗、为什么停」。
        (run_dir / "events.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in events),
            encoding="utf-8")
        metrics.extend([
            {"ts": round(time.time(), 3), "step": r, "name": "throughput_tok_s",
             "value": summary["throughput_tok_s"]},
            {"ts": round(time.time(), 3), "step": r, "name": "requests_in_window",
             "value": summary["n"]},
            {"ts": round(time.time(), 3), "step": r, "name": "ttft_ms_p50",
             "value": summary["ttft_ms"]["p50"]},
            {"ts": round(time.time(), 3), "step": r, "name": "e2e_ms_p50",
             "value": summary["e2e_ms"]["p50"]},
        ])
        if summary.get("steady"):
            metrics.append({"ts": round(time.time(), 3), "step": r,
                            "name": "throughput_steady_tok_s",
                            "value": summary["steady"]["throughput_tok_s"]})
        print(f"repeat {r}/{args.repeats}: tok/s={summary['throughput_tok_s']} "
              f"n={summary['n']} failed={summary['failed']} "
              f"ttft_p50={summary['ttft_ms']['p50']}ms "
              f"e2e_p50={summary['e2e_ms']['p50']}ms "
              f"drained={summary['drained']}")
        if win["arrival_plan_hash"] or win["arrival_lag_ms"]:
            print(f"        arrival: plan={win['arrival_plan_hash']} "
                  f"ids={win['arrival_ids_hash']} n={win['arrival_n']} "
                  f"lag={win['arrival_lag_ms']}")
            grp = summary.get("groups") or {}
            for gname in ("short", "long"):
                gg = grp.get(gname)
                if gg:
                    print(f"          {gname}: n={gg['n']} "
                          f"ttft_p50={gg['ttft_ms']['p50']} "
                          f"p95={gg['ttft_ms']['p95']} max={gg['ttft_ms']['max']} "
                          f"drained={gg['drained']} "
                          f"wait_ms={gg['drained_wait_ms']} "
                          f"with_ft={gg['drained_with_first_token']}")
        if summary.get("steady"):
            st = summary["steady"]
            print(f"        steady(>{args.steady_skip:g}s, {st['span_s']:g}s): "
                  f"tok/s={st['throughput_tok_s']} n={st['n']} "
                  f"tpot_p50={st['tpot_ms']['p50']}ms")
        print(f"        arrival_valid={validity['arrival_valid']} "
              f"queue_exposure_valid={validity['queue_exposure_valid']} "
              f"inflight>seqs={validity['inflight_above_server_frac']:.0%} "
              f"drain_frac={validity['drain_frac']} "
              f"backlog_growth={validity['backlog_growth']}")
        for why in validity["reasons"]:
            print(f"        [WARN] {why}")
        # 跑中硬门禁 ①：到达轨迹不成立就立刻停，不再生产废数据。
        # 200 ms 那轮 lag p95 = 20.2 s 却把 5 窗全跑完了，缺的就是这一行。
        if not validity["arrival_valid"]:
            ev("abort", repeat=r, reason="arrival_trajectory_invalid",
               detail=validity["reasons"])
            stopped_early = {"repeat": r, "reason": "arrival_trajectory_invalid"}
            break
        # 跑中硬门禁 ②：积压随时窗增长（越载）⇒ 窗与窗不再独立，立刻停。
        growth = validity["backlog_growth"]
        if (args.max_backlog_growth > 0 and growth is not None
                and growth >= args.max_backlog_growth):
            ev("abort", repeat=r, reason="backlog_growth",
               backlog_growth=growth, limit=args.max_backlog_growth,
               drained=summary["drained"])
            stopped_early = {"repeat": r, "reason": "backlog_growth"}
            break

    ev("stop", run_id=run_id)
    agg = aggregate(repeats)
    valid = run_validity(repeats, agg, priority_mode=args.priority_mode,
                         max_lag_ms=args.max_lag_p95_ms,
                         stopped_early=stopped_early)
    run_meta = {
        "run_id": run_id, "target": "default", "kind": "llm-serving",
        "created_at": time.time(),
        "status": "invalid" if stopped_early else "finished",
        "model": model,
        "engine": {
            "name": "vllm-ascend", "system_fingerprint": env.get("system_fingerprint"),
            "version_note": "指纹给出引擎版本与并行度；镜像 digest 未取证（无宿主机权限）",
            "parallelism": "tp8-ep-dp1",
        },
        "load": {
            "mode": args.mode, "concurrency": args.concurrency,
            "max_tokens": args.max_tokens, "window_s": args.window_s,
            "temperature": 0, "repeats": args.repeats, "warmup": args.warmup,
            "warmup_scope": args.warmup_scope, "steady_skip_s": args.steady_skip,
            "protocol_rev": PROTOCOL.get("protocol_rev"),
            "prompt_set": prompt_set["path"], "prompt_sha256": prompt_set["sha256"],
            "mix_ratio": args.mix_ratio, "mix_cycles": args.mix_cycles,
            "openloop_interval_ms": args.openloop_interval_ms,
            "prompt_count_used": len(prompts),
            "prompt_sequence_hash": prompt_seq_hash,
            "priority_mode": args.priority_mode,
            "priority_fixed": args.priority_fixed,
            "short_threshold": args.short_threshold,
            "short_priority": args.short_priority,
            "long_priority": args.long_priority,
            "chars_per_token": args.chars_per_token,
            "max_lag_p95_ms": args.max_lag_p95_ms,
            "server_max_num_seqs": args.server_max_num_seqs,
        },
        "protocol": PROTOCOL,
        "endpoint": endpoint,
        "env_probe": env,
        "data_valid": valid,
        "stopped_early": stopped_early,
        "finished_at": time.time(),
    }
    write_json(run_dir / "run.json", run_meta)
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in events), encoding="utf-8")
    bench = {
        "protocol": PROTOCOL,
        "mode": args.mode,
        "repeats": repeats,
        "aggregate": agg,
        "data_valid": valid,
        "stopped_early": stopped_early,
        "summary": repeats[-1]["summary"] if repeats else {},
        "samples": all_samples,
        "env_probe": env,
        "prompt_set": {"path": prompt_set["path"], "sha256": prompt_set["sha256"],
                       "count": prompt_set["count"],
                       "sequence_hash": prompt_seq_hash,
                       "count_used": len(prompts)},
    }
    metrics.append({"ts": round(time.time(), 3), "step": 0,
                    "name": "overall_valid",
                    "value": 1.0 if valid["overall_valid"] else 0.0})
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in metrics),
        encoding="utf-8")
    write_json(run_dir / "bench.json", bench)

    print("\n=== aggregate ===")
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print()
    print(stability_line(agg["stability"]))
    print("\n=== data_valid ===")
    print(json.dumps(valid, ensure_ascii=False, indent=2))
    if not valid["overall_valid"]:
        print("[WARN] 本轮 overall_valid=false，不得进入两组对比表")
    if stopped_early:
        print(f"[WARN] 提前中止于第 {stopped_early['repeat']} 窗"
              f"（{stopped_early['reason']}）")
    print(f"\nrun -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
