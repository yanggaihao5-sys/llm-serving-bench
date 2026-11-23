
"""T2-FROZEN-v1 冻结协议与窗口边界的假数据测试（CI 不碰网络）。"""
import argparse
import json
import time

import pytest

from llmops import frozen_bench, protocol


def fake_transport(endpoint, model, text, max_tokens, api_key):
    """假发送器：固定 5 ms 延迟，64 token 输出。"""
    t_send = time.perf_counter()
    time.sleep(0.005)
    t_end = time.perf_counter()
    return {"ok": True, "t_send": t_send, "t_first": t_end, "t_end": t_end,
            "completion_tokens": max_tokens, "prompt_tokens": 12,
            "cached_tokens": 0}


def test_prompt_set_frozen():
    ps = protocol.load_prompt_set()
    assert ps["count"] == 32
    assert len(ps["sha256"]) == 64
    ids = [p["id"] for p in ps["prompts"]]
    assert ids == [f"p{i:02d}" for i in range(1, 33)]
    assert ps["meta"]["sha256"] == ps["sha256"]
    # prompt_tokens is optional: it is tokenizer/deployment-specific and a
    # public prompt set must not depend on one model's tokenizer.
    if "prompt_tokens" in ps["meta"]:
        assert ps["meta"]["prompt_tokens"]["per_id"]["p01"] > 0
        assert ps["meta"]["prompt_tokens"]["min"] <= ps["meta"]["prompt_tokens"]["median"]
    assert ps["meta"]["char_len"]["min"] <= ps["meta"]["char_len"]["median"]


def test_hash_drift_rejected(tmp_path):
    p = tmp_path / "p.jsonl"
    p.write_text('{"id":"p01","text":"x"}\n', encoding="utf-8")
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError):
        protocol.load_prompt_set(p, m)


def _rec(t_end_rel: float, ok: bool = True, toks: int = 64,
         t_start: float = 0.0) -> dict:
    return {"ok": ok, "t_end_rel": t_end_rel, "t_start": t_start,
            "ttft_s": 0.5, "e2e_s": 1.0,
            "tpot_s": 0.01, "completion_tokens": toks, "cached_tokens": 0,
            "t_send": 0.0, "t_first": 0.5, "t_end": 1.0, "prompt_id": "p01"}


def test_window_boundary():
    # 10 与 60.0 计；60.0001 排空不计；61 排空；失败不计分子
    recs = [_rec(10), _rec(60.0), _rec(60.0001), _rec(61), _rec(5, ok=False)]
    s = frozen_bench.summarize_window(recs, 60.0)
    assert s["n"] == 2
    assert s["drained"] == 2
    assert s["failed"] == 1
    assert s["issued"] == 5
    assert s["completion_tokens"] == 128
    assert s["throughput_tok_s"] == pytest.approx(128 / 60, abs=0.01)


def test_continuous_window():
    ps = protocol.load_prompt_set()
    win = frozen_bench.run_window("continuous", ps["prompts"], "e", "m", "",
                                  concurrency=4, max_tokens=64, window_s=0.3,
                                  transport=fake_transport)
    s = frozen_bench.summarize_window(win["records"], 0.3)
    assert s["n"] > 0 and s["failed"] == 0
    assert s["throughput_tok_s"] > 0
    assert win["batches"] is None
    assert frozen_bench.interval_stats(
        win["records"], "continuous", None)["kind"] == "continuous"


def test_barrier_batches():
    ps = protocol.load_prompt_set()
    win = frozen_bench.run_window("barrier", ps["prompts"], "e", "m", "",
                                  concurrency=4, max_tokens=64, window_s=0.2,
                                  transport=fake_transport)
    assert win["batches"]
    assert all(b["n"] == 4 for b in win["batches"])
    iv = frozen_bench.interval_stats(win["records"], "barrier", win["batches"])
    assert iv["kind"] == "barrier"
    assert iv["batches"] == len(win["batches"])


def test_aggregate_median_range():
    reps = [{"summary": {"throughput_tok_s": v, "n": 100,
                         "ttft_ms": {"p50": 1, "p95": 2},
                         "e2e_ms": {"p50": 3, "p95": 4},
                         "tpot_ms": {"p50": 5, "p95": 6}, "failed": 0}}
            for v in (10, 20, 30, 40, 50)]
    agg = frozen_bench.aggregate(reps)
    assert agg["throughput_tok_s"]["median"] == 30
    assert agg["throughput_tok_s"]["min"] == 10
    assert agg["throughput_tok_s"]["max"] == 50
    assert agg["throughput_tok_s"]["range_pct"] == pytest.approx(133.33, abs=0.01)


# ---------------------------------------------------------------- priority 调度
# 2026-11-18 新增：--priority-mode 三档。默认 off 必须与历史基线完全等价，
# 否则 c=40 176.92 / BS32 163.48 这些既有数据全部不可比。

def _ns(**kw):
    """构造 make_classifier 需要的最小 Namespace。"""
    base = {"priority_mode": "off", "priority_fixed": 0, "short_threshold": 4096.0,
            "short_priority": 0, "long_priority": 100, "chars_per_token": 0.628}
    base.update(kw)
    return argparse.Namespace(**base)


def test_priority_off_marks_group_but_sends_no_field():
    """off 档：priority 为 None（不写 body 字段），但 group 仍按长度标出。

    若 group 也为 None，对照组就没有分组数据，「短请求是否改善」无法回答。
    """
    clf = frozen_bench.make_classifier(_ns())
    assert clf({"id": "a", "text": "x" * 10}) == (None, "short")
    assert clf({"id": "b", "text": "", "n_tokens": 9000}) == (None, "long")


def test_priority_by_prompt_length_prefers_meta_tokens():
    """有 n_tokens 元数据时按真实 token 分组，不靠字符估算。"""
    clf = frozen_bench.make_classifier(_ns(priority_mode="by_prompt_length"))
    assert clf({"id": "s", "text": "", "n_tokens": 264}) == (0, "short")
    assert clf({"id": "l", "text": "", "n_tokens": 7017}) == (100, "long")
    # 恰在阈值上算 long（>=threshold）
    assert clf({"id": "e", "text": "", "n_tokens": 4096})[1] == "long"


def test_priority_falls_back_to_chars():
    """无 token 元数据时用 len(text)*chars_per_token 估算，仍然分得出组。"""
    clf = frozen_bench.make_classifier(_ns(priority_mode="by_prompt_length"))
    assert clf({"id": "s", "text": "x" * 400}) == (0, "short")       # ~251 tok
    assert clf({"id": "l", "text": "x" * 12000}) == (100, "long")    # ~7536 tok


def test_priority_fixed_applies_to_all_but_keeps_group():
    """fixed 档所有请求同 priority，但 group 仍按长度标，供分组统计。"""
    clf = frozen_bench.make_classifier(_ns(priority_mode="fixed", priority_fixed=7))
    assert clf({"id": "s", "text": "", "n_tokens": 264}) == (7, "short")
    assert clf({"id": "l", "text": "", "n_tokens": 7017}) == (7, "long")


def test_run_window_records_priority_and_group():
    """classify 注入后每条 record 带 priority / prompt_group / t_first_rel。

    fake_transport 不接 priority 关键字 ⇒ 同时验证了 _accepts_priority 的降级
    路径（否则真实发送器与 CI 假发送器会被迫同签名）。
    """
    ps = protocol.load_prompt_set()
    clf = frozen_bench.make_classifier(_ns(priority_mode="fixed", priority_fixed=7))
    win = frozen_bench.run_window("continuous", ps["prompts"], "e", "m", "",
                                  concurrency=2, max_tokens=8, window_s=0.12,
                                  transport=fake_transport, classify=clf)
    assert win["records"]
    assert all(r["priority"] == 7 for r in win["records"])
    assert all(r["prompt_group"] in ("short", "long") for r in win["records"])
    assert all("t_first_rel" in r for r in win["records"])


def test_summarize_groups_reports_ttft_max_and_drain():
    """分组统计：短/长分开的 TTFT p50/p95/max + 跨窗未返回条数。"""
    recs = [
        {**_rec(10), "prompt_group": "short", "priority": 0, "prompt_tokens": 264},
        {**_rec(11), "prompt_group": "long", "priority": 100, "prompt_tokens": 7017},
        {**_rec(61), "prompt_group": "long", "priority": 100, "prompt_tokens": 7017},
    ]
    g = frozen_bench.summarize_groups(recs, 60.0)
    assert g["short"]["n"] == 1
    assert g["short"]["priorities"] == [0]
    assert g["short"]["prompt_tokens_p50"] == 264
    assert g["long"]["n"] == 1          # 61 s 那条跨窗，不计入 in-window
    assert g["long"]["drained"] == 1    # 但必须记为「跨窗未返回」
    assert g["long"]["priorities"] == [100]
    assert g["long"]["ttft_ms"]["max"] == 500.0   # _rec 的 ttft_s=0.5


def test_summarize_groups_empty_when_no_group():
    """未开启 priority（group=None）时 groups 为空，不污染既有 summary。"""
    assert frozen_bench.summarize_groups([_rec(10)], 60.0) == {}
    s = frozen_bench.summarize_window([_rec(10)], 60.0)
    assert s["groups"] == {}


def test_drained_only_counts_submitted_before_window_end():
    """drained 口径（用户 2026-11-18）：窗口结束前**已提交**、窗末仍未完成。

    continuous worker 的 `while perf_counter() < deadline` 与真正 t_send 之间
    有几微秒间隙，窗口后才发出的请求必须排除，否则会被误算成「被饿住」。
    """
    recs = [
        {**_rec(10, t_start=1.0), "prompt_group": "long"},
        {**_rec(70, t_start=30.0), "prompt_group": "long"},   # 窗内提交、跨窗完成
        {**_rec(80, t_start=61.0), "prompt_group": "long"},   # 窗后才提交 → 排除
    ]
    g = frozen_bench.summarize_groups(recs, 60.0)
    assert g["long"]["n"] == 1
    assert g["long"]["drained"] == 1
    assert g["long"]["inflight_at_window_end"] == 1


def test_inflight_at_window_end_includes_failures():
    """失败的在飞请求不能被 ok 过滤掉。"""
    recs = [
        {**_rec(70, ok=True, t_start=20.0), "prompt_group": "short"},
        {**_rec(90, ok=False, t_start=21.0), "prompt_group": "short"},
    ]
    g = frozen_bench.summarize_groups(recs, 60.0)
    assert g["short"]["drained"] == 1              # 只有 ok 的那条
    assert g["short"]["inflight_at_window_end"] == 2   # 两条都算在飞


def test_attach_prompt_tokens_handles_both_meta_shapes():
    """两个冻结集的 meta 字段名不同，注入函数必须都能认。

    frozen-v1 用 prompt_tokens.per_id（全覆盖），comm-7k-v1 用
    tokens_measured.per_id（只采样 6/96）——只认前者会让 7K 整组退回字符估算。
    """
    prompts = [{"id": "p01", "text": "a"}, {"id": "c7k-001", "text": "b"},
               {"id": "x99", "text": "c"}]
    n = frozen_bench.attach_prompt_tokens(
        prompts,
        {"prompt_tokens": {"per_id": {"p01": 264}},
         "tokens_measured": {"per_id": {"c7k-001": 7085}}})
    assert n == 2
    assert prompts[0]["n_tokens"] == 264
    assert prompts[1]["n_tokens"] == 7085
    assert "n_tokens" not in prompts[2]      # 无元数据 → 交给字符回退


def test_attach_prompt_tokens_tolerates_missing_meta():
    prompts = [{"id": "p01", "text": "a"}]
    assert frozen_bench.attach_prompt_tokens(prompts, {}) == 0
    assert frozen_bench.attach_prompt_tokens(prompts, None) == 0
    assert "n_tokens" not in prompts[0]


# ---------------------------------------------------------------- 混合集
# 2026-11-18：沙箱禁止脚本写工作区，故不落盘新建 mixed.jsonl，改为
# --prompt-set A --prompt-set B --mix-ratio 8:1 --mix-cycles 4 内存交错。

def _set(n: int, prefix: str) -> dict:
    return {"prompts": [{"id": f"{prefix}{i}", "text": "x"} for i in range(n)]}


def test_mix_prompt_sets_interleaves_not_concatenates():
    """8:1×1 = 8 短 + 1 长，长必须落在第 9 位而不是拼在尾部。"""
    out = frozen_bench.mix_prompt_sets([_set(32, "s"), _set(4, "l")], "8:1",
                                       cycles=1)
    assert len(out) == 9
    assert [r["id"] for r in out] == [f"s{i}" for i in range(8)] + ["l0"]


def test_mix_prompt_sets_four_cycles_hits_positions():
    """8:1×4 = 36 条，长在 9/18/27/36 —— 与手写混合集的位置完全一致。"""
    out = frozen_bench.mix_prompt_sets([_set(32, "s"), _set(4, "l")], "8:1",
                                       cycles=4)
    assert len(out) == 36
    long_positions = [i + 1 for i, r in enumerate(out) if r["id"].startswith("l")]
    assert long_positions == [9, 18, 27, 36]


def test_mix_prompt_sets_9to1_wraps_short_set():
    """9:1×4 = 40 条，短集只有 32 条 ⇒ 循环取，产生重复 id 但比例正确。"""
    out = frozen_bench.mix_prompt_sets([_set(32, "s"), _set(4, "l")], "9:1",
                                       cycles=4)
    assert len(out) == 40
    assert sum(1 for r in out if r["id"].startswith("l")) == 4
    assert sum(1 for r in out if r["id"].startswith("s")) == 36


def test_mix_prompt_sets_requires_ratio_for_multiple_sets():
    with pytest.raises(SystemExit):
        frozen_bench.mix_prompt_sets([_set(2, "a"), _set(2, "b")], "")
    with pytest.raises(SystemExit):
        frozen_bench.mix_prompt_sets([_set(2, "a")], "9:1")   # 段数不匹配
    with pytest.raises(SystemExit):
        frozen_bench.mix_prompt_sets([_set(2, "a"), _set(2, "b")], "9:x")


def test_mix_prompt_sets_single_set_passthrough():
    """单个 --prompt-set 且不给 ratio 时原样返回，不改变既有用法。"""
    s = _set(5, "p")
    assert [r["id"] for r in frozen_bench.mix_prompt_sets([s], "")] == \
           [f"p{i}" for i in range(5)]


# ---------------------------------------------------------------- 输入序列指纹
# 用户 2026-11-18：两组实验的到达序列必须可比，否则测到的是循环/时序差异。

def test_ids_hash_deterministic_and_order_sensitive():
    a = frozen_bench._ids_hash(["p01", "p02", "c7k-001"])
    assert a == frozen_bench._ids_hash(["p01", "p02", "c7k-001"])
    assert len(a) == 16
    # 顺序变 ⇒ hash 变（交错序列的意义就在顺序）
    assert a != frozen_bench._ids_hash(["p02", "p01", "c7k-001"])
    # 内容变 ⇒ hash 变
    assert a != frozen_bench._ids_hash(["p01", "p02", "c7k-002"])


def test_run_window_reports_arrival_hash():
    """每窗必须给出实际到达序列 hash，供两组逐窗比对。"""
    ps = protocol.load_prompt_set()
    win = frozen_bench.run_window("barrier", ps["prompts"], "e", "m", "",
                                  concurrency=4, max_tokens=8, window_s=0.1,
                                  transport=fake_transport)
    assert len(win["arrival_ids_hash"]) == 16
    assert win["arrival_n"] > 0
    # 到达序列只由窗内已提交请求构成
    assert win["arrival_n"] <= len(win["records"]) + 1


# ---------------------------------------------------------------- 开环到达轨迹
# T-SRF-C 要求：到达轨迹与完成时刻解耦，否则短请求先完成会自我放大占比。

def test_openloop_uses_fixed_schedule():
    ps = protocol.load_prompt_set()
    win = frozen_bench.run_window("openloop", ps["prompts"], "e", "m", "",
                                  concurrency=4, max_tokens=8, window_s=0.2,
                                  transport=fake_transport,
                                  openloop_interval_ms=20.0)
    scheds = sorted(r["t_sched_rel"] for r in win["records"]
                    if r.get("t_sched_rel") is not None)
    assert len(scheds) >= 5, "开环必须按时间表发出多条"
    # 计划时刻严格等于 k*interval
    assert all(abs(s - i * 0.02) < 1e-6 for i, s in enumerate(scheds))
    assert len(win["arrival_plan_hash"]) == 16
    assert win["arrival_lag_ms"] is not None


def test_openloop_plan_hash_independent_of_priority():
    """两组 priority 不同，但到达轨迹 hash 必须相同 —— 否则输入不可比。"""
    ps = protocol.load_prompt_set()
    base = dict(priority_mode="off", priority_fixed=0, short_threshold=4096,
                short_priority=0, long_priority=100, chars_per_token=0.628)
    kw = dict(concurrency=4, max_tokens=8, window_s=0.2,
              transport=fake_transport, openloop_interval_ms=20.0)
    w_off = frozen_bench.run_window(
        "openloop", ps["prompts"], "e", "m", "",
        classify=frozen_bench.make_classifier(argparse.Namespace(**base)), **kw)
    w_on = frozen_bench.run_window(
        "openloop", ps["prompts"], "e", "m", "",
        classify=frozen_bench.make_classifier(
            argparse.Namespace(**{**base, "priority_mode": "by_prompt_length"})),
        **kw)
    assert w_off["arrival_plan_hash"] == w_on["arrival_plan_hash"]
    # v2（含内容指纹）同样必须相等：两组发的是同一份完整计划
    assert w_off["arrival_plan_hash_v2"] == w_on["arrival_plan_hash_v2"]
    assert w_off["arrival_plan_hash_v2"] != w_off["arrival_plan_hash"]
    # 处理变量指纹相反：priority 分配必须不同，且被单独记录
    assert w_off["priority_plan_hash"] != w_on["priority_plan_hash"]


def test_plan_hash_v2_covers_prompt_content():
    """同 id、同长度、内容不同 ⇒ v1 撞车、v2 必须区分（用户 2026-11-18 追问）。"""
    kw = dict(concurrency=2, max_tokens=8, window_s=0.2,
              transport=fake_transport, openloop_interval_ms=20.0)
    a = frozen_bench.run_window(
        "openloop", [{"id": "same", "text": "abc"},
                     {"id": "same", "text": "xyz"}], "e", "m", "", **kw)
    b = frozen_bench.run_window(
        "openloop", [{"id": "same", "text": "xyz"},
                     {"id": "same", "text": "abc"}], "e", "m", "", **kw)
    assert a["arrival_plan_hash"] == b["arrival_plan_hash"]      # v1 看不出差别
    assert a["arrival_plan_hash_v2"] != b["arrival_plan_hash_v2"]  # v2 必须看出


def test_drained_wait_and_first_token():
    """条数持平 ≠ 等待未恶化：要给出已等待时长与是否已出首 token。"""
    recs = [
        {**_rec(70, t_start=10.0), "prompt_group": "long", "t_first_rel": 55.0},
        {**_rec(80, t_start=40.0), "prompt_group": "long", "t_first_rel": 65.0},
    ]
    g = frozen_bench.summarize_groups(recs, 60.0)["long"]
    assert g["drained"] == 2
    # 已等待：60-10=50s 与 60-40=20s
    assert g["drained_wait_ms"]["max"] == 50000.0
    # 只有第一条在窗内出了首 token（55s）；第二条 65s 已越窗
    assert g["drained_with_first_token"] == 1


# ---------------------------------------------------------------- 三层门禁
# 2026-11-18 教训：lag p95 = 20 s 的 run 照样跑完 5 窗、照样打印 stability PASS，
# 因为 MAD 只看吞吐离散度。这三层就是把它接成判据。

def _win(records, *, lag_p95=1.0, arrival_n=100, plan="h", peak=5):
    return {"records": records,
            "arrival_lag_ms": {"p50": 0.8, "p95": lag_p95, "max": lag_p95},
            "arrival_n": arrival_n, "peak_inflight": peak,
            "arrival_plan_hash": plan}


def _sum(n=1, drained=0):
    return {"n": n, "drained": drained}


def test_window_validity_rejects_late_arrival():
    """200 ms 那轮：lag p95 = 20.2 s、窗内实发 185/300 ⇒ 必须判废。"""
    recs = [_rec(1.0, t_start=i * 0.2) for i in range(5)]
    v = frozen_bench.window_validity(
        _win(recs, lag_p95=20232.9, arrival_n=185), _sum(),
        mode="openloop", window_s=60.0, interval_ms=200.0,
        max_lag_ms=500.0, server_seqs=40)
    assert v["planned_n"] == 300
    assert v["arrival_valid"] is False
    assert v["ok"] is False
    assert any("lag_p95" in r for r in v["reasons"])
    assert any("185/300" in r for r in v["reasons"])


def test_window_validity_detects_no_waiting_queue():
    """600 ms 那轮：在飞全程 <= 40 ⇒ 调度器无事可排，必须留痕（但不否决到达）。"""
    # 两条串行请求，任意时刻在飞都 <= 1
    recs = [_rec(10.0, t_start=0.0), _rec(20.0, t_start=10.0)]
    v = frozen_bench.window_validity(
        _win(recs, lag_p95=1.4, arrival_n=100), _sum(drained=33),
        mode="openloop", window_s=60.0, interval_ms=600.0,
        max_lag_ms=500.0, server_seqs=40)
    assert v["arrival_valid"] is True
    assert v["queue_exposure_valid"] is False
    assert v["inflight_above_server_frac"] == 0.0
    assert any("无等待队列" in r for r in v["reasons"])
    # 到达没问题 ⇒ 单窗 ok=True；「没有竞争」由 run 级 queue_exposure_valid 否决
    assert v["ok"] is True
    assert v["drain_frac"] == pytest.approx(33 / 100)


def test_inflight_above_frac_weights_time():
    """A 占 [0,10]，B 占 [2,3]，阈值 1 ⇒ 只有 [2,3] 期间在飞 = 2 条。"""
    recs = [_rec(10.0, t_start=0.0), _rec(3.0, t_start=2.0)]
    frac = frozen_bench._inflight_above_frac(recs, threshold=1)
    assert frac == pytest.approx(1.0 / 10.0, abs=1e-6)


def test_queue_formed_ignores_transient_crossing():
    """瞬时越过 max_num_seqs 不算排队：EXP 600 ms 实测峰值 41、只持续一瞬。"""
    recs = [_rec(60.0, t_start=0.0),      # 占满 [0,60]
            _rec(1.05, t_start=1.0)]      # 只让在飞在 [1,1.05] 变成 2 条
    v = frozen_bench.window_validity(
        _win(recs, lag_p95=1.4, arrival_n=100), _sum(drained=33),
        mode="openloop", window_s=60.0, interval_ms=600.0,
        max_lag_ms=500.0, server_seqs=1)
    assert v["inflight_above_server_frac"] < frozen_bench.OPENLOOP_QUEUE_MIN_FRAC
    assert v["queue_exposure_valid"] is False


def _reps(plans, queue_formed, arrival_valid=True, n=5):
    return [{"arrival_plan_hash": plans[i],
             "validity": {"arrival_valid": arrival_valid,
                          "queue_exposure_valid": queue_formed[i]}}
            for i in range(n)]


def test_run_validity_blocks_priority_run_without_queue():
    """priority 组必须真的排过队；到达有效不等于处理有效（四项分开）。"""
    agg = {"stability": {"pass": True, "MAD_pct": 1.0}}
    reps = _reps(["h"] * 5, [False] * 5)
    on = frozen_bench.run_validity(reps, agg, priority_mode="by_prompt_length",
                                   max_lag_ms=500.0)
    assert on["overall_valid"] is False
    assert on["arrival_valid"] is True           # 到达没问题
    assert on["queue_exposure_valid"] is False   # 但没有竞争
    assert on["comparison_valid"] is True
    assert any("等待竞争" in r for r in on["reasons"])
    # 对照组没有可排的东西是正常形态，不得因此判废
    off = frozen_bench.run_validity(reps, agg, priority_mode="off",
                                    max_lag_ms=500.0)
    assert off["overall_valid"] is True


def test_run_validity_rejects_plan_hash_drift():
    agg = {"stability": {"pass": True, "MAD_pct": 1.0}}
    reps = _reps(["a", "b", "a", "b", "a"], [True] * 5)
    v = frozen_bench.run_validity(reps, agg, priority_mode="off", max_lag_ms=500.0)
    assert v["overall_valid"] is False
    assert v["comparison_valid"] is False
    assert any("plan_hash" in r for r in v["reasons"])


def test_run_validity_reports_early_abort_window():
    agg = {"stability": {"pass": True, "MAD_pct": 1.0}}
    reps = _reps(["h"] * 5, [True] * 5, arrival_valid=False)
    v = frozen_bench.run_validity(reps, agg, priority_mode="off", max_lag_ms=500.0)
    assert v["overall_valid"] is False
    assert v["arrival_valid"] is False
    assert any("到达轨迹" in r for r in v["reasons"])


def test_run_validity_stopped_early_fails_comparison():
    """跑中门禁中止过的 run 内部不可比，哪怕到达与队列都过。"""
    agg = {"stability": {"pass": True, "MAD_pct": 1.0}}
    reps = _reps(["h"] * 2, [True] * 2, n=2)
    v = frozen_bench.run_validity(
        reps, agg, priority_mode="off", max_lag_ms=500.0,
        stopped_early={"repeat": 3, "reason": "backlog_growth"})
    assert v["comparison_valid"] is False
    assert v["overall_valid"] is False
    assert any("提前中止" in r for r in v["reasons"])


def test_backlog_growth_frac():
    """积压随时窗增长是越载信号：相对前序最小 drained 的涨幅。"""
    assert frozen_bench.backlog_growth_frac([], 30) is None
    assert frozen_bench.backlog_growth_frac([31, 34, 33], 31) == 0.0
    # 31 -> 47 = +51.6%
    assert frozen_bench.backlog_growth_frac([31, 34, 33], 47) == pytest.approx(0.516)
    # 600 ms 实测序列 31/34/33/33/35 ⇒ 涨幅 12.9%，远低于 40% 门槛
    assert frozen_bench.backlog_growth_frac([31, 34, 33, 33], 35) == pytest.approx(0.129)
    assert frozen_bench.backlog_growth_frac([0, 0], 5) is None


# ------------------------------------------------- 脉冲排空核算 / 静默验证（T-SRF-D）

def _pulse_rec(group="short", ok=True, t_start=0.0, t_first=1.0, t_end=5.0,
               error=None):
    return {"ok": ok, "prompt_group": group, "t_start": t_start,
            "t_first_rel": (t_first if ok else None),
            "t_end_rel": t_end, "error": error}


def test_analyze_drain_counts_never_merge():
    """提交/完成/HTTP 失败/超时/未提交必须分开，未完成不得静默删除。"""
    recs = [_pulse_rec(t_end=4.0), _pulse_rec(t_end=6.0),
            _pulse_rec(ok=False, t_end=9.0, error="HTTPError 500"),
            _pulse_rec(ok=False, t_end=30.0, error="TimeoutError('timed out')")]
    d = frozen_bench.analyze_drain(recs, inject_s=20.0, drain_max_s=180.0,
                                   planned_n=5)
    assert d["submitted_n"] == 4
    assert d["completed_n"] == 2
    assert d["http_failed_n"] == 1
    assert d["timed_out_n"] == 1
    assert d["unsubmitted_n"] == 1
    assert d["completion_rate"] == pytest.approx(0.5)
    assert d["server_may_carry_n"] == 1     # 超时放弃的请求服务端可能还在跑
    assert d["ok"] is False


def test_analyze_drain_long_wait_and_wall():
    """用户要求：长请求「提交→首 token」与「提交→完成」的最长墙钟时间。"""
    recs = [_pulse_rec("long", t_start=0.0, t_first=30.0, t_end=60.0),
            _pulse_rec("long", t_start=5.0, t_first=10.0, t_end=40.0),
            _pulse_rec("short", t_start=0.0, t_first=2.0, t_end=9.0)]
    d = frozen_bench.analyze_drain(recs, inject_s=20.0, drain_max_s=180.0)
    assert d["long_wait_to_first_token_max_ms"] == 30000.0
    assert d["long_wall_max_ms"] == 60000.0
    assert d["short_wait_to_first_token_max_ms"] == 2000.0
    assert d["long_completed_n"] == 2
    assert d["drain_s"] == pytest.approx(60.0 - 20.0)


def test_analyze_drain_flags_late_completion():
    """完成但超出排空期限 ⇒ 不算「期限内完成」，整轮不 ok。"""
    recs = [_pulse_rec(t_end=20.0 + 181.0)]
    d = frozen_bench.analyze_drain(recs, inject_s=20.0, drain_max_s=180.0)
    assert d["completed_n"] == 0
    assert d["completed_late_n"] == 1
    assert d["ok"] is False


def test_quiet_check_separates_client_from_server_evidence():
    """client_ok 与 strict_ok 分开：取不到服务端 waiting 时不算严格通过。"""
    q = frozen_bench.quiet_check(inflight_now=0, pending_tasks=0,
                                 server_waiting=None)
    assert q["client_ok"] is True
    assert q["strict_ok"] is False
    assert q["checks"]["server_waiting_zero"] is False
    assert q["evidence"] == "client_only"
    q2 = frozen_bench.quiet_check(inflight_now=0, pending_tasks=0,
                                  server_waiting=0, hbm_gb=41.2)
    assert q2["strict_ok"] is True and q2["evidence"] == "full"
    # 有在飞 / 有残留任务 ⇒ 直接不 ok
    assert frozen_bench.quiet_check(inflight_now=3, pending_tasks=0)["client_ok"] is False
    assert frozen_bench.quiet_check(inflight_now=0, pending_tasks=2)["client_ok"] is False


def test_quiet_check_detects_probe_drift():
    """静默段要用探针验证，而不是只等待：TTFT 漂移 >10% 判未恢复。"""
    q = frozen_bench.quiet_check(inflight_now=0, pending_tasks=0,
                                 probe_ttft_ms=2400.0, baseline_ttft_ms=2000.0)
    assert q["probe_ttft_drift_frac"] == pytest.approx(0.2)
    assert q["client_ok"] is False
