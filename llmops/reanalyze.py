"""离线重算工具：用 v3 稳态口径回算**历史** run 的原始样本，不需要重跑压测。

为什么可以回算
--------------
每次压测都在 `<run>/raw/samples-<mode>-r<k>.jsonl` 里留下**逐请求**明细
（t_start / t_end_rel / completion_tokens / ttft_s / e2e_s / tpot_s），
所以「窗口边界怎么切、丢不丢开头爬坡段」是**事后可改的口径**，不用重新打负载。
唯一**不能**回算的是加载形态（v3 把预热从「每窗一次」改成「只在第一窗前一次」）。

用法
----
  python -m llmops.reanalyze --root scripts/bench_out
  python -m llmops.reanalyze --root scripts/bench_out --steady-skip 15
  python -m llmops.reanalyze --root scripts/bench_out --steady-skip 10 --json out.json

and **不需要**重跑压测（`--steady-skip` 仅作诊断开关，默认 0.0）。
输出
----
每个 run 一行：窗数 / 全窗吞吐中位 / R / IQR / MAD / 孤点数 / PASS(按 MAD)。
末行给 4 种离散度的达标数与中位，供与文档里的回算对照表核对。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:  # 包内执行（python -m llmops.reanalyze）
    from .frozen_bench import stability_line, stability_stats
except ImportError:  # 直接执行脚本
    sys.path.insert(0, str(ROOT))
    from llmops.frozen_bench import stability_line, stability_stats


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return round(s[k], 4)


def stats(vals: list[float]) -> dict:
    """离散度口径统一走 frozen_bench.stability_stats（唯一实现，避免两处算法漂移）。"""
    if not vals:
        return {}
    st = stability_stats(vals)
    return {"n": st["n"], "median": st["median"],
            "min": round(min(vals), 2), "max": round(max(vals), 2),
            "R_pct": st["R_pct"], "IQR_pct": st["IQR_pct"],
            "MAD_pct": st["MAD_pct"], "stdev_pct": st["stdev_pct"],
            "pass": st["pass"], "outliers": st["outliers"], "rule": st["rule"]}


def load_window(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.append(r)
    return out


def measure(records: list[dict], window_s: float, skip_s: float) -> dict | None:
    """按给定窗口边界算吞吐与 TPOT；skip_s=0 即全窗口径。"""
    span = window_s - skip_s
    if span <= 0:
        return None
    in_win = [r for r in records if r.get("ok") and (r.get("t_end_rel") or 0) <= window_s]
    sel = [r for r in in_win if (r.get("t_end_rel") or 0) > skip_s]
    if not sel:
        return None
    toks = sum(int(r.get("completion_tokens") or 0) for r in sel)
    tpot = [r["tpot_s"] * 1000 for r in sel if r.get("tpot_s") is not None]
    ttft = [r["ttft_s"] * 1000 for r in sel if r.get("ttft_s") is not None]
    return {
        "n": len(sel), "span_s": round(span, 2),
        "throughput_tok_s": round(toks / span, 2),
        "tpot_p50_ms": pct(tpot, 50), "ttft_p50_ms": pct(ttft, 50),
        "failed": sum(1 for r in records if not r.get("ok")),
    }


def analyze_run(run_dir: Path, window_s: float, skip_s: float) -> dict | None:
    raw = run_dir / "raw"
    if not raw.is_dir():
        return None
    files = sorted(raw.glob("samples-*.jsonl"))
    if not files:
        return None

    full_thr, steady_thr, steady_tpot = [], [], []
    per_window = []
    for f in files:
        recs = load_window(f)
        if not recs:
            continue
        m_full = measure(recs, window_s, 0.0)
        m_std = measure(recs, window_s, skip_s)
        if not m_full or not m_std:
            continue
        full_thr.append(m_full["throughput_tok_s"])
        steady_thr.append(m_std["throughput_tok_s"])
        steady_tpot.append(m_std["tpot_p50_ms"])
        per_window.append({"file": f.name,
                           "full": m_full["throughput_tok_s"],
                           "steady": m_std["throughput_tok_s"],
                           "failed": m_full["failed"]})

    if not full_thr:
        return None
    s_full, s_std = stats(full_thr), stats(steady_thr)
    return {
        "run": run_dir.name, "windows": len(full_thr),
        "full": s_full, "steady": s_std,
        "steady_tpot_p50_ms": pct(steady_tpot, 50),
        "R_improve_pp": round(s_full["R_pct"] - s_std["R_pct"], 2),
        # v3 判据：只看全窗序列的 MAD/中位<10%，孤点单独列出
        "pass": s_full["pass"],
        "n_outliers": len(s_full["outliers"]),
        "outliers": s_full["outliers"],
        "per_window": per_window,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v3 稳态口径离线回算")
    ap.add_argument("--root", default=str(ROOT / "scripts" / "bench_out"))
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--steady-skip", type=float, default=0.0,
                    help="丢弃每窗开头 N 秒（诊断开关；已证无效：N=5–30 时 R 达标仍 2–3/18）。"
                         "默认 0.0 = 全窗口径")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    root = Path(args.root)
    if not root.is_absolute():
        root = ROOT / root
    rows = []
    for d in sorted(root.iterdir()):
        if d.is_dir():
            got = analyze_run(d, args.window_s, args.steady_skip)
            if got:
                rows.append(got)

    if not rows:
        print(f"NO_RAW_DATA under {root}")
        return 1

    print(f"离线回算：window={args.window_s:g}s  steady_skip={args.steady_skip:g}s  "
          f"（分母 {args.window_s - args.steady_skip:g}s）")
    print(f"root={root}")
    print("判据 v3：主判据 MAD/中位<10%；|x-中位|>3*MAD 记孤点（必须单独归因，"
          "不得平均或删除）；R 仅参考不作判据")
    print()
    show_steady = args.steady_skip > 0
    hdr = (f"{'run':<38}{'窗':>3}  {'中位':>8}{'R%':>8}{'IQR%':>8}{'MAD%':>8}"
           f"{'stdev%':>8}{'孤点':>4}  {'PASS':<5}")
    if show_steady:
        hdr += f"{'稳态R%':>9}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        f, s = r["full"], r["steady"]
        line = (f"{r['run']:<38}{r['windows']:>3}  "
                f"{f['median']:>8.2f}{f['R_pct']:>8.2f}{f['IQR_pct']:>8.2f}"
                f"{f['MAD_pct']:>8.2f}{f['stdev_pct']:>8.2f}{r['n_outliers']:>4}  "
                f"{'PASS' if r['pass'] else 'FAIL':<5}")
        if show_steady:
            line += f"{s['R_pct']:>9.2f}"
        print(line)

    # 汇总：4 种离散度的达标数与中位（与文档里的回算对照表同口径）
    all_full = [r["full"]["R_pct"] for r in rows]
    all_std = [r["steady"]["R_pct"] for r in rows]
    all_iqr = [r["full"]["IQR_pct"] for r in rows]
    all_mad = [r["full"]["MAD_pct"] for r in rows]
    all_sd = [r["full"]["stdev_pct"] for r in rows]

    def rate(xs, thr=10.0):
        return f"{sum(1 for x in xs if x < thr)}/{len(xs)}"

    print()
    print(f"PASS（按 MAD<10%）达标：{sum(1 for r in rows if r['pass'])}/{len(rows)}")
    r_line = f"R<10% 达标：{rate(all_full)}"
    if show_steady:
        r_line += f"  稳态(>{args.steady_skip:g}s) {rate(all_std)}"
    print(r_line)
    print(f"IQR<10% 达标：{rate(all_iqr)}    MAD<10% 达标：{rate(all_mad)}"
          f"    stdev<10% 达标：{rate(all_sd)}")
    print(f"离散度中位：R {statistics.median(all_full):.2f}%  "
          f"IQR {statistics.median(all_iqr):.2f}%  "
          f"MAD {statistics.median(all_mad):.2f}%  "
          f"stdev {statistics.median(all_sd):.2f}%")
    print(f"口径限定：MAD 达标率只说明典型窗间抖动约 "
          f"{statistics.median(all_mad):.2f}%，**不等于**引擎稳定；"
          f"孤点（{sum(r['n_outliers'] for r in rows)} 个）须回原始样本逐个归因。")

    if args.json:
        outp = Path(args.json)
        if not outp.is_absolute():
            outp = ROOT / outp
        outp.write_text(json.dumps({"window_s": args.window_s,
                                    "steady_skip_s": args.steady_skip,
                                    "runs": rows}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        print(f"\njson -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())