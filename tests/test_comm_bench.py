"""T2-COMM-7K1K 执行器的假数据自检（不碰网络）。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llmops import comm_bench as cb  # noqa: E402


def fake_send(endpoint, model, text, max_tokens, api_key):
    t0 = 1000.0
    return {"ok": True, "t_send": t0, "t_first": t0 + 0.5, "t_end": t0 + 0.5 + max_tokens * 0.02,
            "completion_tokens": max_tokens, "prompt_tokens": 7000, "cached_tokens": 0,
            "chunks": 1, "finish_reason": "length"}


def test_wave_throughput_is_tokens_over_wall():
    prompts = [{"id": f"p{i}", "text": "x"} for i in range(4)]
    wave = cb.run_wave(prompts, fake_send, "e", "m", "", 4, 100)
    s = cb.summarize_wave(wave, 60.0)
    assert s["ok"] == 4 and s["failed"] == 0
    assert s["completion_tokens"] == 400
    # wall ≈ drain 完成时刻 - 首条发送时刻
    assert abs(s["throughput_tok_s"] - 400 / s["wall_s"]) < 0.5
    assert s["prompt_tokens"]["median"] == 7000
    assert s["cached_tokens"] == 0
    assert s["acceptance_rate"] is None  # 不估算 DSpark 接受率


def test_window_secondary_counts_only_completed_inside_window():
    recs = [
        {"ok": True, "t_send": 0.0, "t_first": 0.2, "t_end": 10.0,
         "completion_tokens": 100, "prompt_tokens": 7000, "cached_tokens": 0,
         "t_start": 0.0, "t_end_rel": 10.0, "ttft_s": 0.2, "e2e_s": 10.0, "tpot_s": 0.098},
        {"ok": True, "t_send": 0.0, "t_first": 0.2, "t_end": 90.0,
         "completion_tokens": 100, "prompt_tokens": 7000, "cached_tokens": 0,
         "t_start": 0.0, "t_end_rel": 90.0, "ttft_s": 0.2, "e2e_s": 90.0, "tpot_s": 0.898},
    ]
    s = cb.summarize_wave({"records": recs, "wall_s": 90.0}, 60.0)
    assert s["window_60s"]["n"] == 1
    assert s["window_60s"]["completion_tokens"] == 100
    assert s["throughput_tok_s"] == round(200 / 90.0, 2)


def test_gate_treats_missing_lease_as_free(monkeypatch):
    monkeypatch.setattr(cb, "LOCK", ROOT / "no-such-lease.json")
    assert cb.lease_holder() is None