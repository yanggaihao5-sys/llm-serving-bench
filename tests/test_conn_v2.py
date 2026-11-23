
"""T2.1-CONN-v2 的假数据测试（CI 不碰真机）。"""
import socket
import time

import pytest

from llmops import conn_v2, protocol


def fake_transport(endpoint, model, text, max_tokens, api_key):
    t_send = time.perf_counter()
    time.sleep(0.005)
    t_end = time.perf_counter()
    return {"ok": True, "t_send": t_send, "t_first": t_end, "t_end": t_end,
            "completion_tokens": max_tokens, "prompt_tokens": 12, "cached_tokens": 0}


def test_abba_and_c_schedule():
    assert [conn_v2.abba_order(i) for i in range(1, 6)] == [
        ("A", "B"), ("B", "A"), ("A", "B"), ("B", "A"), ("A", "B")]
    assert [conn_v2.c_order(i) for i in range(1, 6)] == [
        ("B", "C"), ("C", "B"), ("B", "C"), ("C", "B"), ("B", "C")]


def test_paired_stats_positive():
    a = [100, 101, 99, 100, 100]
    b = [120, 121, 119, 120, 120]
    s = conn_v2.paired_stats(a, b)
    assert s["pairs"] == 5
    assert s["median_d"] == 20
    assert s["ci95"] == [20.0, 20.0]
    assert s["ci_pct_of_a"][0] >= 10
    assert conn_v2.verdict(s).startswith("正收益")


def test_paired_stats_negative():
    s = conn_v2.paired_stats([100] * 5, [80] * 5)
    assert s["median_d"] == -20
    assert s["ci_pct_of_a"][1] <= -10
    assert conn_v2.verdict(s).startswith("负收益")


def test_paired_stats_inconclusive():
    a = [100] * 5
    b = [101, 99, 101, 99, 101]
    s = conn_v2.paired_stats(a, b)
    assert s["ci95"][0] < 0 < s["ci95"][1]
    assert "证据不足" in conn_v2.verdict(s)


def test_paired_stats_rejects_length_mismatch():
    with pytest.raises(ValueError):
        conn_v2.paired_stats([100], [100, 101])


def test_connection_counter_counts_new_socket():
    conn_v2.install_connection_counter()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    try:
        conn_v2.reset_connection_counter()
        c = socket.create_connection(("127.0.0.1", srv.getsockname()[1]), timeout=2)
        assert conn_v2.connection_count() == 1
        c.close()
    finally:
        srv.close()


def test_pair_stats_merged_rounds():
    groups = [{"round": 1, "group": "A", "throughput_tok_s": 100},
              {"round": 1, "group": "B", "throughput_tok_s": 120},
              {"round": 2, "group": "B", "throughput_tok_s": 110},
              {"round": 2, "group": "A", "throughput_tok_s": 100}]
    s = conn_v2.pair_stats(groups, ("A", "B"))
    assert s["pairs"] == 2
    assert s["used_rounds"] == [1, 2]
    assert s["d_values"] == [20.0, 10.0]


def test_load_previous_groups_offsets(tmp_path):
    d = tmp_path / "run1"
    d.mkdir()
    import json
    (d / "bench.json").write_text(json.dumps({"groups": [
        {"round": 1, "group": "A", "throughput_tok_s": 100},
        {"round": 1, "group": "B", "throughput_tok_s": 120}]}), encoding="utf-8")
    groups, src = conn_v2.load_previous_groups(d)
    assert [g["round"] for g in groups] == [2, 2]
    assert src.endswith("run1")


def test_group_window_with_fake_transport(monkeypatch):
    ps = protocol.load_prompt_set()
    monkeypatch.setitem(conn_v2.TRANSPORTS, "B", fake_transport)
    monkeypatch.setattr(conn_v2, "_netstat_estab", lambda target="x": 7)
    res = conn_v2.run_group_window("B", ps["prompts"], "e", "m", "", 4, 64, 0.3, 2,
                                   estab_interval=0.05)
    assert res["summary"]["n"] > 0
    assert res["summary"]["failed"] == 0
    assert res["new_connections"] == 0
    assert res["peak_established"] == 7
    assert res["warmup"]["ok"] == 2
    assert all(r["ttft_s"] is not None for r in res["records"])
