from llmops.analyze import compare
from llmops.bench import percentiles, summarize


def test_percentiles():
    assert percentiles([1, 2, 3, 4])["p50"] == 2.0
    assert percentiles([])["p95"] == 0.0


def test_summarize_and_compare():
    samples = [{"ok": True, "ttft_s": 0.1, "tpot_s": 0.02, "completion_tokens": 10},
               {"ok": True, "ttft_s": 0.2, "tpot_s": 0.03, "completion_tokens": 20}]
    s = summarize(samples, 1.0)
    assert s["n"] == 2 and s["throughput_tok_s"] == 30.0
    assert compare({"throughput_tok_s": 100}, {"throughput_tok_s": 120},
                   "throughput_tok_s") == 20.0
