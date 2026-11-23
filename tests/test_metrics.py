from llmops.metrics import parse_nvidia_smi, parse_prometheus, records


def test_parse_prometheus():
    text = "# HELP x\nllamacpp:tokens_predicted_total 10\nserve/ttft_ms 1.5\n"
    assert parse_prometheus(text)["llamacpp:tokens_predicted_total"] == 10.0
    assert parse_prometheus(text)["serve/ttft_ms"] == 1.5


def test_parse_nvidia():
    text = "0, 25966 MiB, 32607 MiB, 27 %\n1, 9248 MiB, 32607 MiB, 2 %\n"
    rows = parse_nvidia_smi(text)
    assert rows[0]["index"] == 0 and rows[0]["mem_used_mb"] == 25966.0
    assert rows[1]["util_pct"] == 2.0


def test_records_prefix():
    recs = records({"m": 1.0}, [{"index": 0, "mem_used_mb": 1, "mem_total_mb": 2,
                                  "util_pct": 3}], ts=5)
    names = {r["name"] for r in recs}
    assert "serve/m" in names and "gpu/0/util_pct" in names
