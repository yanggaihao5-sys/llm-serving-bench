# llm-serving-bench

A small, dependency-free harness for **reproducible LLM serving benchmarks** — plus the
measurement discipline that makes the numbers trustworthy.

Most serving benchmarks answer *"how fast is it?"* and stop there. This repo is built around a
harder question: **"is the difference real?"** It grew out of a long series of A/B experiments on
an 8-accelerator production deployment, where the dominant error source turned out to be the
measurement protocol, not the server.

```
llmops/        harness: protocol, window accounting, metrics, paired A/B, re-analysis
tests/         unit tests for the accounting rules (no network, CI-safe)
bench/         frozen prompt set + hash metadata
docs/          METHODOLOGY.md · FINDINGS.md
```

## What it gives you

| Capability | Where |
|---|---|
| Frozen, hash-verified prompt set (refuses to run on drift) | `llmops/protocol.py` |
| Continuous / barrier / open-loop dispatch with strict 60 s window accounting | `llmops/frozen_bench.py` |
| TTFT / TPOT / throughput / error rate from streamed responses | `llmops/metrics.py` |
| Paired A/BBA experiments with 95 % CI and effect-size reporting | `llmops/conn_v2.py` |
| Configuration fingerprinting (`system_fingerprint`) for every run | `llmops/frozen_bench.py` |
| Three-layer validity gates (arrival / queue exposure / comparison) | `llmops/frozen_bench.py` |
| Offline re-analysis of historical raw samples | `llmops/reanalyze.py` |
| The write-ups: nine findings + the protocol behind them | `docs/` |

Pure standard library — no numpy, no requests, no pandas. It runs anywhere Python 3.10+ runs.

## Quick start

```bash
pip install -e .            # or just: python -m pytest -q
python -m pytest -q         # 62 tests, no network access

# dry-run the whole pipeline with fake transports — no server needed
python -m llmops.frozen_bench --fake --mode continuous --repeats 2
```

Against a real server:

```bash
export LLM_BENCH_BASE_URL="http://<host>:<port>"
export LLM_BENCH_MODEL="<served-model-id>"
export VLLM_API_KEY="<key-if-your-gateway-needs-one>"

python -m llmops.frozen_bench --mode continuous \
    --concurrency 32 --max-tokens 64 --window-s 60 --repeats 5 --warmup 8
```

Every run writes a self-describing directory:

```
runs/<run_id>/
  run.json        target / engine / endpoint / model / protocol snapshot / fingerprint
  metrics.jsonl   {ts, step, name, value} — one line per sample
  events.jsonl    start / stop / abort / OOM
  bench.json      per-window TTFT-TPOT-throughput, aggregate, stability verdict, gates
```

Reports should cite `run_id` only; the directory is the evidence.

## The measurement discipline

Short version — full version in **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

1. **Freeze the inputs.** Prompt set is a file, read-only, sha256-verified. A drifting prompt set
   silently changes what you measured.
2. **Name the configuration.** Every run records the server's `system_fingerprint`. If two runs
   disagree, compare fingerprints *first* — a different fingerprint means a different server.
3. **Warm up until the metric is stationary.** On the deployment studied here, throughput climbs
   for ~13 minutes after a restart and needs **≥16 windows** to stabilise — 8 windows is not enough
   (MAD 11.5 %, verdict FAIL). Consecutive load also degrades the machine; leave ≥30 min of idle
   time between comparison groups.
4. **Use MAD, not range.** `R = (max−min)/median` fails 16 of 18 historical runs on a single
   outlier; `MAD/median < 10 %` passes 17 of 18 on the same data. Range conflates noise with a real
   regime change.
5. **Pair, don't compare medians.** A/B conditions are interleaved (A→B, B→A, …) and reported as
   per-pair differences with a 95 % CI — never as two medians subtracted.
6. **Gate validity separately.** `arrival_valid`, `queue_exposure_valid`, `comparison_valid`,
   `overall_valid` are four independent booleans. Collapsing them into one "OK" hides exactly the
   failure that invalidates the conclusion.
7. **Record invalid runs.** A run that fails a gate is marked `INVALID` with the reason, kept on
   disk, and excluded from statistics — never silently deleted.

## Findings

Nine results, each with numbers and an explicit "what this does *not* show" — see
**[docs/FINDINGS.md](docs/FINDINGS.md)**. Headlines:

| # | Finding | Direction |
|---|---|---|
| 1 | KV block size 32 vs 64 | 64 is **−19 %** short-request throughput, zero gain on long inputs |
| 2 | Speculative decoding depth (K sweep) | TPOT improves but **TTFT eats it**: break-even ≈ 634 output tokens |
| 3 | A "tuning knob" that was never wired | `expert_capacity_factor` 1.3 vs 1.5: **zero** effect across 7 groups |
| 4 | Batch-width trade-off | Peak throughput **+36 %**, per-stream TPOT **+16 %** |
| 5 | Strict priority scheduling | Long-request first-token wait **+128 %**, short requests gain nothing |
| 6 | Version regression caught by protocol | Throughput flat, **TTFT ×6** — reproducible 3/3, rolled back |
| 7 | Steady state is a function of load history | Back-to-back runs measure **−2.5 %** for free |
| 8 | Per-card imbalance | MAX/MIN ≈ 2.0, same card last; HCCS↔utilisation ρ = +0.33 (n.s.) |
| 9 | Long-context behaviour | Decode flat to 200 K; prefill **convex** (4.5–6.4 K tok/s) |

## Scope and honesty

- These are **single-cluster, single-model** measurements. The *directions* reproduce; the absolute
  tok/s values do not transfer to other hardware.
- Numbers are reported with their protocol attached. Any figure quoted without its window,
  warm-up state and fingerprint should be treated as anecdotal — including ours.
- Each finding states what it cannot establish. Where a mechanism was never measured, it says so
  (`Attribution Pending`), rather than offering a plausible story.

## License

MIT — see [LICENSE](LICENSE).
