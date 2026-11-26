# Findings

Nine results from a long series of serving experiments on one production deployment. Each entry
gives the question, the setup, the numbers, the interpretation, and — deliberately — **what the
result does not establish**.

**Deployment under study (common to all findings unless stated):** 8 accelerators, 64 GB each,
fully interconnected; one production MoE LLM served with tensor parallelism 8 + expert parallelism
(TP8·EP), W8A8 weights; vLLM 0.25.1 + vllm-ascend plugin 0.19.1rc2.dev1265+g50e0ce608 (2026-11-26 取证更正 C5-F01：原文「vLLM-Ascend 0.25.1」把主程序版本与插件版本混置). Benchmarks are client-side, end-to-end, with the frozen
protocol in [METHODOLOGY.md](METHODOLOGY.md).

Absolute throughput values are specific to this machine. The **directions** and the **measurement
lessons** are what transfer.

---

## 1. KV block size: 64 costs 19 % on short requests and buys nothing on long ones

**Question.** Does a larger KV block size reduce page-table overhead enough to matter?

**Setup.** Two block sizes, everything else fixed (32 slots, 8192 batched tokens, priority
scheduling). Short-request high concurrency: 32 clients, 64 output tokens, 60 s windows, 5 windows,
8-request warm-up. Long context: 64 K and 200 K inputs, 256 output tokens, 5 repetitions each.

| Workload | block-size 32 | block-size 64 | Δ |
|---|---|---|---|
| Short requests, c=32 (tok/s) | 167.77 · 164.70 | **134.20 · 135.75** | **−19 %** |
| 64 K prefill TTFT p50 | — | — | −0.5 % |
| 200 K prefill TTFT p50 | — | — | +1.5 % |

The two block-size-64 runs were 25 minutes apart and agreed to 1.2 %, with one window discarded as
a warm-up outlier — so this is a stable property, not a transient.

**Interpretation.** Halving the number of pages did not help any measured path, and actively hurt
short-request throughput. **[Corrected 2026-11-25]** The engine's documentation carries no
`block_size` guidance at all (zero mentions in the optimization pages, independently re-verified), so
the original draft's claim of a contradicting "official suggestion" was wrong — there was nothing to
contradict. The measured −19 % stands on its own. **We did not establish why**, and
the block-size-32 baseline was collected the previous day, so the cross-day comparison is labelled
*observed difference — attribution pending* rather than a clean A/B.

**Decision.** Keep 32. The cost is measurable, the benefit is not.

**What this does not show.** It does not show that 64 is universally worse — only that on this
deployment, with this engine build and these workloads, it is. A same-day paired A/B would be
needed to call it causal.

---

## 2. Speculative decoding: TPOT improves, TTFT eats the gain

**Question.** Does a deeper speculative draft improve serving?

**Setup.** Draft depth K ∈ {3, 5, 7}, sweep at `c=1` over 8 K / 64 K / 200 K inputs, 256 output
tokens, plus `c=32` short-request throughput.

| Draft depth | TPOT p50 (8K/64K/200K) | vs K=5 | 200 K E2E (256 out) |
|---|---|---|---|
| K = 3 | 13.01 / 12.52 / 12.70 ms | **+28 % / +22 % / +30 %** | — |
| K = 5 | 10.14 / 10.24 / 9.80 ms | baseline | **39.42 s** |
| K = 7 | 9.87 / 8.60 / 8.52 ms | **−2.7 % / −16 % / −13 %** | **40.22 s** |

K = 7 makes decode measurably faster — per-request TPOT improves by up to 16 %, and the 200 K
per-repetition values (8.1–9.0 ms) do not overlap the K = 5 values (9.49–10.54 ms). And yet the
end-to-end time for the same request **got worse**: +0.8 s, because first-token latency rose by
1.044 s.

**Interpretation.** A deeper draft costs more prefill-side work per request. The TTFT penalty is a
fixed additive cost; the TPOT gain is proportional to output length. They cross at roughly
**634 output tokens** — below that, K = 7 loses; above it, it wins. Benchmarking only TPOT (the
usual shortcut) would have selected the configuration that is worse for typical short requests.
At `c=32` the sign even flips: K = 7 measured 159.22 tok/s against 164.70–167.77 for K = 5.

**Decision.** Keep K = 5 as the production default; keep K = 7 as a long-output candidate, not a
global default.

**What this does not show.** The acceptance-length mechanism is **indirect**: tokens-per-streamed-
chunk rose from ≈5.5 to 6.0–6.1, but chunk count is not decode-iteration count, and the per-position
acceptance counters were lost to two restarts. The `c=32` K = 7 run was also preceded by 25 minutes
of sustained load, so load accumulation is not fully separated from the effect.

---

## 3. A tuning knob that was never connected

**Question.** Does raising MoE expert capacity factor from 1.3 to 1.5 reduce expert padding and
improve throughput?

**Setup.** Two values, seven comparable measurement groups: five long-input levels (8 K → 200 K),
`c=32` short-request throughput, and `c=1` TPOT.

| Metric | 1.3 | 1.5 | Δ |
|---|---|---|---|
| TPOT p50, 5 input levels | 10.14 / 9.45 / 9.48 / 10.24 / 9.80 ms | 11.13 / 9.72 / 9.60 / 9.09 / 10.70 ms | within noise |
| Throughput, c=32 | 167.75 | 167.17 | **−0.35 %** |
| TPOT, c=1 | 19.3 ms | 19.2 ms | −0.5 % |

Seven groups, both directions, all inside the noise band.

**Interpretation.** The configuration key is **never consumed** by this engine build: it is
recorded, hashed, and ignored. The two configurations had *different* `system_fingerprint` values
purely because `additional_config` participates in the fingerprint hash — which is exactly the trap
described in METHODOLOGY §6. Two weeks of A/B work were, in behaviour, comparing a configuration
with itself. A separate key in the same block *was* live and *did* change behaviour, which is why
the fingerprint difference looked credible.

**Decision.** Revert to the original 1.3 (no behavioural difference; 1.3 is the pre-tuning
baseline), and treat "the config changed" as a hypothesis to be verified, never as evidence.

**What this does not show.** It does not tell you how to detect this class of problem in advance.
The only reliable check found was to read the engine's own parsed startup arguments and its warning
lines — i.e. verify consumption, not presence.

---

## 4. Batch width: +36 % aggregate throughput, +16 % per-stream TPOT

**Question.** Should the server admit more concurrent sequences (32 → 40)?

**Setup.** `max_num_seqs` 32 vs 40, 11 concurrency levels scanned, 20 s windows, hot state, then a
paired single-stream comparison at the same `c=32`.

| Concurrency | 16 | 32 | 48 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| tok/s (40 slots) | 91.05 | 140.50 | 173.95 | **198.10** | 193.15 | 189.95 |

Peak throughput rose from 145.6 to 198.10 tok/s (**+36.1 %**), and the knee moved from `c=32` to
`c=64`. But the paired single-stream comparison at `c=32` moved the other way:

| | 32 slots | 40 slots | Δ |
|---|---|---|---|
| Single-stream tok/s | 5.36 | 4.80 | **−10.4 %** |
| TTFT p50 | 1635 ms | 1861 ms | **+13.8 %** |
| TPOT p50 | 150.4 ms | 174.5 ms | **+16.0 %** |

**Interpretation.** Raising the admission limit buys *more users served at once*, paid for by
*each user being slower*. Cross-validation: with 40 slots, a single stream at `c=40` gets 4.84 tok/s
— the same as its `c=32` value — i.e. "whether 32 or 40 users arrive, each gets ~4.8 tok/s".

**Decision.** Keep 32, as an explicit product decision (protect per-user latency), not because it
is faster in aggregate.

**What this does not show.** The +16 % TPOT penalty has **no confirmed mechanism**. Expert-padding
was ruled out (the captured batch-size set includes 32), and a draft-batch explanation for the
speculative path remains unproven. Short-window absolute values are also systematically
underestimated (`drained == concurrency`), so the peak location — not the magnitude — is the
quotable part.

---

## 5. Strict priority scheduling: short requests gain nothing, long requests pay 2.3×

**Question.** With 8:1 short:long mixed arrivals, does strict priority scheduling protect short
requests?

**Setup.** 10 interleaved rounds (priority off / on), 40 in-flight clients against 40 server slots
(server `max_num_seqs` was 40 at run time, per 实验登记表 and C-118; 11-19 才定格 32 — 2026-11-26 取证更正 C5-F08)
so that a queue genuinely forms, 8:1 short:long arrival mix, ≥2 h apart from other load.

| Metric | priority off | priority on | Δ |
|---|---|---|---|
| Long-request first-token wait (max) | 5938 ms | **13545 ms** | **+128.1 %** (5/5 pairs positive) |
| Drain time | — | — | **+18.2 %** |
| Short-request TTFT p95 | — | — | **+8.2 %** (no improvement) |
| Completion rate | 1.0 | 1.0 | — |

**Interpretation.** Under overload, strict priority does not make short requests faster — it was
already serving them first by arrival order. What it does is *displace* queued long requests, and
with zero nominal slot headroom (40 in-flight − 40 slots; the queue forms from the 67-request burst
per 20 s window, not from slot oversubscription) each displaced request waits
proportionally longer. Second-order finding: this engine implements strict priority **without a
`long_max_wait_ms` bound**, so a long request can be deferred indefinitely under sustained
overload.

**Decision.** Enabling priority is only defensible in (a) non-overloaded conditions, or (b)
overload *with* admission control. "priority on" ≠ "good for both classes".

**What this does not show.** Server-side `num_requests_waiting` was not sampled during the pulse
rounds, so the +128 % is a correlation with priority that is stable at 6/6 — not a proven causal
attribution. The admission-control branch of the decision was never validated and must not be
cited.

---

## 6. A version upgrade with unchanged throughput and 6× worse TTFT

**Question.** Does the new engine version improve serving?

**Setup.** Same deployment, same protocol, three independent measurement rounds per version.

| Metric | baseline version | new version (3 rounds) | Δ |
|---|---|---|---|
| Throughput, c=32 | 172.65 tok/s | 172.58 / 168.23 / 172.42 | **−0.13 %** |
| TTFT p50 | 1706 ms | 10134 / 10413 / 10277 ms | **+502 %** |
| TPOT p50 | 147.6 ms | 62.4 / 63.5 / 60.6 ms | −58 % |
| 7 K workload throughput | 538.03 tok/s | 301.04 / 301.71 | **−44 %** |
| Streaming chunks | 211–260 | **0** | broken |

**Interpretation.** Aggregate throughput is unchanged, which is exactly why a throughput-only
benchmark would have shipped this. Per-token decode got 2.4× faster while first-token latency got
5–6× worse — a fan-out/queueing pathology — so end-to-end latency for any request that is not
long-output became worse. The 7 K workload also showed the streaming interface returning zero
chunks, i.e. the response was effectively non-streaming.

**Decision.** Roll back. Rollback verified at 165.62 tok/s with TTFT 1635 ms.

**What this does not show.** The mixed cause is not fully separated: only one config key differed,
and that key plausibly *hurts* the new version, so it cannot explain the 2.4× TPOT speedup. The
honest statement is "reproducible behaviour change, mechanism partially attributed".

---

## 7. Steady state is a function of load history, not of the clock

**Question.** Why do repeated measurements of the "same" configuration disagree by 10–35 %?

**Setup.** Same container, same config, same fingerprint; three consecutive benchmark rounds, then
one round after ≥1 h idle.

| Round | Throughput |
|---|---|
| A2 | 162.42 tok/s |
| A2b (immediately after) | 158.37 tok/s (**−2.5 %**) |
| A2c (after 1 h idle) | **165.62 tok/s** |

Cold-start ramp measured separately: 101.75 tok/s in the first window, climbing to ~181 tok/s over
roughly 13 minutes.

**Interpretation.** The machine is not in a fixed state; it is in a *load-history-dependent* state.
Consecutive groups penalise the later group, and a "the service runs 36 % slower after four days"
story was traced back to a warm-up artefact rather than degradation. Host-side checks (temperature,
disk, memory, container restarts) ruled out hardware causes.

**Decision.** ≥30 min idle between comparison groups; discard warm-up windows; take the stationary
segment only.

**What this does not show.** The physical mechanism of accumulation (clocks? allocator? page
cache?) was never identified — only its existence, magnitude and decay time.

---

## 8. Per-card imbalance: the same card is slow, and bandwidth does not explain it

**Question.** Is the deployment limited by its slowest accelerator, and is communication the cause?

**Setup.** Three `c=32` observations with per-card utilisation sampled in the same window, plus
PCIe/HCCS topology and per-card bandwidth counters.

| Observation | MAX/MIN utilisation | Slowest card |
|---|---|---|
| 1 | 2.314 | card 4 |
| 2 | 1.948 | (rotated) |
| 3 | 2.085 | card 4 |

**Interpretation.** A ~2× spread between cards, with the same card last in two of three runs. The
obvious hypothesis — "the card with the least interconnect bandwidth is compute-starved" — is
**refuted by the data**: the card with the lowest HCCS bandwidth had the *highest* utilisation, and
the rank correlation between bandwidth and utilisation was **+0.33 (n = 8, not significant)**. The
topology is fully interconnected (no PCIe fallback), and profiling showed no per-layer EP
communication domain to blame.

**Decision / next step.** The cheapest discriminating experiment is to change the prompt set so a
*different* card becomes the slowest: if the slow card moves, it is routing; if `card 4` stays last,
it is structural.

**What this does not show.** The mechanism is unresolved. Seven candidate explanations were
eliminated; the surviving one ("structural card difference") is a label for ignorance, not an
explanation. Also: `AI Core` utilisation is only usable as a *relative* comparison — with profiling
active it reads 0 for 800/800 samples.

---

## 9. Long context: decode is flat, prefill is convex

**Question.** How does serving degrade from 8 K to 200 K input tokens?

**Setup.** 8 K / 16 K / 32 K / 64 K / 200 K inputs, 256 output tokens, `c=1`, same fingerprint
throughout (`cached_tokens = 0`, so no prefix-cache contamination).

| Input | TTFT p50 | TPOT p50 | Effective prefill rate |
|---|---|---|---|
| 8 K | 1.801 s | 10.14 ms | 4.53 K tok/s |
| 16 K | 2.703 s | 9.45 ms | 6.03 K tok/s |
| 32 K | 5.097 s | 9.48 ms | 6.39 K tok/s |
| 64 K | 10.237 s | 10.24 ms | 6.37 K tok/s |
| 200 K | 36.882 s | 9.80 ms | 5.39 K tok/s |

**Interpretation.** Decode throughput does **not** collapse at long context: all five TPOT values
sit in 9.4–10.2 ms, and the between-level spread (0.79 ms) is smaller than the within-level range
(1.02–1.05 ms) — i.e. not statistically distinguishable. Prefill, by contrast, is convex: it ramps
up to a plateau around 32–64 K, then drops ~21 % at 200 K.

**Decision.** Long-context cost is a **prefill** problem, not a decode problem. Any plan that
attacks long-context decode (KV formats, quantisation) is aimed at the wrong term.

**What this does not show.** The 8 K point is anomalous (0.44 s slower than interpolation, sitting
exactly at the 8192 batched-token boundary) and was never explained; and the 200 K value is the
median of five repetitions whose raw artefacts were lost, so it cannot be independently recomputed.

---

## Cross-cutting lesson

The findings above are the interesting part. The reason they are trustworthy is duller: every one
of them was measured under a protocol that had already been *wrong* several times — a phantom 36 %
regression, a connection-reuse "win" that vanished under pairing, a two-week A/B on an unwired
config key, a priority experiment with no queue, and a run filed under the wrong configuration
name. The protocol in [METHODOLOGY.md](METHODOLOGY.md) is the list of ways it broke.
