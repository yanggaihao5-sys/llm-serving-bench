# Measurement methodology

> The harness is the easy part. This document is the part that decides whether a number means
> anything.

Everything below was learned by getting it wrong first. Each rule is paired with the evidence that
produced it, so you can decide whether it applies to your setup.

---

## 1. Window accounting

A "window" is a fixed observation interval. Ours is 60 s, and the accounting rules are strict
because loose ones produce numbers that look precise and are not comparable:

| Element | Rule | Why |
|---|---|---|
| `t0` | the instant warm-up ends and dispatch begins | anchoring anywhere else makes the denominator drift |
| Numerator | requests with `t_finish ≤ t0 + window_s`, summed over `usage.completion_tokens` | per-request sums avoid sampling bias from streaming chunks |
| Cross-window requests | **excluded entirely** (not counted in either window) | splitting a request inflates whichever window it lands in |
| Failed requests | excluded from the numerator, counted separately | a 500 is not a fast response |
| Dispatch stop | no new requests after `t0 + window_s` | otherwise the tail of the window is denser than its head |
| Drain | in-flight requests finish but are not counted | otherwise the last window is systematically slower |

The denominator is always the nominal window length, never the observed span. A window that
dispatched less work is a slower window, not a shorter one.

### Dispatch modes

- **continuous** — `concurrency` workers, each re-issues the moment it finishes (closed loop).
- **barrier** — `concurrency` requests fired together; next batch waits for the last return.
- **open loop** — arrivals follow a fixed schedule (`k × interval`), independent of completions.

Closed loop and open loop answer different questions. Under closed loop, short requests finish
first and their workers re-issue sooner, so **the short-input share amplifies itself**; two input
mixes measured this way are not comparable. Use open loop whenever arrival mix is a variable.

---

## 2. Freeze the inputs

The prompt set is a file on disk with a sha256 recorded in a sidecar `.meta.json`. The loader
**refuses to run** if the file's hash does not match the metadata:

```
ValueError: prompt set hash drift: meta=51e1a484c481 file=0f3c9a12bb7d
```

This is not ceremony. A prompt set regenerated per round, or shuffled, or padded with one extra
line, silently changes what was measured, and the difference is usually smaller than the effect you
are looking for.

The shipped set is 32 short Chinese instructions (9–29 characters, median 17). It is deliberately
small, uniform and boring: the benchmark measures serving behaviour, not content quality.

---

## 3. Warm-up, steady state, and load history

**Throughput on this class of deployment is a function of how long it has been loaded**, not a
constant to be sampled. From one 8-accelerator deployment:

| Observation | Value |
|---|---|
| Climb from cold start to plateau | ~13 minutes |
| Windows needed to satisfy the stability gate | **≥16** (8 windows: MAD 11.49 %, verdict FAIL; 16 windows: MAD 4.13 %, PASS) |
| First window after restart | cold-compile artefact — discard it (TTFT p50 18.8 s) |
| Back-to-back comparison groups | later group measures **−2.5 %** for free (162.42 → 158.37, same config, no restart) |
| Recovery after ≥1 h idle | 165.62 (i.e. the loss was load accumulation, not time-of-day drift) |

Host-side checks during that −2.5 % loss ruled out thermal throttling (33–37 °C), disk (22 % used),
memory (918 GB available) and container restart. The cause is on the load side.

**Consequences for your protocol:**

1. Discard warm-up windows; report only the stationary segment.
2. Leave **≥30 min** between comparison groups, or the second group is handicapped by design.
3. Never compare a "just restarted" measurement to a "running for four days" measurement. We did
   this once and spent a day chasing a phantom 36 % regression that was a warm-up artefact.
4. `--warmup N` in this harness means **N warm-up requests**, not N windows. To obtain 16 warm-up
   windows, use `--repeats 16` and discard all of them.

---

## 4. Stability: MAD, not range

Three candidate stability statistics, evaluated offline on 18 historical runs (all windows, skip 0):

| Statistic | Runs passing `<10 %` | Median |
|---|---|---|
| `R = (max−min)/median` | 2 / 18 | 19.06 % |
| `stdev/median` | 12 / 18 | 6.79 % |
| `IQR/median` | 11 / 18 | 7.93 % |
| **`MAD/median`** | **17 / 18** | **4.47 %** |

A typical failing case: windows `169.95 / 147.88 / 166.75 / 173.63 / 167.93` → `R = 15.33 %`
(fail), but `MAD = 1.20 %` (pass). Four of five windows sit inside 166.75–173.63; the range
statistic is dominated by a single outlier. Range conflates *measurement noise* with a *real regime
change*, which are different problems requiring different responses.

Hence the gate: **`MAD/median < 10 %`**, with outliers (k = 3) reported individually and attributed
rather than silently trimmed. `R` is still printed — as a reference, never as a verdict.

---

## 5. Pair, don't subtract medians

Two configurations are measured in **interleaved pairs** (A→B, then B→A, …), and the analyse step
reports per-pair differences `d_i = X_B,i − X_A,i`, a median `d`, and a 95 % CI
(t = 2.776 for n = 5, 2.262 for n = 10):

- CI lower bound ≥ +10 % of the A-median → positive effect
- CI upper bound ≤ −10 % → negative effect
- **CI crossing 0 → "insufficient evidence"**, which is a result, not a failure

Subtracting two medians from different runs throws away the pairing, and with a 5–20 % window-to-
window spread it will manufacture effects that do not exist. In our own data the 10-pair median
difference was `+0.57 tok/s` with CI `[−3.04, +10.18]` — i.e. **the connection-reuse "optimisation"
was never demonstrated**, despite looking like a win in unpaired runs.

`insufficient evidence` must be written as such. It is not "no difference" and it is not "trending
towards".

---

## 6. Fingerprints: know which server you measured

Every run records the server's `system_fingerprint`. Compare fingerprints **before** comparing
numbers.

Three traps we hit:

1. **A fingerprint can change while behaviour does not.** The fingerprint hashes the whole config
   including `additional_config` keys that the engine never reads. Two runs with different
   fingerprints were, behaviourally, the same configuration — and vice versa is possible too. Use
   the fingerprint as an *identity check*, not as proof of effect (see FINDINGS #3).
2. **The fingerprint is the only trustworthy label.** One 21-minute benchmark was filed under the
   wrong configuration name; the fingerprint had said otherwise the whole time. Since then, any
   benchmark ≥15 min must carry its **expected fingerprint** in the run log and abort on mismatch.
3. **Restarts reset counters.** Prometheus counters (e.g. per-position speculative acceptance) do
   not survive a restart. If you need them, sample them *before* the restart, or accept that the
   quantity is permanently unmeasurable.

---

## 7. Validity gates, kept separate

Four independent booleans, never collapsed into one:

| Gate | Question | Failure mode it catches |
|---|---|---|
| `arrival_valid` | did requests actually arrive as planned? | client thread pool saturates; planned schedule silently becomes "as fast as possible" |
| `queue_exposure_valid` | did the server actually queue? | you think you measured scheduling, but `in_flight < max_num_seqs` the whole time |
| `comparison_valid` | were the two arms comparable? | different prompt hash, different fingerprint, different warm-up state |
| `overall_valid` | all of the above | — |

The arrival gate uses a plan hash that includes a sha256 of each prompt's content, not just its id
and length: same id + same length + different text must not collide.

The queue gate is the one most often skipped. A "priority scheduling" experiment where the client
never exceeds the server's batch limit measures nothing: there was never a queue to prioritise.
We shipped exactly such a run, marked it `INVALID`, and re-ran with `in_flight > max_num_seqs`.

---

## 8. Client-side versus server-side evidence

- Client-measured TTFT/TPOT/throughput are **end-to-end**: they include network, gateway and queueing
  time. That is usually what you care about, but it is not a statement about the engine.
- Server-side quantities (`num_requests_waiting`, KV cache usage) require the metrics endpoint.
  A gateway may proxy `/v1/*` and `/health` but **not** `/metrics` — check before you plan an
  experiment around it.
- A "no difference" result measured client-side at sub-millisecond scale is not evidence of no
  effect. Our noise floor at the time was 5.4 % on throughput and 21 % on short-request TTFT p95;
  a block-size effect of a few milliseconds at `c=1` is two orders of magnitude below that, and was
  correctly declared **not measurable** rather than reported as zero.

---

## 9. Invalid runs are part of the record

An invalid run is kept, not deleted: the directory gets an `INVALID.md` with the reason and the
run is excluded from statistics by rule.

Causes seen in practice: overlapping concurrent experiments, missing credentials (401 during warm-up),
wrong launch switch, mid-run termination, and cold-start ramp not finished. Each one would have
produced a plausible-looking number. Keeping them is what lets a later reader trust the ones that
survived.

---

## 10. Reporting checklist

Before calling a result a result:

- [ ] protocol id, window length, dispatch mode, concurrency, max tokens, temperature
- [ ] prompt set sha256 and count
- [ ] `system_fingerprint` for every arm
- [ ] warm-up windows discarded, count stated; MAD for the retained windows
- [ ] n per arm, median difference, 95 % CI, and the verdict that CI supports
- [ ] all four validity gates, individually
- [ ] invalid runs listed with reasons
- [ ] what the result does **not** establish
