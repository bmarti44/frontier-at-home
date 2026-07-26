# GLM-5.2-on-Spark: authoritative claims register

Every claim from this workstream, restated in the strongest form the evidence
actually supports. Where an earlier statement was too strong, the original is
shown struck and the reason recorded. This supersedes prose in older notes.

Status vocabulary:
- **ESTABLISHED** — measured with a valid same-binary control and repeats.
- **SUPPORTED** — measured, but with a named limitation (n, single fixture, an
  uncontrolled variable). Usable, not shippable as a general claim.
- **OPEN** — mechanism identified, not established.
- **RETRACTED** — was claimed, no longer supported.

Last updated 2026-07-26.

---

## Performance

### C1. Expert skipping — MEASURED, REJECTED (fidelity)
**Final decision: not adopted at any level.** The fidelity gate (C2) reports
keep-6 ΔNLL +0.0799 nat/token, 95% CI [+0.0135, +0.1463] — 8x the 0.01
threshold with the CI excluding zero — for ~11% decode. keep-7 is inconclusive
(ΔNLL +0.0169, CI [-0.0397, +0.0735] spans zero) and its speed benefit is not
established: the trajectory-controlled run gave -2.0%, against +10%
free-running. `52_engine_switch.sh` sets neither knob.
The speed measurements below stand; they are simply not worth their cost.

#### Speed as measured — SUPPORTED
`DS4_GLM_TOPK_KEEP=N` + `DS4_GLM_TOPK_SKIP_LOAD=1`:
keep-7 **+10.0%**, keep-6 **+13.0%** decode vs a same-binary keep-8 control
(2.293 t/s), n=4 per arm, ABBA across two passes.
Byte reduction independently verified: `unique=8/7/6` on **all 23,400** decode
loads per arm; 74.2 / 65.0 / 55.7 MiB per routed layer; cache misses
55,132 / 45,348 / 39,134.
**Limitations (sol):** one prompt; the four values are two measurements inside
each of two server runs, not four independent samples; and keep-6 produces a
*different token trajectory*, so expert-routing locality is not controlled.
The trajectory-controlled measurement is the teacher-forced NLL pass.
**Not shippable until** the fidelity gate (C2) reports.

~~"Expert skipping is dead on GLM-5.2"~~ — RETRACTED. That came from a probe
that ablated contributions while still loading all 8 experts, with a control
from a *different* configuration, under a `batchall` flag later shown to
corrupt short-prompt output on its own.

### C2. Fidelity cost of expert skipping — MEASURED, GATE FAILED
Coherence, valid UTF-8 and low 3-gram repetition are **liveness checks, not
fidelity**; they cannot see factual drift, reasoning collapse or wrong code.
Ran: paired teacher-forced NLL, `glm52-openrouter-100`, 16 of 100 cases,
identical subset in all three arms, scorer relinked against current objects.

| arm | mean NLL | first-token | ΔNLL vs keep-8 | 95% CI |
|---|---|---|---|---|
| keep-8 | 0.5267 | 12/16 | — | — |
| keep-7 | 0.5436 | 11/16 | +0.0169 | [-0.0397, +0.0735] |
| keep-6 | 0.6066 | 12/16 | +0.0799 | [+0.0135, +0.1463] |

**keep-6 FAILS** (8x the 0.01 threshold, CI excludes zero, worse on 11/16
cases). **keep-7 is INCONCLUSIVE** — point estimate 2x over, CI spans zero.

The reason the earlier checks missed this: first-token agreement is *identical*
(12/16) between keep-8 and keep-6, and both produce clean coherent English. The
damage is distributed across the probability mass, invisible at the argmax.
That is sol's "coherence is liveness, not fidelity" objection, demonstrated.

### C3. Warm TTFT 1.755 s — SUPPORTED, with a dependency
Measured cold 147.6 s / warm1 2.378 s / warm2 1.755 s, identical output sha
`b344d80e24a3`, zero invalid UTF-8.
**Dependency discovered 2026-07-26:** this requires the disk-KV checkpoint
configuration (`--kv-disk-dir` plus the boundary align/trim flags). An
identical replay against the same live server *without* those flags
re-prefilled in **147 s**. Any benchmark omitting them must not quote a
warm-TTFT number.
**Limitation (sol):** it is a third-identical-request measurement with
`max_tokens=1`; p50/p95 over many fresh servers has not been collected.

### C4. Expert-cache flush storm — mechanism ESTABLISHED, speed benefit RETRACTED
`g_model_load_generation++` ran before the "same map" early-return in all three
model-map setters, so every prefill span install looked like a model reload and
flushed the whole 72 GB expert cache: **452 flushes** in one long-prompt run,
hit rate 9.3%, resident slots collapsing to 245/7398.
Fix verified: **452 → 0 flushes**.
~~"This is the biggest remaining decode lever"~~ and sol's pre-registered
"20–40% speedup" — **both FALSIFIED by measurement**: cold TTFT 158.1 → 163.2 s,
second prefill 147.4 → 143.9 s, decode 1.84 → 1.79 t/s, hit rate only
9.3 → 12.6%. A prefill chunk sweeps ~253 distinct experts per routed layer
(~170 GiB across 75 layers, measured 339 GiB over two sweeps) against a 68 GB
cache — about 2.5x — so it thrashes regardless of flushing. Scoped per sol:
**no measurable benefit observed on this fixture**, not a universal null.
**Kept as a cache-invalidation/lifecycle fix only** (sol notes it is not an
inference-correctness fix: the old behaviour was output-safe but wasteful) — a cache that silently discards
itself is a latent hazard, and the generation doubles as the prefetch pool's
validity token.

### C5. Prefill chunk size — ABANDONED, the lever does not exist
Hypothesis was: each chunk pays the full ~170 GiB expert sweep regardless of
token count, so making a 5047-token prompt one chunk instead of two would halve
prefill traffic.
**Falsified mechanically.** GLM bypasses `ds4_prefill_cap_for_prompt()` and uses
the indexed-prefill path, where `glm_graph_limit_indexed_prefill_chunk()` clamps
the chunk to `n_indexer_top_k - pos` = 2048 and the size comes from the
compile-time `DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS`. Neither
`DS4_METAL_PREFILL_CHUNK` nor `--prefill-chunk` reaches it — the observed
2048 + 2999 split is structural.
The first attempt also produced **no result at all**: both arms ran
`prefill_chunk=4096` (identical sweeps=2 and prefill bytes to one decimal),
the same "arms were secretly identical" failure as the flush A/B.
sol pre-registered the outcome before the run and matched it exactly: "two
sweeps, ~339 GiB, identical output, causal TTFT change exactly 0%".
A real version would need the indexed workspaces resized and the top-k bridge
redesigned, with a fidelity gate because merging batches changes FP results.
**Not attempted.**

### C6. Levers measured and NOT adopted
- **MTP speculation**: ~+10%, output not identical. One non-repeated
  OFF-then-ON observation; the "1.5× unique experts" explanation is
  contradicted by the committed counters (1.30×). Directional only.
- **Cross-layer expert prefetch**: net-neutral on decode wall. The "46% fewer
  misses" framing is RETRACTED — the miss counter increments before the
  prefetch claim, so prefetched experts still count as misses that were merely
  served faster.
- **`DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL` (batchall)**: **REMOVED as a
  fidelity hazard** — produces repetition loops and invalid UTF-8 on short
  prompts. It also invalidated every result that had used it.

### C7. DSV4 parity — NOT MET, structurally
DSV4 bar: 467 t/s prefill, 18.4 t/s decode, <2 s warm TTFT.
GLM-5.2 streaming: ~23 t/s prefill, ~2.3–2.6 t/s decode, 1.755 s warm TTFT.
Warm TTFT **meets** the bar. Decode is ~8× short and prefill ~20× short.
Independent arithmetic (sol): 74.2 MiB × 75 layers = 5.8 GB of expert weights
per decode token; at 10.7 GB/s the all-miss floor is ~0.55 s/token = **1.83
t/s**. The software levers found are worth tens of percent, not 8×. Closing
the gap requires weights that fit in memory, not a faster streaming path.

---

## Compression (closed)

### C8. Lossless compression of IQ2_XXS — RETRACTED and restated
~~"1.12% entropy ceiling"~~ — my byte-lane split did not match the real
IQ2_XXS fields. With the correct field split plus position-in-block context the
ceiling is **2.925%**, measured on 8% of one tensor. The *conclusion* (lossless
compression cannot close a 211 GB → 120 GB gap) survives; the number and method
did not.

### C9. Shared-basis factorization — RETRACTED as "dead"
The probe tested a single common right subspace; MoBE uses multiple bases with
expert-specific coefficients and a nonlinear transform, and the 90% gate was a
strawman (MoBE's own formula predicts ~27.8% at full expert rank). Nothing
about GLM was disproven. **MoBE-class factorization is untested, not dead.**

### C10. Router flatness — restated
~~"GLM's router is unusually flat"~~ → **contribution mass among the 8 selected
experts is diffuse (top-2 = 42%)**. The probe says nothing about global router
flatness, had no comparator model, and its "14,175 observations" were three
identical trajectories.

---

## Correctness

### C11. Append-resume divergence — OPEN STATE BUG
~~"chunk-shape numerics, not corruption"~~ — RETRACTED. The signature (outputs
continuing the *previous generation's* mid-word tail; different appends
producing identical outputs) fits stale state, not FP amplification.
Eliminated: stale-row attention (0/777 ROWTRACE violations), cross-process
nondeterminism (max|Δ| = 0 over 154,880 logits), and "any resume diverges".
Identified mechanism: BPE re-merge at the generation junction causes a
live-cache miss, a disk-KV load of a shorter checkpoint, then a long suffix
extension. ~~**Blocker:** the cold-boundary disk store never fires in the reproduction
harnesses~~ — **RETRACTED 2026-07-26.** The store fired every time and failed
with `No space left on device`; the filesystem was at 100% for the whole
investigation. My harness grepped only for the success string, so an attempted
-and-failed write looked like an unmet precondition. 168 GB has since been
freed and the decisive comparison is now runnable. The same logs confirm the
mechanism directly: `live kv cache miss live=5063 prompt=5066 common=5045
reason=token-mismatch`.
**RESOLVED 2026-07-26 (root cause, not fix).** With disk space freed the regime
reproduces on demand and the logit comparison is decisive: at the same final
position for the same 5066-token prompt, the resumed path (start=5044,
suffix=22) and a fresh-process reference (start=5064, suffix=2) give
**max|Δ| = 5.911 across 154,880 logits (18.2% of the logit range), mean |Δ| =
1.19, and different argmaxes** (14181 vs 785) with healthy margins (1.05, 1.35).
FP reassociation lands near 1e-3, and the engine is deterministic (an earlier
cross-process test measured max|Δ| = 0 on the same 154,880 logits), so this is
**state, not numerics**. The resumed text corroborates it by continuing the
previous generation's mid-thought.
*Caveat:* the reference arm also resumes by 2 tokens, because GLM's indexed
prefill always chunks a prompt this long; a fully chunk-free reference is not
obtainable on this path.
**The strict resume guard remains default; the fix (truncate to the common
prefix instead of loading a shorter checkpoint) is not yet implemented.**

### C12. 32K context — restated
~~"32K functionally proven"~~ → **retrieval beyond the 8192 dense cap is
proven**; the test touched 11,648 of 32,768 tokens (35.5%) with a single
salient-passphrase task. Capacity-edge behaviour is untested.

### C13. Engine changes reviewed by sol
- `DS4_GLM_TOPK_SKIP_LOAD` — **CORRECT** (round 2). Math equivalent to the
  implemented weight truncation for finite values; id collapse gated on the
  weights write succeeding. Caveat: collapsed ids reach profiling/debug output.
- Model-generation flush fix — round 1 BROKEN (pointer+size identity),
  round 2 BROKEN (same-process close/reopen). Now: identity is
  (pointer, size, GGUF header fingerprint, fd) plus `cuda_model_gen_invalidate()`
  on teardown and fd-clear. **Round 3 verdict pending.**

---

## Methodology corrections adopted

1. **State which phase a lever acts on, then prove the metric isolates it.** A
   harness timing 5047-token-prompt requests is ~95% prefill and cannot measure
   a decode lever.
2. **Verify the arm actually differs.** A "bug vs fix" A/B recorded 0 flushes
   and byte-identical cache counters in *both* arms — the storm is prefill-only,
   so a short-prompt harness could not produce a bug arm at all.
3. **One contrast is not a noise floor.** Retracted the "8× noise" framing.
4. **Repeats and ABBA ordering**, because arm order and machine state drift.
5. **Deterministic verification over inference** — unique-expert histograms and
   flush counts from the trace, not conclusions drawn from timings.
6. **Reviewers must be able to read the code under review** — `/home/dsv4` is
   unreadable to the sandbox and `vendor/ds4/` is a stale snapshot that
   silently misled a review round.

---

## Model facts (corrected 2026-07-26)

GLM-5.2 shape, read from `DS4_VARIANT_GLM52` in ds4.c: **256 experts per
layer** (not 160, which was wrong in the ledger, the memory file and several
harness comments), 8 used, 1 shared, 79 layers of which 75 are routed,
`n_indexer_top_k = 2048`.
This does not change any measured number, but it is *why* prefill is so
expensive: a 2048-token batch already touches ~99% of all 256 experts in every
routed layer, so no chunking strategy can reduce the sweep.

### Still unsupported (sol round 3)
- keep-7/keep-6 gains are one-fixture observations until the trajectory-
  controlled pass finishes.
- `unique=8/7/6` proves reduced *logical loader slab bytes*; the 29% drop in
  cache misses is the supporting evidence for physical NVMe reduction, but
  kernel-compute reduction is not shown.
- gen3 is statically reviewed but, at time of writing, unbuilt and unrun.
