# GLM-5.2 on a single DGX Spark — research + ds4 streaming path (2026-07-24)

Question: can GLM-5.2 (753B MoE, 1M ctx) run with the full context window on one
DGX Spark (GB10, 128 GB unified = 119.7 GiB MemTotal, ~273 GB/s, sm_121)?
Method: direct web research + HF API verification + source reading of the
engines involved + adversarial codex (sol, high effort, web search) review.
Everything load-bearing below was verified against primary sources or code.

## Part 1 — The flat answer on "full context window"

**No. Full 1M context on one Spark is impossible today, on every axis at once:**

1. **Weights don't fit.** No published GLM-5.2 GGUF — pruned or not — fits in
   119.7 GiB. Verified smallest artifacts (HF API shard sums, not model-card
   claims): full-model Unsloth UD-IQ1_S **201.8 GiB**; ik-style 1.63 bpw
   **143.0 GiB**; REAP-50% (381B) Q2_K **129.5 GiB** (and it collapses under
   greedy decoding); REAP-34% (504B) Q2_K_XL **196.4 GiB** — the model card
   says "~111 GB", the actual shards sum to 210.9 GB. Trap for others: verify
   shard sums via the HF API before planning around card claims.
2. **KV at 1M doesn't fit either.** GLM-5.2 is MLA-compressed like DeepSeek
   (kv_lora_rank 512 + rope 64 = 576 elems/layer/token × 78 layers). Verified
   in llama.cpp source (`has_v = !is_mla`, K-only store): **87.7 GiB at 1M in
   f16**, 32.9 GiB even at q4_1 — before weights, buffers, or indexer state.
3. **The compute path for 1M doesn't exist in stock llama.cpp.** GLM_DSA runs
   as plain dense MLA; the DSA/IndexShare sparse-indexer runtime is not
   implemented (PR #19460 merged Feb 13 loads indexer tensors unused; #24770
   just made them optional; #24730 still open). Dense prefill to 1M at the
   observed O(10-100) tok/s GLM-5 rates is a multi-day operation.

Multi-node is the only demonstrated large-context route: 2× Spark llama.cpp
RPC (UD-IQ1_S, 256K ctx, 3.4-8.9 tok/s decode), 2× Spark 2-bit experts
(~21.5 tok/s @ 96K), 4× Spark (~24 tok/s, 128K-1M, re-architected stack).
Hosted `GLM-5.2[1m]` remains the only practical full-window option.
No GLM-5.2 Air/Flash exists (community requests open, no Z.ai commitment).

## Part 2 — The single-box loophole: ds4/DwarfStar SSD expert streaming

antirez's ds4 (DwarfStar) — the same engine lineage we already serve DSV4
with — runs **unpruned** GLM-5.2 on single 128 GB machines by keeping
non-routed weights resident (~19.6 GiB) plus a dynamic routed-expert cache,
and loading expert misses from the GGUF on demand. Demonstrated on M5 Max
128GB (Metal) and documented for Strix Halo 128GB (ROCm). Related prior art:
Colibri ran full int4 GLM-5.2 on ONE DGX Spark at **2.39 tok/s** faithful
routing (3.33 tok/s with ~14% expert substitution — not faithful); NeutronStar
(DwarfStar-derived) does CUDA io_uring/O_DIRECT streaming + DSA at ~0.4 tok/s
decode / 6.5 tok/s prefill on consumer CUDA.

### Finding 1: the "CUDA port" already happened upstream — docs lag code

Everything published (README, blog, our first fetches) says streaming is
Metal + ROCm-for-GLM. The code says otherwise. Upstream master
(`antirez/ds4`, commit `bbd069d` "Add CUDA and ROCm SSD streaming",
2026-06-14):

- `ds4_backend_supports_ssd_streaming()` returns **true for CUDA** on any
  non-Apple GPU build (ds4.c:393).
- `ds4_cuda.cu` has the full expert-cache machinery: selected-expert compact
  load, seed/budget/hotness API, staged 4-deep pipelined chunk reads with
  O_DIRECT alignment (NOT bare mmap faults — the GB10 mmap pathology we
  documented for llama.cpp is already sidestepped by design), single-GPU
  guard, GLM selected-id device readback.
- The built binary's own help: `--ssd-streaming  Metal/CUDA/ROCm: opt in...`.
- **Built and linked clean on this host today** (make cuda CUDA_ARCH=sm_121,
  upstream master `0a7ad77`, at /home/dsv4/ds4-project/src/ds4-upstream-master).

What is still genuinely not on CUDA (verified in source after sol review —
these findings downgrade the performance expectations, not the feasibility):
- **No persistent expert cache.** The CUDA path only does per-batch
  selected-expert loads (`prepare_selected_batch` → begin_load, invalidated
  each round); `seed_experts` is a no-op, `budget_for_expert_size` returns 0,
  hotlist seeding never engages. Decode re-streams its experts from NVMe
  essentially every token. This is THE gap: Metal/ROCm keep a resident
  hotness-managed cache; CUDA does not.
- **auto cache sizing** (`ds4_backend_supports_streaming_auto_cache`):
  CUDA requires explicit `--ssd-streaming-cache-experts N|NGB` — and worse,
  on CUDA that budget is guard *accounting* for a resident cache the backend
  does not actually create.
- **GLM full-layer streaming prefill**: the CUDA setter is a stale no-op
  (ds4_cuda.cu:26941 — "SSD streaming is not used on the CUDA backend"),
  while the generic graph still enables a full-layer mapping path at chunk
  ≥64 on non-ROCm builds (ds4.c:38075-38146) — inconsistent; needs runtime
  tracing for mmap-fault behavior on GB10 before trusting prefill.
- **12 stubbed GLM CUDA functions** (print "CUDA stub called", return 0),
  including `glm_attention_indexed_decode_split_group8_typed`,
  `glm_attention_indexed_batch_typed`, the flash/full attention builders,
  and a q4 direct MoE batch variant. The live path avoids them today, but
  any config change (notably cache_f16) must be checked against routing
  into a stub.

### Finding 2: GLM already uses the DeepSeek KV compression — the gap is one
### Apple-only constant (user asked exactly this)

ds4's GLM path stores the **same compressed MLA latent as DeepSeek** — there
is no expanded K/V anywhere (`glm_graph_expanded_kv_cache_enabled()` is
hard-false; `glm_store_compact_kv` kernels store the 576-elem latent). The
IndexShare schedule is implemented (`glm_graph_layer_uses_full_indexer`:
dense leading layers, then every 4th — matches index_skip_topk_offset=3,
index_topk_freq=4), and a full **DSA indexer top-k kernel suite exists in
ds4_cuda.cu** (indexer_topk_* incl. CUB variants) — ds4 has the sparse
runtime stock llama.cpp lacks, so decode attention reads top-2048 tokens
regardless of context depth.

The only compression difference vs DeepSeek-on-Metal is storage width
(ds4.c:14821):

```c
#if defined(__APPLE__)
#define DS4_GPU_ATTN_COMP_CACHE_F16 1      /* Metal: f16 compact cache  */
#else
#define DS4_GPU_ATTN_COMP_CACHE_F16 0      /* CUDA/ROCm: f32            */
#endif
#define DS4_GPU_GLM_COMPACT_CACHE_F16 DS4_GPU_ATTN_COMP_CACHE_F16
```

So CUDA stores the latent in f32: **175.5 KiB/token** (sol independently
measured the same figure in other CUDA streaming stacks). Metal stores f16:
87.8 KiB/token. Three facts make flipping this on CUDA low-risk and cheap:
(a) the CUDA GLM attention kernels already take and honor `cache_f16`
(cache_elem = 2 or 4); (b) the compressor **already FP8-rounds rows in F32
staging** before the cache write (upstream's own comment: "a storage
optimization rather than a semantic approximation") — f16 storage is
numerically free; (c) it is one shared constant, so DSV4 serving would gain
the same halving. Natural follow-up: true FP8 storage (43.9 KiB/token, 4×
vs today's CUDA) since the values are already FP8-rounded — ROCm has an
fp8-kv module to crib from; that one is real kernel work.

KV budget at GLM-5.2 dims — CORRECTED per sol (adds the 128-elem indexer-key
cache on the 21 full-indexer layers; source estimator ds4.c:34608 includes
both): f32 **186.0** KiB/t | f16 93.0 | 1-byte 46.5 → 32K: 5.8 / 2.9 / 1.5
GiB; 128K: 23.3 / 11.6 / 5.8 GiB; plus ~4.19 GiB persistent graph scratch.
The engine's non-Apple memory guard also reserves 32 GiB by default
(ds4.c:37341-37474), so engine-accounted cache maxima on this host are
~40-61 GiB depending on ctx/dtype — NOT the 80-90 GiB the first draft of
this doc assumed. Every GiB saved still goes to the (future) expert cache
(~0.6 GiB ≈ 50-60 IQ2/Q2 experts).

### Finding 3: this host is unusually well suited

- NVMe O_DIRECT sequential read measured today: **10.7 GB/s** (faster than
  the M5 Max SSDs the Metal demos ran on). Expert I/O per token — CORRECTED
  per sol: 75 routed layers (3 dense) × 8 experts; the P2 artifact mixes
  IQ2_XXS gate/up + Q2_K down = 10,616,832 B/expert → **6.37 GB/token**
  uncached → **~1.68 tok/s** as an OPTIMISTIC sequential-I/O ceiling (real
  selected reads are scattered + synchronized; wild results agree: 0.40-0.46
  tok/s NeutronStar/SpeedyColibri; Colibri's 2.39 required its own resident
  cache). Until a persistent CUDA cache exists, cache-hit math does not apply.
- GB10 unified memory = a future expert cache is directly GPU-addressable,
  same as Apple silicon; no PCIe hop like consumer-GPU setups — the port is
  simpler here than anywhere else.

### What it will NOT be

Not 1M, and not fast until the CUDA cache work lands. With f16 compact
cache the memory arithmetic supports ~64-128K context, but today's
upstream CUDA decode is I/O-ceilinged at ~1.7 tok/s (optimistic) with
prefill behavior unverified on GB10. This is a "run the actual flagship
locally" proof first, a usable-agent-endpoint project second — the
persistent expert cache (P4a) is what would move decode toward the
hit-rate regime. DSpark/MTP speculation is not supported for GLM upstream.

## Part 3 — Phased plan (tracked as tasks #26/#27; REORDERED per sol review:
## diagnostics approved, performance work re-sequenced)

Strategy (user-set): most-likely-to-work first, one change at a time, never
touching the DeepSeek production path. GLM runs vanilla upstream master;
DSV4 stays on Entrpi v0.4.2 (that tree has no GLM code and no ds4_ssd.c —
cross-merging the two heavily-diverged forks is explicitly rejected).

- **P0 (done):** upstream master built clean for sm_121; flags verified.
  Caveat (sol): compiling proves nothing about the 12 GLM CUDA stubs — they
  fail at runtime only if a path routes into them.
- **P1 (diagnostics; needs a ~15-min prod window, NOT disk):** smoke CUDA
  streaming with the DSV4 IQ2XXS weights already on disk (81 GiB), small
  explicit budget. Validate: O_DIRECT staged reads actually engage (vs
  mmap-fault fallback — trace bytes read), throughput under scattered load,
  memwatch interaction, and DSA top-k with context > 2048 so the indexer
  path really runs. Do NOT read this as cache validation — on CUDA there is
  no persistent cache to validate.
- **P2 (disk, go/no-go):** free ~90 GiB (delete the falsified teamblobfish
  IQ2_XXS-XL — no MTP tensors, no other use; NVMe free 190 GB today), then
  `download_model.sh glm-antirez-iq2xxs` (196.6 GiB). GO only if we accept
  P3 as a proof-of-concept at ~1.7 tok/s ceiling, or after P4a exists.
- **P3 (POC measurement, expectations corrected):** GLM-5.2 on one Spark,
  vanilla flags; measure faithful tok/s (expect 0.4-1.7), TTFT vs context,
  actual read amplification. Purpose: ground truth + upstream-reportable
  numbers, not a usable endpoint.
- **P4a (the real enabler):** implement/port the persistent expert cache on
  CUDA — fill in the no-op seed/budget/hotness API against the Metal/ROCm
  reference semantics, unified-memory resident (GB10 needs no PCIe
  management). Engine-accounted cache maxima on this host: ~40-61 GiB
  (guard reserves 32 GiB by default; possibly tunable). This is what moves
  decode from the I/O ceiling into the hit-rate regime. Check upstream/
  NeutronStar first — this gap is known and may be in flight.
- **P4b (KV width):** only after P4a: parametrize DS4_GPU_ATTN_COMP_CACHE_F16
  on CUDA; first verify no consumer routes into the stubbed *_typed/group8
  variants (disable or implement them), test at >2048 ctx, run the GLM
  quality fixtures + DSV4 goldens. FP8 store is real kernel/checkpoint work,
  not a free extension (sol).
- **P5 (switching, user requirement; sol: sensible, but wire it only once
  GLM CUDA is qualified under the corrected constraints):** DeepSeek work is
  preserved untouched;
  DSV4 and GLM become selectable profiles behind the SAME front door. The
  exposure chain (Caddy/tailscale → auth :8010 → loopback :8011) is
  engine-agnostic, so clients keep one URL + key. Design:
  `scripts/30_switch_model.sh {dsv4|glm52|status}` —
  (a) refuses to run if the target is already serving; (b) stops the current
  8011 backend (systemd unit or dev ds4, using the existing rollback
  procedure); (c) starts the requested profile: `dsv4` = the current
  qualified/override stack exactly as today; `glm52` = upstream-master
  ds4-server --cuda --ssd-streaming with explicit cache budget; (d) runs the
  existing verify (tailnet 401 + local health + one timed completion);
  (e) re-arms the profile's own memwatch (separate kill lines: DSV4 12 GiB
  qualified / 10 GiB dev; GLM gets its own, sized in P3) and separate
  disk-KV/warm-state dirs so neither model can corrupt the other's caches.
  Memory makes the two mutually exclusive by construction — the switch is
  stop-one-start-other, ~2-3 min bounded by GLM resident load (~20 GiB at
  10.7 GB/s NVMe) or DSV4's 92 s no-mmap load. An unattended reboot still
  lands on the qualified DSV4 systemd unit (unchanged), so GLM can never
  become the accidental default.

## Part 4 — Deterministic verification and gates (added 2026-07-24, user
## requirement before implementation starts)

Rules: every phase ends at a GATE with binary pass/fail assertions —
temperature 0, fixed committed fixtures, token/byte/count thresholds, never
wall-clock feelings. Each gate writes a JSON evidence file to
results/glm52-gates/ (gate name, timestamp, assertions with expected/actual,
raw log paths). A gate PASSES only after (a) all assertions green AND
(b) an adversarial review of the evidence by codex sol at
`model_reasoning_effort=xhigh` finds no fatal objection. When stuck at any
point, escalate to sol xhigh before improvising. Anti-over-engineering rule:
gates reuse existing repo tooling (regression-suite.py patterns,
gguf-tensors.py, 42_verify_exposure.sh, memwatch) — no new frameworks;
a gate is a script or a checklist with evidence, whichever is smaller.

- **G0 (build integrity — already satisfiable):** upstream pin commit +
  binary sha256 (12-char) recorded; `--help` advertises `--ssd-streaming`
  on Metal/CUDA/ROCm; `make` exit 0 warning-free tail. Evidence: build log
  tail + hashes.
- **G1 (CUDA streaming machinery, DSV4 weights):** with prod stopped
  (bounded window, rollback = existing tested procedure): server starts in
  `--ssd-streaming` with explicit cache arg; log proves streaming mode
  engaged and staged O_DIRECT reads active (assert on the engine's own
  startup lines; cross-check /proc/<pid>/io read_bytes grows during a
  completion while page cache is cold); one fixed temp-0 prompt returns
  200 with byte-identical output across two consecutive runs; ZERO
  "CUDA stub called" lines in stderr; MemAvailable never crosses the
  armed memwatch line; prod restored and verified (tailnet 401 + warm
  probe < 2 s TTFT). All binary.
- **G2 (artifact):** teamblobfish deletion logged with freed bytes; GLM
  GGUF downloaded; exact byte size matches HF API; sha256 recorded;
  gguf-tensors.py parses header, arch == glm-dsa, tensor count recorded;
  free NVMe ≥ 40 GB after. All binary.
- **G3 (GLM POC on one Spark):** server loads GLM via streaming; fixed
  temp-0 fixture set: one short prompt AND one > 2048 tokens (forces real
  DSA top-k, per sol); assertions: 200s, byte-stable outputs across two
  runs, zero stub lines, no watchdog breach, coherence spot-check against
  upstream's GLM quality fixtures (subset, deterministic). Measured tok/s
  and read-amplification are RECORDED as ground truth, not gated — the
  corrected expectation is 0.4-1.7 tok/s and the POC passes on
  correctness, not speed.
- **G4a (persistent CUDA expert cache — the feature):** correctness gate:
  outputs byte-identical cache-on vs cache-off on the full fixture set at
  temp 0; effectiveness gate: on an identical repeated request,
  /proc/<pid>/io read_bytes for run 2 < 25% of run 1 (deterministic proxy
  for hit rate; threshold revisitable with sol at gate review);
  memory gate: cache stays within its configured budget (engine census /
  MemAvailable delta ≤ budget + 1 GiB); upstream GLM quality fixtures pass
  identically to cache-off. DSV4 prod untouched by construction (different
  tree); still re-run our regression-suite.py against prod afterward as a
  no-interference check.
- **G4b (f16 compact cache):** precondition gate: instrumented sweep of the
  full fixture set shows zero routes into stubbed *_typed/group8 variants
  with f16 enabled; quality gate: GLM fixtures pass; size gate: engine's
  reported cache bytes halve (±5%); stability gate: temp-0 outputs
  byte-stable across runs (f16-vs-f32 outputs may legitimately differ —
  compare each against fixture expectations, not each other).
- **G5 (profile switch):** scripted round-trip dsv4 → glm52 → dsv4;
  after each leg: correct engine identity string, tailnet 401, local
  health 200, one timed temp-0 completion 200, memwatch armed on the
  verified engine pid (child-of-sudo-wrapper procedure), other profile's
  state-dir checksums unchanged; static assert that the systemd unit still
  starts the qualified DSV4 stack. All binary, all in one evidence file.

Each of G1-G5 closes with: `codex exec -c model_reasoning_effort=xhigh`
adversarial review of the gate's evidence JSON + logs; findings addressed
or explicitly waived with reasoning recorded in the evidence file.

## Sources (primary ones)

- https://github.com/antirez/ds4 (master @ 0a7ad77; bbd069d CUDA/ROCm streaming; 005afed GLM inference, Jul 18)
- https://huggingface.co/zai-org/GLM-5.2 (config.json: 78L, 512 rank, DSA fields) / release https://z.ai/blog/glm-5.2
- https://github.com/ggml-org/llama.cpp — PR #19460 (GLM_DSA, indexer unused), #24770 (optional indexer tensors), issue #24730 (open)
- https://huggingface.co/0xSero/GLM-5.2-REAP-504B-GGUF (API tree = 196.4 GiB Q2_K_XL) · unsloth/GLM-5.2-GGUF · pipenetwork/GLM-5.2-REAP50-* · sokann 1.63bpw
- https://github.com/JustVugg/colibri/issues/161 (single-Spark 2.39 tok/s faithful) · https://github.com/GriffinPilz/SpeedyColibri · https://github.com/giannisanni/neutronstar
- https://forums.developer.nvidia.com/t/374523 (2× Spark UD-IQ1_S RPC) · /t/375416 (4× Spark) · howtospark.com/recipes/glm-5-2-dual-spark-tp2
- ik_llama.cpp DSA: https://github.com/ikawrakow/ik_llama.cpp/pull/2093
- atcyrus.com "GLM-5.2 local hardware requirements" (user-supplied 2026-07-24): general roundup; repeats the falsified 0xSero "~111 GB" Q2_K_XL claim (HF API shard sums = 196.4 GiB), llama.cpp-only, no single-Spark or streaming path — no plan impact
