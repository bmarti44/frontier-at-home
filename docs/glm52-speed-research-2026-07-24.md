# GLM-5.2 on one Spark: speed research synthesis (2026-07-24)

Four independent research streams (academic-literature agent, systems-practice
agent, GLM-specific agent, codex sol xhigh with web search) on how to close the
gap between faithful SSD-streamed GLM-5.2 (~1.7 tok/s I/O ceiling measured at
G1-class physics) and the acceptance bar (DSV4 on this box: 467 tok/s prefill,
~18.4 tok/s decode, 32K ctx). Full agent outputs in the session record; this
file keeps the load-bearing numbers and the ranked plan.

## The four verdicts in one table

| Source | Faithful decode ceiling on this box | Key evidence |
|---|---|---|
| Academic lit | ~3.5-5 tok/s (caching+bytes levers); >5 only via 40-50% expert pruning to RAM-resident | near-uniform expert popularity in DeepSeek-class routers (arXiv:2505.16056); prefetch measured NEGATIVE at saturated bandwidth (WiSP 2606.21868); MTP union ~i.i.d. (MoESD 2505.19645) |
| Systems practice | 11.1 tok/s measured faithful on GB10 (Colibri #161: ~64 GB expert tier, 99.9% hit, PLD n-gram drafts, GPU MLA; attention becomes the wall) | workload temporal locality >> aggregate popularity; NeutronStar (ds4 CUDA fork) has LFU cache/warm-start/chunked prefill working |
| GLM-specific | MTP head is good (SGLang accept ~5/6 coding); Mesh-LLM merged a GLM-DSA MTP graph 2026-07-23 (CPU/Metal reference); NU176 saliency map usable as cache prior; DSA prefill crossover only >40K ctx | but NeutronStar measured GLM MTP NET NEGATIVE in streaming regime (~10-20% cross-row expert overlap) |
| sol xhigh (projection w/ arithmetic) | 18 tok/s faithful AR: NO (needs >=90.7% exact hit). Honest: 8 conservative / 10-12 strong / 12-15 max sustained coding w/ MTP / 16-18 experimental stretch. TTFT: cold 20K prompt 115-190 s; WARM repeated prefix 0.5-2 s | miss-bytes equation: 6.4 GB/tok x (1-H) vs 9.5-10.7 GB/s |

Reconciliation: the academic pessimism (uniform popularity) and the Colibri
11.1 tok/s are both right — aggregate popularity is flat, but a live agent
session has strong SHORT-HORIZON temporal locality (its working set of experts
is small). Which regime Hermes lands in is measurable, not arguable.

## Against the user's acceptance bar

- Decode: 18.4 tok/s sustained faithful — NOT credible. 12-15 tok/s is the
  honest max (coding workloads, MTP, big cache, kernel work); 8-12 realistic.
- TTFT/prefill: cold 467 tok/s — no (110-180 tok/s evidence-backed via
  full-layer chunked prefill). BUT the agent workload repeats a ~20K system
  prompt: exact disk prefix cache gives 0.5-2 s warm TTFT, which BEATS the
  DSV4 experience (~90 s TTFT on cold 19K prompts, <2 s warm).
- Context: parity achievable (32K KV = 5.8 GiB f32 / 2.9 f16 + scratch; 128K
  possible in-budget with f16 compact cache).

## Ranked lever stack (faithful-first, multiplicative)

0. MEASURE FIRST (cheap, decides everything): instrument ds4 to log per-layer
   top-8 expert IDs for ~10k decode tokens of real Hermes-style traffic ->
   consecutive-token overlap, hit-vs-residency curve, gate-mass concentration.
   (All four streams independently demand this.)
1. Persistent host expert cache at Spark scale (~90-100 GiB budget; exact LRU
   or Least-Stale + small static hot pins from the in-tree
   ds4_streaming_hotlist_glm52.inc / NU176 saliency prior) + warm-start
   persistence of the cache index. NeutronStar has this working in a ds4 CUDA
   fork (DS4_CUDA_HOST_EXPERT_CACHE_GB, DS4_CUDA_HOST_CACHE_STATE=file).
2. Chunked union ("weight-stationary") prefill: read each layer's chunk-wide
   expert union once per chunk. NeutronStar: 21x prefill (6.5 t/s on ~2 GB/s
   disk); our disk is ~5x that. REQUIRES the PR#513 LUT fix (applied to our
   tree 2026-07-24, binary f3772018b22d) — without it GLM batch prefill is
   silently corrupt.
3. Exact disk prefix cache for the stable agent prompt (ds4 upstream already
   has --kv-disk-dir; must verify DSA indexer state is serialized — ik_llama
   needed PR #2146 for exactly this). Target: warm TTFT <2 s.
4. I/O path: batched aligned O_DIRECT with inflight dedup; threaded pread
   QD>=32 or io_uring+O_DIRECT QD64. Buffered io_uring measured WORSE on GB10;
   GPUDirect Storage measured NO-GO on GB10 (unified memory; cuFile falls back
   to compat mode).
5. Zero-copy unified-memory expert compute (no H2D copies) + recycled aligned
   buffers (SpeedyColibri: warm loads 21.7x, decode 2.6x from buffer reuse).
6. Router-lookahead prefetch ONLY after residency work (+21% at NeutronStar
   once disk had headroom; negative while saturated).
7. Native MTP + PLD n-gram drafting ONLY after residency (Colibri crossover:
   +66% at 99.9% hit, -48% at 92.5%; NeutronStar: net negative cache-less).
   Union-aware adaptive draft depth; pin the MTP expert pool (~2.65 GiB).
8. f16 compact KV (planned G4b) — halves KV, frees ~gigabytes for cache slots.
9. Non-faithful fallbacks (explicitly gated, quality-scored, off by default):
   cache-aware routing (Colibri CACHE_ROUTE: +39% measured on GB10 at ~14%
   substitution — but SpecMD found substitution catastrophic on other MoEs;
   full quality gate required), sub-2-bit cold-expert requant, REAP-pruned
   variant as a separate model profile (fits RAM, >5 tok/s class, known
   quality cost: loop-rate doubles; NOT the flagship).

## Engineering-source shortlist

- NeutronStar (github.com/giannisanni/neutronstar): ds4 CUDA fork with levers
  1,2,4,6 implemented + the PR#513 fix author. Evaluate vendoring vs porting
  at G4a — porting individual features keeps our upstream pin; vendoring gets
  everything tested-together on 4060Ti-class hardware but adds fork risk.
- Colibri #161/#199 threads: GB10-measured ladder 2.39 -> 3.33 (cache-route)
  -> 11.1 (residency+PLD+GPU MLA) tok/s.
- Mesh-LLM feat/jianyang-glm-52 patch queue: reference GLM-DSA MTP graph
  (merged 2026-07-23, CPU/Metal).
- ISCA 2026 "Approaching Shannon Bound" (arXiv:2606.15789): lossless rANS over
  quant blocks, ~10-23% byte cut, zero quality risk, decode fused in kernel.
- WiSP (arXiv:2606.21868): expert-cache vs KV-cache marginal-value RAM split.
- IndexCache (arXiv:2603.12201) + ik_llama #2068/#2109/#2146: DSA long-context
  decode/prefill and indexer-state serialization precedents.

## Revised expectations to plan against (sol's numbers, adopted)

- 8 tok/s decode = conservative success; 10-12 = strong; 12-15 = max honest
  sustained (coding+MTP); 16-18 = stretch experiment, never an SLO.
- Prefill 110-180 tok/s cold; warm TTFT 0.5-2 s via prefix cache.
- The G5 switch decision uses these against the recorded DSV4 baseline; if
  the bar is strictly 18.4 tok/s sustained faithful decode, expect NO-GO on
  the switch with GLM qualified as secondary profile instead.

## Impact on the gate plan

- G3 (unchanged): correctness POC; ALSO record the routing trace (lever 0) if
  cheaply possible, else a dedicated trace run follows G3.
- G4a (expanded): persistent expert cache = levers 1+4+5; gate criteria as in
  Part 4 (byte-identical outputs cache-on/off, run-2 read_bytes < 25% run-1,
  budget adherence) plus recorded hit-rate curve.
- G4-prefill (new sub-gate): chunked union prefill + disk prefix cache with
  DSA-state round-trip proof (byte-identical continuation after restore) +
  warm-TTFT measurement.
- G4-mtp (new sub-gate, last): native MTP; only attempted after G4a shows
  >=90% hit on real traffic; acceptance + net tok/s recorded; auto-off below
  break-even.
- G4b f16 compact KV: unchanged.
