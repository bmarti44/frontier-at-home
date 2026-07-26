# ds4 GLM disk-KV resume: stale-frontier contamination on different-suffix resume

**Status: repro complete, root cause confirmed by discriminating experiments + source review. NOT yet filed upstream (needs owner go-ahead — public report). Fix not yet implemented locally.**

## Summary

On the CUDA SSD-streaming GLM-DSA path, resuming a disk-KV checkpoint with a
prompt that *extends* the stored lineage (multi-turn append — the normal agent
pattern) produces output that continues the **previous request's generation
from its old frontier**, largely ignoring the appended text. Exact-replay
resume (identical prompt) is byte-correct; only genuine
resume-with-different-suffix corrupts.

Upstream code (pin 0a7ad77 + master), reproduces **without** any local patches
applied to the MoE/cache layer (the bug is in the GLM sync/resume path, which
we never modified).

## Repro (committed harnesses)

Engine: `ds4-server --cuda -m GLM-5.2-UD-IQ2_XXS... -c 32768 --ssd-streaming
--kv-disk-dir ... --kv-cache-boundary-align-tokens 4
--kv-cache-boundary-trim-tokens 0` (also reproduces at align 64 / trim 32 —
the boundary just moves).

1. Request A: 5047-token prompt, `max_tokens=16`, temp 0 → server checkpoints
   at 5044, generates 16 tokens (graph rows written through ~5063).
2. Request B (same server): prompt = A's prompt + A's generated text + an
   appended turn (any shape — adversarial mid-word glue *or* a clean
   `"\n\nUser: ..."` turn). Sync trace shows genuine resume:
   `start=5044 suffix=22-29 checkpoint=5044 indexed_keep=1`.
3. Control: same appended prompt as the FIRST request on a fresh server +
   wiped kv-dir (canonical cold).

**Result:** B diverges from control at character 0, deterministically. All
resumed outputs begin with the same continuation of A's generation tail
(A's last generated token ends mid-word; every resumed output starts by
completing it), regardless of what was appended. Two different appended
prompts produced byte-identical resumed outputs in one run.

Discriminators run:
- Per-token decode vs indexed-batch suffix evaluation: **identical corrupted
  outputs both ways** → not a batch-kernel bug; follows the resume itself.
- Prompts that diverge from the stored lineage *before* checkpoint length
  silently fall back to cold (starts_with fails) and match control exactly —
  this masks the bug in some traffic mixes.
- Exact-replay (suffix rewrites identical tokens over the same rows):
  byte-identical to cold at every align/trim tested.

## Root cause (source review, sync block ~ds4.c:58030-58330)

`start = s->checkpoint.len` establishes the resume point, but nothing rewinds
the graph: no truncation of the sparse/indexer frontier, valid-row counts, or
incremental compressor state to `start`. When
`indexed_resume_keeps_sparse_state` is taken, state built over the previous
lineage's rows `[start, old_frontier)` (old suffix + generated tokens)
remains live and dominates the continuation. Exact-replay is immune because
the recomputed suffix and re-generated tokens are identical to the stale
content.

Invariant that should hold before evaluating token `start`:
every live graph frontier represents exactly tokens `[0, start)`.
Save/restore should assert `saved_graph_frontier == saved_token_count`.

## Update (post-guard, cache exonerated)

A prefix-replay-only guard (v4.8d) now contains the bug locally
(correct-or-cold). Two further discriminators sharpen the root cause:
(1) with the local persistent expert cache fully disabled the corruption is
byte-identical — all local patches exonerated; (2) even a forced from-zero
re-prefill on a session that just underwent the GLM `ds4_session_load_payload`
restore deterministically differs from a virgin-server prefill of the same
prompt. The restore itself leaves the graph in a state that corrupts any
subsequent evaluation except pure prefix-replay. A full stateful-member audit
found no graph-side frontier scalar and correct row/visibility bounds at every
C call site — remaining suspects are the restore's GPU writes (e.g. indexer
key cache content/layout/ordering) or the process-global selected-expert
staging cache interactions.

## Fix options

- **A (minimal, correctness):** on resume, if the graph frontier extends past
  `start` and the new prompt is not a pure prefix-replay of the previously
  evaluated tokens, force the full-reset/cold path (empirically proven
  correct: cold-after-generation matches virgin-server cold byte-for-byte).
  Preserves exact-replay fast path; multi-turn appends pay cold prefill.
- **B (proper):** truncate sparse/indexer frontier state to `start` on resume
  (row counts, partial-window accumulators, sequence length). Restores fast
  multi-turn resume.

## Evidence

`results/glm52-gates/loadprof-2026-07-25.json` sections `append_probe`,
`append_isolate`, `append_discriminator`, `append_clean`;
harnesses `results/glm52-gates/harness/ttft_append_probe.sh`,
`append_isolate.sh`, `ttft_append_clean.sh`; sol xhigh source-review trail
`results/glm52-gates/logs/loadprof1/sol-append-rootcause.txt`.
