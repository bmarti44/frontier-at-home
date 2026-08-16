# Decode-stage timing instrumentation

Date: 2026-08-16

## Scope and source

Evidence-only CUDA decode instrumentation for the `ds4` 4d54edd source plus
the dynamic expert-count patch. The supplied source tree was not a Git
worktree, so these pre-edit hashes define the patch base:

- `ds4.c`: `5b47fcb4ffd8…`
- `ds4_cuda.cu`: `6a8a2f7e7ec6…`
- `ds4_gpu.h`: `d733edc14c70…`

The session sandbox denied writes outside the repository and `/tmp`, despite
the task authorization. The implementation and build therefore used the
byte-identical mirror `/tmp/glm52-dynexp2-decode-stage-timing`; the requested
cache tree `/home/bmarti44/.cache/glm52-dynexp2-patched` remains unchanged.
Apply `decode-stage-timing.patch` from that tree with `patch -p1`.

## Behavior

- `DS4_DECODE_STAGE_TIMING` is presence-tested once by
  `cuda_decode_dispatch_env_refresh()` during CUDA initialization. It defaults
  off and logs when enabled.
- `DS4_DECODE_STAGE_TIMING_EVERY` is parsed once at the same point, defaults to
  32, and accepts positive `uint32_t` values.
- Enabled mode records timed CUDA event pairs on `cudaStreamLegacy`, then
  synchronizes only the final token event at the reporting boundary.
- Each line has the required prefix and metrics, followed by window deltas for
  `majflt`, `minflt`, and static call counters for the four CUDA async-selected
  compatibility stubs (`signal`, `read`, `wait`, and `commit_wait`).
- Event creation, allocation, elapsed-time reads, `getrusage`, formatting, and
  output occur only when the cached flag is enabled.

The disabled decode path adds cached-boolean branches only at the stage
boundaries in `ds4.c`, the warm arena-copy boundary in `ds4_cuda.cu`, and the
four optional stub entry points. No kernel body, kernel math, launch stream, or
copy stream was changed, and no new `getenv` occurs in the decode path.

## Group boundaries in the patched source

- Router select + selected-ID D2H + CPU remap + remap H2D envelope:
  `ds4.c:40078-40201`. Warm arena-copy event time is subtracted from this
  envelope when reporting `router`.
- Arena-to-staging expert copies, one event pair around each completed warm-hit
  gate/up/down triplet: `ds4_cuda.cu:24136-24155`.
- Routed-MoE kernels: `ds4.c:40277-40295`.
- Indexer score + top-k: `ds4.c:45847-45863`.
- Selected attention + non-routed GEMVs + output head: the full-token envelope
  is `ds4.c:45527-46211`; `attn_dense` is its non-overlapping remainder after
  subtracting router, copies, MoE, and indexer. `total` is the full envelope.

The remainder construction is necessary because selected attention and dense
work are interleaved with router/MoE work across layers; it yields five
non-overlapping completed buckets without changing stream structure. The
router envelope likewise necessarily contains the interleaved staging copies,
so those completed copy intervals are subtracted.

## Validation

`make cuda-spark -j2` completed successfully in the exact mirror and linked
`ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`, and `ds4-agent`. A first build
failed at compile time because `ds4_cuda.cu` intentionally does not include
`ds4_gpu.h`; adding the matching internal stage enum fixed that interface-only
issue. No model or GPU executable was run.

Runtime measurements and byte-identical output qualification were intentionally
not attempted because the task prohibited running a model/GPU process.
