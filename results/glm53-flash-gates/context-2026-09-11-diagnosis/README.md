# Why the four-slot 1M fill breached the 10 GiB floor, and the fix

Profile `glm-5.3-flash/cuda-spark-128g-1m` (pack A + K2 MTP shards, runtime-003,
FULL_DECODE_ONLY graphs, prefix caching on, `--kv-cache-memory-bytes 10600000000`).

| run | allocator | concurrency | outcome |
|---|---|---|---|
| `../context-2026-09-10-attempt1-batched2048-FAIL` | `expandable_segments:False` (inherited) | 4 x 250,128 | BREACH 9.71 GiB ~4.5 min into the fill |
| `../context-2026-09-10-attempt2-batched1024-FAIL` | `expandable_segments:False` | 4 x 250,128 | BREACH 9.90 GiB ~7 min into the fill |
| `one-250k-expandable-false.log.txt` | `expandable_segments:False` | 1 x 250,128 | flat 16.2-16.5 GiB, 568 s |
| `four-250k-expandable-false-BREACH.log.txt` | `expandable_segments:False` | 4 x 250,128 (slot 0 prefix-cache hit) | flat 14.5 GiB for 450 s, then 14.5 -> 9.48 GiB in ~100 s, BREACH |
| `four-250k-expandable-true.log.txt` | `expandable_segments:True` | 4 x 250,128 | flat, minimum 15.43 GiB, all four completed (1,841 s) |

Reading: batch size was not the cause (2048 and 1024 both breached at similar
points). A single 250K prefill does not grow memory; three or four interleaved
250K prefills with the default caching allocator fragment the CUDA pool and
MemAvailable plunges 5-7 GiB within ~100 s (process RSS unchanged, so the
growth is device-side on the unified memory). The inherited profile pinned
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`; switching it to `True`
keeps the fill flat. The profile now sets `expandable_segments:True`; the
qualifying evidence is `../context-2026-09-11/`.

Requests were the prepared `~/.cache/glm53-flash/context-direct-011/{0..3}-request.json`
inputs (250,128 prompt tokens each, `max_tokens` 48). `one_250k.py` /
`four_250k.py` are the samplers used (MemAvailable + top RSS every 5 s).
