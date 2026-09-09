# GLM-5.3-Flash full-load sizing audit

**NO_GO: the current full-resident K2 plus dense-overlay representation cannot
complete a cache-off full load with a 40 GiB whole-host floor when starting from
115 GiB MemAvailable. Feasibility with an 18 GiB floor remains NO_RESULT.**

This is a metadata and source audit, not a model load, allocation measurement,
quality result, or permission to launch. No weights were downloaded, no Torch or
CUDA code was imported by the census, and no production code changed. Existing
native, cache, MLA and KDA attempts retain their original verdicts.

## Reproduce

From the repository root:

```bash
/home/bmarti44/.cache/glm53-flash/native-runtime-002/runtime/bin/python3 -I -B \
  results/glm53-flash-gates/load-sizing-audit-001/census.py
```

The output is [census.json](census.json). The script verifies every file in the
archived model-layout inventory against the retained metadata directory before
counting all 120 K2 shard headers. It validates tensor dimensions, byte offsets,
unique names and index coverage, and reproduces overlay shapes using the exact
pinned `plan_outputs` function. No remote request or weight access is involved.
[source-pins.json](source-pins.json) records complete SHA-256 hashes and exact
local paths for all source/configuration inputs. The output binds both that
file and the census script. Header bytes are independently bound by the existing
model-layout inventory; full weight contents have not been downloaded or
independently verified by this audit.

The default 115 GiB starting value is a fixed sizing assumption supplied for
this audit, not a fresh run's start-memory evidence. At audit start the host
reported MemAvailable 121259652 kB, SwapTotal 16777212 kB, SwapFree 16607500 kB;
both restore and guard units were active. These observations do not replace a
future stable-start gate or a before/after swap-delta check.

## Payload arithmetic

The base headers describe 150,226 tensors. Removing the 315 BF16 overlay targets,
adding their 1,260 packed replacements, and excluding the MTP layer leaves
147,690 selected tensors:

```text
  97,709,588,472  original tensor payload bytes
- 14,361,296,896  replaced dense BF16 payload
+  3,539,957,996  packed dense overlay payload
-  2,192,230,400  excluded layer-45 MTP payload
= 84,696,019,172  selected payload bytes = 78.87931463494897 GiB
```

| Selected category | Bytes | GiB |
| --- | ---: | ---: |
| Routed experts | 76,547,503,872 | 71.290418 |
| Other language-layer tensors, including dense overlay | 4,483,699,172 | 4.175770 |
| Token embedding | 1,268,776,960 | 1.181641 |
| Output head | 1,268,776,960 | 1.181641 |
| Vision tower | 1,127,254,016 | 1.049837 |
| Final language norm | 8,192 | 0.000008 |

These are tensor payloads, not a measured process footprint. Runtime alignment,
auxiliary tensors, Python/native objects, allocator reservations, activations,
load buffers and temporary copies are additional or require separate accounting.
CPU offload on this UMA host does not by itself remove bytes from whole-host
resident memory. Reclaimable file mappings and a deliberately smaller disk-backed
working set would be a different, separately tested loading/execution policy.

At a 40 GiB floor the incremental budget from the stated start is
`(115 - 40) * 2**30 = 80,530,636,800` bytes. The selected payload exceeds it by
**4,165,382,372 bytes (3.879314634948969 GiB)** before engine and temporary costs.
Even removing the entire vision payload would not make this representation fit
that envelope; vision removal would also change the requested model scope.

At an 18 GiB floor, adding the already measured physical cache allocation of
9,565,306,880 bytes gives **94,261,326,052 bytes (87.78770086541772 GiB)**. Only
**9,891,630,876 bytes (9.212299134582281 GiB)** remain within a 97 GiB incremental
budget. This remainder must cover every unmeasured cost above. It is not a fit
claim, and the unmeasured profile's 90G/96G containment values must not be used as
measured limits.

## Actual pinned load and finalization path

All source paths below are relative to the roots recorded in source-pins.json.
The vLLM base is `878631b6079d2cf9fb80830ef9cb41b43aded098`, plugin base is
`6b26e5c9d35a455f61ee5b5529151ff2e5c08103`; this audit hashes the actual prepared
005 source, including the reviewed local BF16 geometry correction, rather than
assuming that the base revisions describe the final source bytes.

1. **Complete model allocation precedes streaming.**
   `vllm/model_executor/model_loader/base_loader.py:43` selects the target device,
   calls `initialize_model` under that device context, then loads weights, then
   calls `process_weights_after_loading`. A prefix-limited weight iterator alone
   cannot make full constructor allocation safe. A bounded probe must restrict
   the instantiated modules or use a genuinely nonallocating metadata census.

2. **MTP is excluded; embedding, head and vision are not.**
   `vllm/models/glm5next/nvidia/model.py:635` constructs the configured 45 main
   layers. Its `load_weights` at line 710 calls the skip helper at line 1055
   before copying any prediction-layer tensor. The checkpoint declares
   `num_hidden_layers=45` and `num_nextn_predict_layers=1`. The census executes
   that exact source helper on ordinary and MTP names, confirming both
   `layers.45.*` and `model.layers.45.*` are skipped. The multimodal wrapper maps
   `model.language_model.*` into the nested text model and constructs the vision
   tower at line 1020. The checkpoint has untied word embeddings, so embedding
   and output-head tensors both belong in the budget.

3. **The ordinary loader does not provide persistent pinned staging.**
   `default_loader.py:248` chooses the ordinary safetensors iterator unless an
   alternate load format or threading option is explicitly selected.
   `weight_utils.py:914` uses `safe_open(..., framework="pt")` and `get_tensor`
   on the normal path. `default_weight_loader` at line 1222 copies those tensors
   with `param.data.copy_`. The plugin's expert loader at `exl3.py:1928` similarly
   uses `loaded_weight.detach().contiguous()` and `dest.copy_`; no pinning or
   persistent staging buffer is introduced there. The pointer-table builder at
   line 705 constructs CUDA tensors from Python pointer lists. Those are small
   startup copies, but are not evidence of a persistent pinned transfer policy.

4. **Overlay indexing does not imply all stale source bytes disappear from I/O.**
   `tools/dense_overlay.py:260` links base shards and adds the overlay file. Its
   rewritten index replaces target names, while the ordinary safetensors
   iterator enumerates the keys of selected shard files. Plugin
   `exl3.py:2857` explicitly discards stale BF16 weights for EXL3 shards after
   shape validation. Contiguous mmap views may avoid reading their full payload;
   an eager whole-shard read will not. The largest base shard contains
   **3,906,091,028 payload bytes**. An eager strategy reads a complete file into
   memory before constructing its state dictionary; multithreaded/eager/prefetch
   paths must not be enabled in a bounded probe without separate peak accounting.
   MTP skipping likewise does not prove that an eager iterator avoided its bytes.

5. **Packed routed experts do not double into a permanent dense arena.**
   Plugin `exl3.py:1823` allocates fused gate/up and down packed Parameters; it
   explicitly rejects dense `w13_weight`/`w2_weight` allocation. Finalization at
   line 1982 creates per-expert `LinearEXL3` handles over contiguous slices.
   `exllamav3/modules/quant/exl3.py:20` retains the passed trellis and sign tensors;
   `exllamav3_ext/libtorch/linear.h:31` stores/moves tensor handles, not reconstructed
   BF16 matrices. The original Parameters and handles therefore share the packed
   storage in this path. Alternate codebook marker slots add at least 145,152
   bytes beyond the expert checkpoint's one-marker-per-projection payload.
   Pointer tables, native object overhead and scratch remain additional costs.

6. **Dense finalization has real overlapping storage.**
   Plugin `exl3.py:2674` initially allocates a fused trellis covering all output
   shards, including the columns corresponding to retained BF16 shards, plus
   sign vectors, both codebook markers and BF16 staging. Under TP1, BF16 dtype
   and `GLM53_EXL3_BF16_SHARD_FIX=1`, integer replay of those formulas across the
   191 overlay modules gives **3,652,228,616 constructor bytes**. That includes
   **89,128,960 BF16 staging bytes** and exceeds overlay payload plus its BF16
   staging by **23,141,660 bytes**. This is only this module family's parameter
   census, not a full-model allocator measurement.

   At `exl3.py:3013` a multi-shard trellis slice is made contiguous, and at line
   3033 BF16 staging is cloned. Original fused Parameters are deleted only after
   the handles and BF16 clone exist. The largest indicated per-module extra copy
   is **52,953,088 bytes**, for a KDA mixed projection. Do not add that copy for
   every layer as if all finalization copies must coexist; conversely, do not
   assume freed allocations immediately reduce reserved CUDA memory or RSS.
   Surviving views can retain whole sign-vector/marker backing stores.

7. **Load-only memory is not serving memory.**
   The plugin caches fused scratch by geometry at `exl3.py:705`. ExLlama's
   `GTensorCache` also retains buffers by shape, and `LinearEXL3.forward` can
   reconstruct a temporary FP16 matrix for larger batches at
   `modules/quant/exl3.py:163`. That reconstruction is a runtime workspace, not
   a second persistent copy of every expert. Multimodal activations and engine
   request buffers still need measurement. A tensors-only storage counter would
   miss native handles and retained allocator memory.

## Next bounded measurement and fixed acceptance

The next useful experiment is **cache-off load-path sizing**, not another full
load attempt with the already-falsified 40 GiB envelope. Nothing in this document
changes the mandated whole-host floors or authorizes lowering one.

1. Freeze a sizing-only test covering the largest actual module shapes: one
   routed-expert layer, a mixed KDA dense projection, a representative merged
   MLA projection, and a bounded stream matching the largest tensor/shard shape.
   Retain only the chosen module(s); reject construction of the complete model.
   The largest selected main-layer payload is 1,938,060,532 bytes; embedding and
   head tensors are each 1,268,776,960 bytes. Use these exact shapes to select
   probes, not reduced dimensions presented as full-load evidence.

2. Before any measured large transfer, select a reusable, bounded pinned staging
   implementation or explicitly measure and justify the pageable alternative.
   Record the actual source buffer's pin status, staging allocation size and
   reuse count, copy sizes and completion synchronization. The default loader's
   code is insufficient to claim pinned transfers. Use byte-identical transfers;
   no quantization or quality change is part of this sizing experiment.

3. Preserve RED evidence and freeze code, runtime, fixture/configuration and
   tool hashes before a fresh public seed. Run under the inference lock and a
   fresh hardened cgroup, with stable start at least 110 GiB, a 40 GiB kill floor,
   no swap allowance, a fixed wall-clock timeout, at most two compilation jobs
   when compilation is actually needed, and continuous independent identity.
   Prewarm and freeze any needed kernels before a confirmation measurement.

4. Record timestamped whole-host MemAvailable, cgroup current/peak/events, process
   RSS/PSS and anonymous/file-backed breakdown, CUDA allocated/reserved/peak,
   pinned allocations, and unique-storage addresses/byte sizes at construction,
   each streamed copy, finalization, and cleanup. Retain all samples. Require
   loaded/finalized tensor bytes and storage aliasing to match the fixed source
   contract, untouched inputs, no extra tensors or duplicate IDs, clean exit and
   no surviving process/cgroup. OOM, timeout, identity gaps, Xid, new swap, or a
   floor violation are FAIL, irrespective of the measured component result.

5. Keep separate maxima for persistent incremental memory and the largest
   concurrently live loading/finalization/activation peak. A later full-load
   proposal must fit both its startup peak and running peak. For the running
   four-slot model, the frozen payload plus measured cache is already
   94,261,326,052 bytes. Any proposed 18 GiB-envelope sizing must conservatively
   bound all additional and outside-cgroup pressure within the remaining
   9,891,630,876 bytes at the stated start, with explicit measurement uncertainty.
   Derive cgroup ceilings against actual physical memory and outside occupancy;
   cgroup accounting and CUDA allocation alone are not whole-host availability.

If component measurements cannot establish such a conservative bound, retain
**NO_RESULT for 18 GiB full-load feasibility** and investigate a separately
bounded representation/working-set alternative. Passing a component sizing
probe does not satisfy the required full-load, stability, fidelity or direct
context-capability gates, and does not by itself satisfy every prerequisite for
an 18 GiB-floor qualification campaign.
