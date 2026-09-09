# GLM-5.3-Flash integration research

GLM-5.3-Flash should enter this repository as an optional CUDA profile with
1,048,576 aggregate context tokens divided into four 262,144-token request
slots. The strongest initial single-Spark candidate is the pinned vLLM/EXL3
implementation with K2 routed experts and a higher-precision dense overlay.
It still needs local runtime, fidelity, memory, multimodal, concurrency and
switching qualification. No public result establishes this repository's
complete acceptance contract, and GLM must not become the default.

This assessment covers public material available on September 8, 2026 and
the repository's preserved local evidence. Current implementation progress is
recorded separately in [STATUS.md](../results/glm53-flash-gates/STATUS.md).
Source support, advertised context, successful allocation and measured
application capability are distinct claims throughout this report.

## Model and backend selection

The official model uses hybrid sparse and linear attention. It is a new
architecture, so the GLM-5.2 cache geometry and native engine patches cannot
be transferred by renaming the model. Z.ai documents multiple serving
frameworks and exposes `reasoning_effort` levels `low`, `high` and `max`;
chat applications should explicitly set `clear_thinking=true`. The proposed
interactive profile selects `low` deliberately, and benchmark comparisons
must record that setting.[^1]

The single-Spark EXL3 recipe provides concrete ARM64/SM121 integration work:
specialized sparse-MLA support, a K-pool tail correction, dense quantization
plumbing and a native EXL3 plugin. Its dense overlay replaces selected BF16
linears while retaining the K2 routed-expert pack. This is a credible starting
point for local reproduction, although its small-context measurements and
allocated-cache figures do not qualify four simultaneous long requests.[^2]

| Candidate | Reason to consider it | Remaining qualification boundary |
| --- | --- | --- |
| Pinned vLLM + EXL3 K2/dense overlay | Existing single-Spark integration, native routed-expert kernels and multimodal model path | Actual runtime closure, FP8-KV fidelity, four-slot occupancy and safe Python lifecycle |
| Current upstream vLLM | Architecture support merged upstream; reduces future fork maintenance | EXL3 format and Spark-specific kernel compatibility must be independently demonstrated |
| ExLlamaV3 1.4.8 | New GLM cache-quantization and allocation work | Replacing the pinned engine creates a new candidate with fresh fidelity and serving qualification |
| Pinned llama.cpp GGUF | Alternative engine and tensor format if the EXL3 branch fails | ARM64/CUDA correctness, multimodal projector, memory fit and full aggregate-context test |
| Multi-GPU TR3/TrellisMX recipes | Useful kernel and evidence references | Hardware topology and memory assumptions differ from one Spark |

Upstream vLLM added GLM-5.3-Flash support in commit `98ed0856`. Consequently,
older instructions saying stock vLLM lacks the architecture are stale. That
merge does not itself establish EXL3 pack support or qualify this Spark's
complete attention and multimodal paths.[^3] ExLlamaV3 1.4.8, released
September 6, adds GLM cache quantization and allocation improvements. It is
a justified later alternative, rather than a silent substitution into a
candidate based on 1.4.4.[^4]

Unsloth publishes a separate GGUF deployment route. The fallback must pin its
engine revision, exact quantization and matching multimodal projector; a
working text-only launch would leave the requested image/video integration
unfinished.[^5] Recent TrellisMX material targets four 96 GB Blackwell GPUs
and explicitly distinguishes its configured million-token cap from tested
accuracy. It is not a one-Spark fit result.[^6]

## Reuse from the local GLM-5.2 and Qwen work

The most valuable reusable work is the production and evidence machinery.
The GLM-5.2 production profile is an owner-accepted native streaming engine
with a 32,768-token cap. Its large pinned expert arena, streaming assumptions
and model-specific quality decision belong to that candidate. They cannot
qualify a resident EXL3 model with hybrid attention.[^7]

| Existing component | Reuse | Required GLM-5.3 adaptation |
| --- | --- | --- |
| `52_engine_switch.sh` transaction | Serialization, authenticated validation, previous-profile state and rollback | Optional alias, Python worker identities, complete environment rendering and cgroup cleanup |
| Qwen declarative profiles | Separate profile per topology, aggregate context accounting and existing render fixtures | 262,144-token vLLM request cap and four sequences; preserve all existing snapshots |
| GLM cgroup wrapper | Fresh containment, inference lock, memory start gate, timeout and cleanup | Python-specific provenance before large model runs |
| GLM external memory observations | Whole-system UMA measurements, swap and kernel-error rejection | Include every worker and terminate the verified cgroup on failure |
| Closed inventory verification | Exact file coverage, hashes, sizes and replacement detection | Bind interpreter, packages, native libraries, model, tokenizer and processor to the candidate |
| GLM paired fidelity formula | Fixed 100-case token-weighted statistics | Bind case IDs, teacher and candidate inputs, vocabulary, references and capture provenance |
| Immutable evidence directories | Preserve raw failed and null attempts | New GLM-5.3 manifests and terminal verdicts; no inherited performance claims |

The existing generic profile launcher is insufficient for this Python model:
its launch behavior must not bypass the campaign's lock, containment or
identity requirements. The integration currently rejects that route before
creating runtime state. The production switch likewise rejects GLM before
changing state while its profile is estimated. Editing a status string alone
cannot authorize the unfinished lifecycle.[^8]

The established switch's readiness request uses `model: "default"`, and the
authentication proxy forwards request bytes. A future vLLM launch should
preserve that API compatibility with a secondary served-model alias while
keeping `glm-5.3-flash` as the canonical model identity. An API alias does not
change the default startup profile. Status, restore and rollback must still
select GLM only through an explicitly qualified optional profile.

## Context and memory contract

The requested topology is one million aggregate tokens, not a million tokens
in each of four requests. vLLM's `--max-model-len` applies to each sequence;
the profile therefore combines `262144` with `--max-num-seqs 4`. Neither flag
reserves or proves simultaneous residence. The direct capability gate requires
four distinct simultaneous requests, each processing at least 250,000 actual
input tokens, with at least one million collectively resident before eviction
or reuse. Retrieval at multiple positions, negative controls and completed
timestamped generation are required for every slot.[^9]

GLM-5.3's sparse/linear mixture makes GLM-5.2's bytes-per-token estimate
inapplicable. The local source audit executes the pinned attention
canonicalizer and finds that both `auto` and `fp8` select packed
`fp8_ds_mla` on the relevant Spark backend. The profile now declares FP8
explicitly. This source result identifies the representation; it does not
demonstrate fidelity or actual allocation under load.[^10]

The memory budget must include resident weights, sparse-MLA values and
indices, recurrent KDA state, vision tensors, persistent staging, scheduler
workspaces, graph pools, allocator reserve and engine RSS. A nominal weight
size plus a nominal KV pool omits important allocations. The initial profile
therefore carries a conservative, deliberately unqualified estimate and
admission ceilings. A measured preflight must replace those assumptions
before any large load is authorized.

The safe initial operating envelope requires at least 110 GiB stable
`MemAvailable` before build/load, fresh cgroup containment with swap disabled,
and an external whole-system floor. The campaign starts with an 18 GiB floor;
the cache-off streaming probe uses 40 GiB. Any qualification observation below
10 GiB available fails, even if the engine continues responding. A cgroup's
accounted memory cannot substitute for whole-system UMA observations.[^9]

Disk space also constrains reproduction. The pinned K2 artifact inventory
contains about 97.7 billion weight bytes. The dense source is another large
pack, so only selected dense tensor ranges should be downloaded. The local
overlay must use hard links or copies because closed artifact inventories
reject symlinks. Reserve disk for build outputs, immutable evidence and
download partials before fetching weights.[^11]

## Multimodal and request semantics

The request policy is four images OR one video across the entire conversation,
with video sampled to at most 16 frames. Image and video counts alone are
insufficient: frame extraction can expand a video substantially. Request
overrides, cached media identifiers and ambiguous content shapes also need
to be controlled before media fetching or model dispatch.

The local source audit reproduced a frame sampler selecting up to 1,200
frames despite a nominal 16-frame request. The first patch capped a processor
method but missed the normal loader's direct call to a shared helper. The
reviewed replacement caps that helper and sets the loader's `max_frames`.
Tests exercise the actual source functions and loader metadata, including
the path that disables later processor resampling.[^10]

The middleware uses closed content shapes and rejects UUID/media ambiguity,
processor overrides and multiple completions. Accepted request bytes are
replayed unchanged. These checks are request-level admission controls, not
per-token instrumentation. They still need installed-runtime and authenticated
HTTP tests, including genuine image/video interpretation, malformed media,
tool calls and reasoning parsing. CPU source tests are useful falsifiers but
are not multimodal capability evidence.[^8]

## Fidelity and public evidence limitations

Lossy weights and packed FP8 KV require the repository's fixed 100-case paired
gate. Token-weighted delta NLL and its one-sided 95% upper confidence bound
must both be at most 0.01. Top-1 loss and its upper confidence bound must both
be at most 0.5 percentage points. An improved average cannot compensate for a
failed bound. Passing these limits also does not authorize adoption of a
nonzero fidelity delta; that decision requires the measured performance it
buys and the owner's acceptance.[^9]

The public fidelity-suite correction identifies post-final-normalization
captures and invalidates prior assumptions about the scorer's input location.
It also distinguishes native serving from replay. Shared-head replay cannot
measure a changed output head, and a short-context replay result cannot
qualify the long-context cache path. The reference and scoring seam must be
re-established before those artifacts can support local acceptance.[^12]

Another public source provides BF16 teacher logits and exact token windows,
with full vocabulary and aggregate receipts. That offers a possible reference
without loading the BF16 teacher on this host. It is a large dataset, however,
and its calibration/qualification partition, checkpoint identity and causal
token alignment must match the local fixed suite. Selected authenticated
windows may be streamed; unsupported or misaligned references produce
`NO_RESULT`, not a substitute quality score.[^13]

KL divergence, top-1 agreement with a teacher, ground-truth top-1 accuracy,
and delta NLL measure different properties. Public tables using one metric
cannot be inserted into the repository scorer as another. Candidate captures
must preserve the actual input tokens, all scored positions and the complete
normalization needed for the specified metric. Missing cases, duplicate IDs,
unequal fixtures, malformed numbers and stale binaries must fail closed.

## Reproducible build and serving qualification

The source lock selects exact recipe, vLLM, EXL3-plugin and ExLlamaV3 commits.
Preparation uses separate clean worktrees and records complete binary diffs.
Local review removed unused DFlash allocation/MTP changes from the no-spec
baseline, pinned mutable CUTLASS/Triton tags and bound plugin headers to the
prepared ExLlamaV3 tree. The first candidate deliberately leaves speculation
and prefix caching off so correctness and occupancy have a simpler baseline.[^11]

Dependency preparation exposed three concrete portability issues: a stale
FlashInfer URL, a required ARM64 source package and missing system-Python
headers. The isolated managed interpreter resolved the header problem. A
separate NVIDIA wheel advertised AArch64 in its filename but SBSA in internal
metadata; the recorded repair changes only WHEEL/RECORD and preserves every
library byte. It makes no additional ABI-compatibility claim.[^14]

The build driver limits CUDA/C++ compilation to two jobs and NVCC to one
thread per invocation. vLLM also needs pinned Rust, a fresh Cargo home,
`CARGO_BUILD_JOBS=2` and locked resolution for both Rust artifacts. Cargo's
job limit is separate from CMake and Make settings.[^15] A successful wheel
build still precedes import checks, bounded native-kernel comparisons,
JIT-cache preparation, runtime inventory freezing and large-model admission.

The production lifecycle must bind the unit invocation, cgroup, frontend
PID/start time, interpreter, argv, environment, runtime and model inventories.
Workers may outlive the frontend or use separate process groups, so cleanup
must prove the entire verified cgroup empty and listener closed before
restoring another model. Readiness must validate authentication, canonical
model identity and semantics before committing active state.

Performance remains **not yet measured** until the fastest qualified serving
path passes with diagnostics disabled. Comparisons require five fresh-server
ABBA/BAAB blocks, equal fixtures and repository-derived confidence intervals.
Decode uses token timestamps over at least 128 generated tokens; prefill uses
actual evaluated tokens and synchronized time. Smoke outputs, internal
capacity estimates and build success are not substitutes.[^9]

## Sources

[^1]: Z.ai, [GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash), accessed September 8, 2026.
[^2]: vcruz305, [GLM-5.3-Flash EXL3 single-Spark recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe), current README and dense-overlay discussion, accessed September 8, 2026.
[^3]: vLLM project, [GLM-5.3-Flash support commit](https://github.com/vllm-project/vllm/commit/98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5), September 2026.
[^4]: Turboderp, [ExLlamaV3 1.4.8 release](https://github.com/turboderp-org/exllamav3/releases/tag/v1.4.8), September 6, 2026.
[^5]: Unsloth, [GLM-5.3-Flash GGUF model card](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), accessed September 8, 2026.
[^6]: Brandon Music, [GLM-5.3-Flash TrellisMX MXFP8](https://huggingface.co/brandonmusic/GLM-5.3-Flash-TrellisMX-MXFP8), accessed September 8, 2026.
[^7]: Local repository, [GLM-5.2 production profile](../configs/profiles/glm-5.2/cuda-spark-128g.json) and its linked owner-accepted candidate evidence.
[^8]: Local repository, [admission contract](../scripts/lib/glm53_contract.py), [request policy](../scripts/lib/glm53_runtime_policy.py), and [GLM-5.3 test suite](../scripts/tests/test_glm53_contract.py).
[^9]: Local repository, [fixed GLM-5.3 acceptance plan](../results/glm53-flash-gates/PLAN.md), `AGENTS.md`, and `scripts/glm52_goal.py::quality_verdict`.
[^10]: Local repository, [source audit](../results/glm53-flash-gates/source-audit-001/summary.json), [frame review](../results/glm53-flash-gates/frames-review-001.md), and retained `frames-red-001`/`frames-red-002` attempts.
[^11]: Local repository, [pinned source and artifact lock](../configs/build-manifests/glm53-flash-sources.json), [prepared source manifest](../results/glm53-flash-gates/source-preparation-003/manifest.json), and adjacent complete diffs.
[^12]: malaiwah, [GLM-5.3-Flash fidelity suite](https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1), September 8 correction; see also the local plan's reference limitations.
[^13]: Brandon Music, [BF16 teacher-logit dataset](https://huggingface.co/datasets/brandonmusic/GLM-5.3-Flash-BF16-Teacher-Logits) and [publication commit describing its receipts](https://huggingface.co/brandonmusic/GLM-5.3-Flash-tr3-4bpw/commit/4739eb1bcfd478e8a32da6358908567bc3a9ac51), September 2026.
[^14]: Local repository, [runtime dependency lock](../configs/build-manifests/glm53-runtime-dependencies.json), [NVIDIA repair manifest](../results/glm53-flash-gates/wheel-repair-001/manifest.json), and individually retained dependency/install attempts.
[^15]: Rust project, [Cargo configuration: build jobs](https://doc.rust-lang.org/cargo/reference/config.html#buildjobs), accessed September 8, 2026; local [Rust artifact lock](../configs/build-manifests/glm53-rust-toolchain.json).
