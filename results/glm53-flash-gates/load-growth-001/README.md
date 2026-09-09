# Two live expert modules — PASS

Scope: model-free incremental loading/storage only. Frozen source9ca1273f,
freeze1788968063.6680865, post-freeze drand6451235 published1788968070,
seed1148146129390848175. BLS receipt and two relay responses are retained.

Both independent full288-expert modules load3,456 selected tensors each.
Every source, uploaded, loaded and finalized byte matches. Module one remains
owned and unchanged after module two completes, including backing identities,
native handles and pointer-table contents. Shared scratch and GTensorCache
identities and key coverage remain stable.

Exact storage union:3,947,169,792 bytes. Observed final CUDA allocated:
3,947,176,960 bytes (7,168 bytes allocator rounding). The second module adds
1,822,587,392 observed CUDA allocation bytes versus1,822,583,808 exact private
parameter/table bytes. Observed retained RSS, PSS and anonymous PSS increase by
4,500 KiB; file PSS is unchanged. Raw phase counters are unrounded in summary.json
and checks/raw.jsonl. These two observations do not establish linear growth over
all model layers or include surrounding model constructors/forward workspaces.

Host/identity PASS: minimum MemAvailable115,304,528 KiB; cgroup peak2,880,344,064
bytes;102 external memory samples and109 identity samples. No new swap, kernel
errors or generated runtime artifacts. Frozen runtime/code inventories match
before and after. Process/unit/cgroup cleanup completes. Cgroup memory remains
an incomplete account of UMA GPU allocations; whole-host observations are required.

Large canonical synthetic input remains at the original local path in archive.json,
independently bound by the complete generated-file hash. All other source/evidence
files were copied byte-for-byte and rehashed. This is not cold-checkpoint I/O,
model fidelity, context, performance or full-model memory-fit evidence.
