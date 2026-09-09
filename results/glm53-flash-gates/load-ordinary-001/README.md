# ordinary component load001 — PASS

Scope: model-free component loading/storage only. No model weights, forward
inference, fidelity, context or production performance result.

Frozen source: `481c0a94c2e6963bc5c911a2271bdfaf9ebea4a7`. Public drand round6451189,
seed17259192682203841248, obtained after freeze and verified with the retained BLS receipt.
All1 selected tensors (1268776960 bytes
per stage) match independent canonical input bytes through persistent pinned
upload, actual parameter loaders and final native handles. Exact retained GPU
storage is1268776960 bytes. Unrounded observed phase counters
and storage descriptors are in checks/raw.jsonl.

Host and identity PASS: minimum MemAvailable117800564 KiB;
cgroup peak2099068928 bytes;62 memory
samples and66 identity samples. No new swap; complete
process/unit/cgroup cleanup. Full frozen runtime and code inventories matched
before and after. No generated runtime artifacts were accepted.

Large generated source payload remains at its original local path, bound in
archive.json and checks/fixture.json. All other files were copied byte-for-byte
and independently hash-checked. Frozen generator+case+seed reproduce the complete
canonical input file. These synthetic fixtures were freshly generated and do not
establish cold checkpoint I/O behavior or a full-model memory ceiling.
