# Real-shard InstantTensor loader smoke

The installed vLLM InstantTensor loader returned all 1,280 tensors from a
completed real GLM shard. All 675,022,080 tensor bytes matched safetensors after
GPU transfer. Identity monitoring passed, the hardened wrapper exited zero,
no new swap occurred, and containment was removed. This is a loader bring-up
smoke, not model loading, fidelity, context qualification or serving speed.

`attempt.tar.gz` retains the exact script, source digest, explicit AIO settings,
raw per-tensor digests, identity observations and host samples. The original
remains at `/home/bmarti44/.cache/glm53-flash/real-loader-smoke-001`.

The existing InstantTensor AIO backend registers one persistent host arena and
reuses it after copy completion. Startup requests 24 MiB pinned I/O staging and
a 1.25 GiB device ring; for this single shard the library reduces the ring to
its smaller total tensor size. vLLM selects `copy=True`, so yielded tensors own
their storage. No replacement loader or new runtime patch was required.
