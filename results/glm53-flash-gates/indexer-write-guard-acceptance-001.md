# Indexer replay write guard acceptance

New evidence-only component; production selection remains off. No changes to
the frozen numerical probes, loaders, fixture or existing service boundary.

Acceptance requires all real subprocess tests in
`scripts/tests/test_glm53_replay_write_guard.py` to pass, with no skips:
explicitly selected enforcement permits evidence writes and cache reads,
rejects kernel mutation and the pinned DeepGEMM `tmp/uuid` creation seam,
rejects overlapping/symlink allowances, writable descriptors, hardlink aliases,
shared writable mappings and existing threads, and is inherited by future
threads and executed children. Existing completion pipes must remain usable.
Default-off selection must return without accessing even nonexistent roots.

The helper requires Landlock ABI >=3, single-threaded unprivileged startup,
matching real/effective/saved/filesystem identity and no active capabilities.
It sets and verifies no-new-privileges, binds specific directory/device inodes,
and rejects failed admission or enforcement. Device tests initially use only
`/dev/null`; CUDA device coverage still requires the contained native replay.

This boundary applies to this process and future children. It does not restrict
unrelated host processes, chmod, reads, execution or device IOCTLs. It does not
prove that unrelated compilers cannot write into allowed scratch space. Native
DeepGEMM cache inventory checks, actual missing-key controls and frozen-loader
selection remain required. The constructor's `nvcc --version` is allowed.

Design follows the kernel's [Landlock API documentation](https://docs.kernel.org/userspace-api/landlock.html),
independently checked by the persistent adversarial reviewer. Host read-only
ABI query returned 7. Enforcement must precede Torch/Triton imports.
