Indexer preparation 001 — FAIL
================================

The frozen contained attempt stopped during Python imports, before configured/profiled rows or any indexer computation. Importing sparse_attn_indexer_kpool first recursively loads the GLM package, whose attention module imports SparseAttnIndexerKpool from the still-partially-initialized module. The exact traceback is retained. No generated JIT files exist.

Source 67b8e067, freeze 1788973280.1809115, drand round 6451409, seed 1237677668765861990. The wrapper returned 1; identity has 37 samples and no surviving process group, but correctly reports FAIL because successful completion handshake is absent. The controller does not replace this with a host or inner PASS. All raw observations and code are copied byte-for-byte.

The next bounded correction is import ordering in the probe, backed by a fresh-process CPU import regression. No engine or frozen reference changes are justified. No weights loaded, no model tokens processed, no context or performance claim.

Postflight detail: 33 memory samples reached a minimum of 119,143,808 KiB available. Cgroup swap stayed zero. Whole-system counters show pswpin +1 and pswpout +0; the strict no-swap gate would reject that delta as well. No host PASS is claimed. Raw records and original summary are unchanged.
