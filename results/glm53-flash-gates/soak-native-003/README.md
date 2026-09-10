# Native four-client durability003: FAIL

The experimental aggregate-1M profile completed its full 1,800-second admission
period and drained by 1,853.909702124 seconds. All 68 native requests returned
correct final retrieval answers and negative controls. Four clients overlapped,
health probes passed, and the profile stopped cleanly.

The fixed gate failed. Each worker had to admit at least five requests in both
the first and final 300-second windows. Actual counts were `[3, 3, 4, 3]` and
`[2, 3, 3, 2]`. The unchanged scorer's FAIL is retained. These count failures
alone do not establish a runtime malfunction. Whole-host swap use also increased
by 108 KiB after preflight and stayed elevated after shutdown. GLM's cgroup swap
samples and peak were zero; attribution of the host increase is unknown.

The guard recorded 8,142 identity samples and no surviving process group. Memory
recovered to 114.882 GiB. Default, proxy and guards stayed unchanged. All 2,463
prepared compiled inputs and the closed runtime/model inventories verified after
stop. The 22 existing scorer mutation/regression tests passed. Independent client
and terminal-host reviews are retained in the archive.

Source `2f0cff89e6cf6aa664e7bbeb6fae59b07292940e` was frozen before public drand
round 6453189. This candidate jointly enabled startup allocator cleanup and
verified-model file-cache advice after the preserved [startup001](../soak-startup-001/README.md)
and [startup002](../soak-startup-002/README.md) failures. The advice reclaimed model
file-cache pages before CUDA initialization; this startup loaded and served with
no recorded kernel OOM/Xid. No isolated causal or performance claim follows.

This was a short-prompt durability workload with 4,224 actual input tokens per
request. The earlier [context007](../context-direct-007/README.md) remains the
separate aggregate million-token capability result on its preceding startup
configuration. Paired fidelity, a passing durability gate and production
switching remain pending. Qualified production performance is not yet measured.

`manifest.json` hashes the summary, compressed raw observations and two archive
parts. Concatenate the parts in numerical order to read the gzip tar archive.
All 5,140 archived files were round-trip verified; the API key is excluded and
its bytes were scanned for before publication. Every original request stream,
freeze, public-randomness receipt, host observation and failure is retained.

The public manifest's uncompressed-raw checksum uses the repo's nested checksum
format after a preserved [publication-format rejection](../soak-publication-format-001/README.md).
The original reviewed manifest is retained there. Its exact reversible mapping
to the public manifest leaves archive, raw and summary bytes unchanged.
