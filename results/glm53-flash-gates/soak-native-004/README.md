# Scheduler 256/64 first-window attempt: FAIL

The optional aggregate-1M profile started through `scripts/93_profile_serve.sh`
with four 262,144-token slots. This attempt used 4,224-token requests as a bounded
correctness/admission falsifier; it is not a context or production-speed result.

All 17 completed native replies independently rescore correct, including negative
controls. Admissions per worker were `[5, 4, 4, 4]`, below the fixed five each.
Every admission occurred before 300 seconds; drain completed at
324.009181744 seconds. No full 1,800-second durability run was admitted.

Whole-host used swap stayed unchanged and GLM cgroup swap remained zero, but
`pswpin` increased by one page. Attribution is unknown and the no-swap-I/O check
fails. Minimum external available memory was 18.20391845703125 GiB.
Containment events, kernel OOM/Xid, process identity, clean shutdown, memory
recovery above 110 GiB and default/proxy/guard checks passed.

Fourteen new generated kernel-cache files appeared, with no changed/missing baseline files;
two additional files are allowed metadata. Frozen-kernel confirmation is
`NO_RESULT` and would require preparation freeze/replay. Post-stop model/runtime
inventory verification passed. The next declared bounded scheduler alternative
is 512/128; no speed or fidelity benefit is assumed.

`manifest.json` binds every archive part/member, `raw.jsonl.gz`, the exact frozen
sources and `summary.json`. Concatenate the numbered parts to read the gzip tar.
API credentials are excluded. Review receipts and any publication-only metadata
are retained separately; all earlier attempts remain unchanged.
