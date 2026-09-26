# Fidelity smoke at a 36 GB expert cache: INTERRUPTED at the owner's request (not a result)

The owner asked, on 2026-09-26, to pause all GPU work. The run was stopped with `systemctl --user stop` after 109 of 112 cases, about 32 minutes in. The bundle's FAIL comes from the interruption: incomplete `cuda.tsv`, no kernel.log, and the wrapper killed. It is not a fidelity or safety verdict.

**Correction (same day).** An earlier version of this note said there was no kernel fault. That was wrong. `kernel-events.txt` records `NVRM ... Out of memory [NV_ERR_NO_MEMORY] ... _memdescAllocInternal` at 17:43:54 UTC, about 26 minutes in. That is roughly when the long-prompt cases begin (case_100, where the 42 GB run printed "moe gateup y-indirect q8 staging engage"). Lowering the expert cache from 42 GB to 36 GB did not prevent the line.

The working hypothesis is that the failure comes from a driver-side allocation during long-prompt prefill, not from the cache size. Next diagnostic on resume: reproduce with the long cases alone (case_100 to case_111) and a timestamped log, to identify the allocation. A full re-run at an even smaller cache would not answer this.
