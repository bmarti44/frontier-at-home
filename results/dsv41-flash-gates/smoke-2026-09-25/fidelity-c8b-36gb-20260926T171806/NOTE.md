# Fidelity smoke at a 36 GB expert cache: INTERRUPTED at the owner's request (not a result)

The owner asked, on 2026-09-26, to pause all GPU work. The run was stopped with `systemctl --user stop` after 109 of 112 cases, about 32 minutes in. The bundle's FAIL comes from the interruption: incomplete `cuda.tsv`, no kernel.log, and the wrapper killed. It is not a fidelity or safety verdict.

Before the stop, the kernel journal (`kernel-events.txt`) showed no NVRM, Xid or OOM line. On resume, re-run `run_smoke.sh fidelity OUTDIR 36gb` in a fresh directory.
