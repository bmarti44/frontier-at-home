# Native BF16 layer attempt 002: FAIL; feasibility NO_RESULT

The reviewed probe passed the corrected runtime freeze and later verified public
seed. It then failed the unchanged host gate at the first shard's verification/
copy stage: five host pages (20 KiB) were written to swap. No completed shard
receipt, GPU forward or output was recorded. This is not native reference data,
a fidelity result, a context result or a serving-speed result.

External memory stayed above 100 GiB. Sampled probe cgroup swap stayed zero,
but the owner and trigger of the global writes are unknown. The probe exited 1;
the identity guard correctly reported missing successful completion. The process
group and cgroup were gone afterward and memory recovered. Keep the overall
FAIL; individual cleanup observations do not promote the result.

The archive retains the frozen source/configuration/runtime inventory, pinned
model shard metadata, later public seed, controller invocation, original raw
logs, memory/identity sampling, failures and terminal cleanup. Publication
verified every archived member byte against the original and all frozen file
bindings. Raw source lines are also provided in raw.jsonl.gz.

Docker service/socket and containerd were observed stopped before launch. Earlier isolated attempts
also failed global swap gates. No new warmup or additional daemon-stop variant
is proposed; native GPU feasibility and full qualification remain unresolved.
