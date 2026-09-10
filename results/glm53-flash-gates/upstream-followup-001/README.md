# Upstream follow-up: no demonstrated local crash fix

The [upstream crash discussion](https://github.com/vllm-project/vllm/issues/54317)
reports improvement from the [ordered state-copy fix](https://github.com/vllm-project/vllm/pull/50729).
Installed source already contains ordered overlap handling, so simply applying
that fix again is not a justified next step.

The [metadata-lifetime fix](https://github.com/ZJY0516/vllm/pull/7) forwards
positions and retains mapping storage. The local recipe already forwards
positions and writes its mapping in place; upstream instead owns a separate
buffer in the builder. This source comparison does not qualify their equivalence.
The live server and failed full-context run use eager execution, so the upstream
graph-replay failure is not established as the local cause. The local recipe also
already selects the 64-wide indexer pages required on SM12x.

The [short-decode optimization](https://github.com/ZJY0516/vllm/pull/12) enables
an existing causal-index shortcut on CUDA and removes a dispatcher wrapper. It
is absent locally and is a bounded candidate for later speed measurement. Its
shortcut applies only while the sequence fits within the top-k selection; it
does not establish long-context acceleration or a fix for the earlier failure.

No runtime files, launch settings or live processes were changed. The installed
file hashes and exact source markers are recorded in `summary.json`. This audit
is NO_RESULT for crash attribution and supplies no new performance claim.
