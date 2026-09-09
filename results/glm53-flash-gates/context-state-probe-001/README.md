# Bounded convolution and recurrent-state diagnostic

Use four 32-token sequences, 64 heads ×128 dimensions, the 24,896-wide fused
projection view and the serving engine's 4,456,448-byte padded state pages.
Bind the pages through the unchanged native `MambaBase.bind_kv_cache` method.
Use distinct nonzero state IDs and nonzero initial states, then advance twice.

Synchronize after convolution, state gathering, recurrence and state scattering
to localize any CUDA failure. Compare convolution with the repository's existing
analytic tolerance (absolute 0.001 + relative0.01); require exact history/input
preservation, exact state gathering/scattering, finite recurrence outputs, and
unchanged unused pages and padding. Full recurrence fidelity is not asserted.
Require recurrent state to remain unchanged across convolution and gathering,
and convolution history to remain unchanged across the recurrent path.

This is a model-free diagnostic, not a context-capacity or production-performance
result. Preserve raw tensors, stage records, identity and memory observations.
Run only after the serving model is safely stopped and110GiB available; use the
existing lock/wrapper,40GiB floor,32/34GiB cgroup and bounded wall timeout.
Prepared Triton caches are copied without changing their source artifacts.
Any new JIT compilation makes this preparation, not frozen confirmation.
