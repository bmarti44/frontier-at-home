# Transport controller correction

The first new controller has two verified high issues: cleanup errors can
release the inference lock early, and parsed manifest bytes can differ from the
recorded digest. A medium issue reports FAIL with exit0. Exact CPU witnesses and
source are archived in controller-candidate1-review.tar.gz before implementation.

The bounded correction reuses the already reviewed one-layer controller's
stable-read bindings and unchanged closed capture/identity/host-scorer path.
Do not maintain a second cleanup implementation. Acceptance requires those
closed helper sources and bound_read/check_bindings functions to remain exact,
the existing malformed-launch tests to pass, nonzero failure exit, and real
transport evidence to pass the unchanged host scorer before any timing claim.

The reused hardened wrapper requires at least32GiB MemoryHigh, giving a34GiB
ceiling. Declare this accurately; it does not reserve34GiB. The diagnostic
allocates a256MiB buffer, keeps110GiB start admission and adds a64GiB kill floor
and180-second timeout.34+64GiB must fit measured available memory. This changes
containment from the proposed512/768MiB envelope; it does not claim equivalence.
No model/GPU, serving, native reference, or full-file throughput qualification.
