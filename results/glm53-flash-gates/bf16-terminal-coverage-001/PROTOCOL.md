# Native terminal coverage correction

Attempt004 remains FAIL. The frozen guard completed the native calculation and
installed its terminal exec filter, then took3.579708 seconds to observe normal
interpreter exit and cleanup. Its existing exit allowance is five seconds. The
host scorer applied the two-second active sampling cadence across that teardown
and also compared the final memory sample with the terminal identity rather than
cleanup. Active identity gaps stayed below0.363s and memory gaps below0.463s.

Keep every memory interval and every active identity interval at most two seconds.
Require the verified terminal-to-cleanup interval positive and at most five seconds
on both clocks, with the existing clock-agreement check across all rows. Require
memory coverage through cleanup, the same terminal filter, unchanged exit0 and
empty-group checks, wrapper chronology, containment, swap and memory limits. Report
active identity cadence separately from teardown duration. Do not widen the active
sampling threshold or change the guard, native calculation or runtime.

The synthetic regression reproduces the unchanged scorer failure using a3.5-second
verified teardown with continuous memory observations. Reject missing teardown
memory, active identity gaps, >5-second teardown, invalid terminal filtering and
clock disagreement. Existing host scorer mutations remain required. This is an
evidence-only scorer correction with no import into serving code.

Preserve the original004 result. A corrected candidate must use a new clean freeze,
post-freeze public seed and native replay with unchanged computation and safety.
No native replay may run alongside the full GLM profile. These preparation tests
cannot establish full-model BF16 reference or paired fidelity.
