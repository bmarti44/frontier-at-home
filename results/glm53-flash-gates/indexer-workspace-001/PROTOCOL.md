# Native indexer workspace falsifier

This is a new evidence-only candidate. It changes no serving profile, installed
runtime, numerical kernel, closed45probe or closed fixture/scorer file. It tests
the existing `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` setting at512 and64. Preserve
actual direct010 floorFAIL; this candidate does not attribute its memory drop.

The two fixed geometries are four requests of128 projected query rows and one
request of512 rows, all ending at token position262143. Each process configures
four262144 slots and512 scheduler tokens. These are synthetic cache addresses,
not actual long-context input tokens. A private import of the existing fixture
changes only its two prefill length declarations; the native compression,
valid-logit and selection oracles are reused unchanged. Request/block order is
selected from verified post-freeze public randomness. Inputs are identical across
arms, including all cache/tail bytes, query/key, weights, gate, APE and hidden data.
Actual GPU input digests are checked against independently regenerated CPU bytes.

Run two fresh512MiB baseline processes for the first seeded geometry and compare
them immediately, then the same for the other geometry. Any valid-logit,
cache/tail or **ordered final token-index byte difference** fails and stops before
either64MiB arm. Do not sort indices, drop an unstable pool, weaken the comparison
or call this noise: native top-k uses unsorted ordering and GLM consumes only511
of512 selected pools. A nondeterministic baseline falsifies this byte-identical
route for these inputs. After both controls agree, run fresh64MiB processes on
exactly the same two inputs and compare each to its baseline. No alternate arm,
case, repetition, seed or tie rule may be selected from outcomes.

For each arm require all fixed native metadata, chunk layouts and valid row bounds,
every valid FP32 logit byte, exact compressed cache/tail bytes (including untouched
regions), valid complete pool membership and exact tail padding. Capture all final
ordered indices. Canonical logit streams concatenate valid columns in original
global query order; masked/uninitialized columns are never evidence. Missing,
duplicate, extra, nonfinite, shortened or altered evidence fails. Repeated-baseline
and candidate comparisons must exhaust every artifact byte. The new scorer does
not change any existing scorer's acceptance.

The512 limit gives one512MiB logits tensor for four128-row requests;64 gives four
32MiB tensors, with up to64MiB adjacent-result overlap from Python assignment.
For one512-row request, the baseline result is128MiB;64 produces two64MiB tensors
and may still overlap128MiB. Record actual current/reserved/peak CUDA counters and
continuous external memory; logical arithmetic is not observed host saving. The
1,384,120,320-byte gather arena and1MiB radix workspace are unchanged. Do not claim
full-model fit or speed from this probe. The paired four-request candidate peak
must be at least256MiB below both baseline peaks to support the intended workspace
reduction; the one-request memory delta is descriptive only. A failure is preserved.

Every arm uses a fresh process and fresh hardened cgroup, held under the existing
inference lock through cleanup, scoring and post-verification. Required unchanged
safety:110GiB stable start,40GiB host floor,600seconds per arm,MemoryHigh32GiB,
MemoryMax34GiB,MemorySwapMax0,OOMPolicy=kill,KillMode=control-group. The wrapper's
32GiB minimum is conservative containment, not measured allocation or reservation.
The actual largest logits512MiB, pinned readback512MiB, gather1.29GiB and roughly
45MiB cache/tail plus small projected inputs fit well below this cap; native JIT
workspace remains measured under the cap rather than assumed absent. Require
broad pre-freeze through post-verification swap counters unchanged, continuous
identity and floor checks, no OOM/Xid, zero exit and no surviving descendants.
No model, other GPU probe, CUDA build, publication or large verification may run
concurrently. The reviewer performs only CPU controls before submission.

All probe-owned H2D sources live in retained pinned buffers until synchronization;
keys are reinterpreted as BF16 before copy and APE stays FP32. Readback uses a
persistent pinned512MiB arena with completion fence before reuse. Metadata uses
the native device-side builder and retained pinned CPU bounds. This does not
claim pinning for every inherited compiler/native launch allocation. Capture
wrappers remain local to this evidence process and are restored on failure.
Their synchronization and gzip I/O prohibit using these CUDA intervals as speed
measurements.

Before any hardware work, commit the proposed test/protocol and retain its actual
missing-implementation RED, implement and commit a clean independently reviewed
candidate, and run the full relevant CPU audit. `freeze.py` must run unloaded. It
copies exact source/dependencies and metadata only, verifies the current profile's
closed runtime inventory, snapshots the prepared cache tree, and binds interpreter,
compiler,libc,node,BLS verifier,wrapper,fixture,scorer and configuration. Fetch the
exact next eligible public beacon using the frozen fetcher only after manifest
completion. `run.py` re-verifies the BLS receipt and bindings, retains invocation
arguments/environment and uses the existing capture/host/guard helpers unchanged.
Use the frozen run.py copy, never a direct native invocation.

This first execution is **preparation-only**: new shapes may compile and cache
metadata may retain existing absolute paths. Preserve each arm's complete generated
cache inventory after cleanup. Inner synthetic checks may PASS, but combined
success is NO_RESULT for kernel-binary qualification. A baseline mismatch or any
host/scorer failure is FAIL. No automatic large-model admission follows. A later
prewarm/freeze/sealed replay with a new public seed is required for any authoritative
binary-equivalence claim; it is deliberately not implemented or claimed here.
