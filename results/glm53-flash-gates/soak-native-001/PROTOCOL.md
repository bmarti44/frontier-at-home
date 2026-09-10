# Native four-client durability gate

Prospective scope: sustained operation of the unchanged experimental full-context
profile for a 1,800-second request-admission interval, followed by bounded drain.
Four clients issue back-to-back 4,224-token retrieval requests. This is a short
prompt durability workload; context-direct-007 remains the separate aggregate
context capability evidence. No production speed, fidelity or switching claim.

Reuse the existing explicit retrieval instruction from context-clear-instruction-001,
fixture builder and final-answer scorer from script57, native stream validator
from script48, and memory/health thread structure from script35. Script35's
existing stream parser and verdict are unsuitable here: the parser does not
recognize native `delta.reasoning`, and its success gate does not require correct
final answers or preserve the native token streams. The compatibility witness
uses retained context007 bytes and makes no network or model request.

Freeze this protocol, client/preparation/scoring source, all reused sources,
profile, closed runtime/model inventories and prepared compiled inputs. Obtain
verified public randomness after freeze. Derive four fixture seeds from that
seed and prepare exact 4,224-token requests before loading the model. Use the
existing instruction, temperature zero, low reasoning effort, clear_thinking,
return_token_ids and 2,048 maximum output tokens. Bind fixtures and actual launch
before admission. A new seed changes identifiers only, not acceptance rules.

Run the exact named `glm-5.3-flash/cuda-spark-128g-1m-experimental` profile through
script93. Keep four configured 262,144-token slots, 92/94 GiB cgroup limits,
zero allowed cgroup swap, 18 GiB whole-host kill floor, inference lock, continuous
identity/watchdog sampling and its existing 9,000-second server timeout. Require
at least 110 GiB available before load. No serving or kernel changes.

Run one request with the first prepared fixture as a startup correctness
falsifier. Its raw bytes and verdict are separate from the timed interval. A
failed startup falsifier ends this attempt. Record prepared-cache equality after
startup, before timed admission and after shutdown; additions or mutations to
compiled inputs invalidate confirmation. Keep usage/log/lock metadata distinct.

Client acceptance is the conjunction of these fixed conditions:

- Admission lasts the full 1,800 seconds. No request starts at or after the
  admission deadline. All admitted requests finish by 2,400 seconds; each HTTP
  operation has a 600-second timeout, with an external client-unit wall timeout.
- Four workers start in the first 60 seconds. Each completes at least five
  requests admitted in the first 300 seconds and at least five admitted in the
  final 300 seconds. At least 30 requests complete overall. At least one common
  interval of generated tokens proves all four clients actually overlap. Between
  successive requests, each worker may spend at most five seconds outside HTTP
  processing; its final request must finish within five seconds of the admission
  deadline or later. This bounds client-side idle gaps in the back-to-back load.
- Every request has its own raw native stream, unique request identifier,
  exact prompt-token IDs, complete and equal output-token/usage counts, normal
  `stop` completion and a correct final answer including negative controls.
  Missing, malformed, duplicate, truncated, failed or unaccounted requests fail.
- Independent health probes cover admission and drain, all return 200, with
  at least 30 probes, first probe within five seconds, final probe within
  60 seconds of drain completion, and no probe-start gap over 60 seconds.
- Memory sampler stays healthy through drain, retains at least 0.8 samples per
  elapsed second, no sample gap over two seconds, endpoint coverage within two
  seconds and minimum available memory at least 18 GiB. External watchdog and
  cgroup observations remain mandatory for the combined host verdict.

Stop on a recorded client failure and preserve it; never replace failed rows.
All requests already admitted must drain or fail within the timeout. Store raw
streams, health/memory samples, timestamps and failures incrementally. Validate
the unchanged fixture/source bindings before and after the run. Scoring must
reject nonfinite numbers, inconsistent time/order/coverage and extra raw files.

Combined acceptance also requires the already established native profile
lifecycle checks: exact source/configuration/containment and owned process
identity, authentication and completed READY, no Xid/OOM/new swap, clean stop,
zero surviving descendants, memory recovery above 110 GiB, unchanged default/
proxy/guard state, and post-run runtime/model/cache verification. Both persistent
reviewers review only this new evidence gate; frozen serving components stay
closed. Retain PASS, FAIL or NO_RESULT with all raw evidence. No decode ratio or
headline throughput is inferred from these short final answers.
