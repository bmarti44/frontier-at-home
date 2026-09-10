# Prepared-kernel replay of the unchanged 512/128 profile

Attempt `soak-native-005` passed the 20-request necessary window, with five
admissions per worker inside 300 seconds. Overall qualification remains FAIL:
one host swap-in page occurred before the window, and seven generated kernel
files changed the frozen input set. Full durability was not admitted.

This candidate addresses only those observed execution findings. Append the
completed `server-20260910-054432` state to the existing prepared-cache sources.
Before implementation, require a genuine RED on unchanged launcher code: all
seven committed generated files must exist and match their recorded hashes in
the source and in the launcher-created copy. Existing sources and scorers remain
unchanged. Preserve all earlier attempts and do not promote or change defaults.

Keep the exact 512-token batch, 128-token long-prefill threshold, four 262,144-token
slots, aggregate 1,048,576 cap, weights, precision, startup flags and containment.
No serving token/layer/expert implementation changes or new diagnostic switches.
The external swap observer is explicitly launched only for this evidence run;
it is absent from serving profiles. Record phase times and host counters before
freeze, after freeze, after public randomness and fixture preparation, immediately
before profile start, after READY, before and after the necessary window. Retain
the broad baseline as well as all phase observations; do not move or weaken the
existing no-unattributed-swap-I/O gate. Major-fault correlation is not attribution.

Freeze clean source, runtime, model, tokenizer, generated caches, scorer and
observer before obtaining a later verified public seed and preparing fixtures.
Run the unchanged short correctness check and 300-second necessary window using
the closed `soak-scheduler-002` adapter. Only all existing window, frozen-cache and
host checks PASS can admit the unchanged 1,800-second full-duration gate. New
cache files require another separate freeze/replay; failed or null results stay
preserved. Direct aggregate 1M, native paired fidelity, switching and headline
performance are separate gates and inherit no result from this short check.
