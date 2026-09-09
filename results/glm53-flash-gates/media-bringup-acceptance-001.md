# Bounded media bring-up from the working text configuration

Keep attempt 008's 128-token prefill chunks, exact 9,565,304,320-byte cache budget,
four 262,144-token slots, standard allocator, one-time startup cleanup, prepared
kernels, disabled autotuning, 18GiB floor and containment. Enable the native
vision tower and select `--mm-processor-cache-gb 0` to avoid optional retention
of decoded/preprocessed media in API/core processes. Preserve resolution and
frame limits. Reuse actual008 kernels, including TileLang's omitted cache.

Acceptance: existing API key still authenticates; unauthenticated access rejects;
text/tools still work; single-image, four-image and 16-frame video requests give
correct simple visual answers and complete. All external memory samples stay
above 18GiB, cgroup swap is zero, and no CUDA error or unintended process survives.
These short visual checks do not prove maximum context, fidelity or throughput.

007 used 512-token chunks and vision while 008 changed both factors. It does not
prove a vision-only 2GiB requirement. Actual visual tensors total 1,127,254,016
bytes. The running 008 snapshot has 20.1008GiB measured minimum availability.

Persistent gap reviewer recommended this bounded comparison. Development
startup numbering is distinct from formal per-gate candidates; latest formal
indexer sign-off remains candidate 4 / campaign 49. No frozen component is reopened.
On failure, restore the working text setup with the same credential.
