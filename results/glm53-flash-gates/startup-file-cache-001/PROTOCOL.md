# Verified model-file cache advice before CUDA initialization

Second bounded startup hypothesis after soak-startup-001 and002. Attempt002
failed during native CUDA context creation before model loading; the first
pre-profile allocator-cleanup candidate was never exercised. Its efficacy
remains NO_RESULT. Timed durability has not started in either attempt.

Use only POSIX_FADV_DONTNEED on the already verified model's listed regular files,
after final artifact hashing and before importing vLLM/CUDA. Place the advice in
the existing guarded serve() entry, under the inference lock and containment.
The exact GLM53_RELEASE_MODEL_FILE_CACHE=1 flag is resolved from the logged launch
configuration once at startup and defaults off. Enable only the experimental1M
profile. Keep the first startup flag selected; no context, tensor, arithmetic,
KV, kernel, scorer, production/default or safety-setting changes.

Check the inventory hash against launch.json before parsing paths. Reject nested
or escaping paths, symlinks, non-regular files and size mismatches. Open read-only
with O_NOFOLLOW and close every descriptor, including failures. Advice failure
aborts before CUDA. Do not hash/read weight payloads after advice before importing
the engine. Log complete before/after host memory observations and advised file
count/bytes through existing startup stdout; this is startup-only evidence.

CPU acceptance exercises the real serve() method: the enabled arm cannot reach
the mocked engine until advice completes, the disabled arm retains its previous
path, malformed inventory/path/file/advice failures abort before engine import,
and no weight payload is read by advice. Preserve genuine RED before the change.

Actual acceptance remains the unchanged native-profile startup and durability
protocol: fresh clean freeze and later verified public seed; exact logged flags;
>=110GiB before load,18GiB external floor, zero cgroup swap; unchanged compiled
inputs and pinned model/runtime; no kernel OOM/Xid; authenticated READY and correct
4224-token startup retrieval. Only then admit the unchanged30-minute four-client
soak. Preserve exact shutdown/identity/default/host evidence. If this second
bounded hypothesis fails, retain NO_RESULT for this branch and report the exact
failure instead of adding allocator knobs or rebooting.

Page-cache reclamation is an experiment, not a proven cause. NVIDIA documents a
system-wide buffer-cache debugging workaround, but this candidate uses narrower
read-only file advice already used by this repository's scripts69 and95:
https://docs.nvidia.com/dgx/dgx-spark/known-issues.html
