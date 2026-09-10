# Scheduler 512/128 preparation: window PASS, overall FAIL

The optional aggregate-1M profile ran through `scripts/93_profile_serve.sh` with
four 262,144-token slots. These 4,224-token requests are a bounded admission/correctness
falsifier; they do not requalify context, fidelity or production performance.

All 20 native streams independently rescore correct, including final retrieval
and negative controls. Each worker admitted five requests before 300 seconds;
four-way generated-token overlap passed and drain finished at 303.226897276 seconds.
The first-window client verdict is PASS. No full 1,800-second client was admitted.

Overall prerequisites did not pass. Seven generated kernel-cache files appeared,
with no changed/missing entries among 2,477 frozen files; two other additions are
allowed metadata. Frozen-kernel confirmation remains NO_RESULT pending freeze/replay.
Host `pswpin` also increased by one page between the prelaunch baseline and the
before-window observation. Used swap and pswpout stayed unchanged, as did the
swap counters through the window and stop. Attribution is unknown; the fixed
no-swap-I/O check remains FAIL. This does not establish a model-caused swap event.

External memory stayed above 18 GiB (minimum 18.259899139404297 GiB).
All GLM cgroup events and swap were zero, the kernel journal was clear, and the
identity guard confirmed a clean stop with no surviving descendants. Memory
recovered above 115 GiB, model/runtime verification passed, and default/proxy/guards
remained unchanged. Keep 512/128 for a separately frozen replay; no scheduler
improvement or numerical-equivalence claim is inferred from this preparation.

`manifest.json` binds both archive parts, every member, raw observations and the
summary. Concatenate numbered parts to read the gzip tar. Credentials are excluded.
Publication/terminal review receipts are retained separately. Earlier attempts
remain unchanged; full model qualification and qualified speed remain unmeasured.
