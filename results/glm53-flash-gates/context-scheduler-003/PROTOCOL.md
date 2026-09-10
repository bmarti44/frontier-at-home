# Direct context on the current 512/128 scheduler

The current profile uses 512-token batches and a 128-token long-prefill threshold.
The old direct-context invocation still requires 128/32, preventing its use with
the current declared profile. Reuse scheduler002's exact configuration validator
for direct preparation, actual-launch binding, short correctness, run and score.
Do not change serving code, the profile, the request builder or acceptance math.

Keep the reviewed explicit instruction, four inputs of exactly 250,128 tokens,
all retrieval positions and negative controls, prompt/output IDs, timestamps,
concurrent-generation and no-preemption checks. Short correctness remains only
a startup falsifier. Native lifecycle, memory, identity, OOM/Xid, swap, compiled
input and terminal gates remain separate and required. No fidelity, production
speed, switching or default promotion follows from an API-only PASS.

Before any preparation or run, freeze the adapter, all reused source bodies and
the exact current profile/model/runtime/tokenizer/cache inputs from clean source.
Every adapter action must verify its declared source bindings against that freeze.
Then obtain a later verified public seed and prepare while unloaded. Bind actual
launch arguments exactly to the planned fixture configuration without changing
request bytes. Preserve the earlier context007 result on its earlier settings.
