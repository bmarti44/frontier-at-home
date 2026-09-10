# Bounded scheduler candidate after durability003

The preserved durability003 run returned 68 correct replies but failed both
five-admissions-per-worker window minima. Its host swap increase remains a
separate failure. The source-bound diagnosis is in `diagnosis.tar.gz`.

Candidate 1 changes only the experimental full-context profile's native scheduler
arguments: batch 256 and long-prefill threshold 64 replace 128/32. Keep four 262,144-token slots, the 1,048,576 aggregate cap, weights, tokenizer, cache formats/budget,
media policy, runtime, allocator/file-cache startup flags, and all containment,
identity, memory, swap and default/rollback requirements. Agent-fast and production
profiles stay unchanged. No numerical equivalence or speed improvement is assumed.
If this candidate is falsified, the second bounded configuration is 512/128;
declare and freeze it separately before use. No prefix-cache or quantization change.

The retained runtime window failure is the genuine RED. Also preserve the old
configuration validator rejecting the planned 256/64 configuration before adding
the candidate's evidence-only validator. Keep the old context/durability sources
and token/retrieval/window scorers unchanged. A separately frozen adapter selects
only a strict 256/64 launch validator and binds its own source/protocol into the
existing input manifest. It must reject wrong caps, slots, scheduling values,
duplicate flags, prefix-cache enablement and malformed argv. Changing this adapter
requires a new freeze and public seed; it cannot validate the old 128/32 result.

After a clean commit, freeze all candidate/unchanged sources, exact profile,
runtime/model/tokenizer inventories and prepared compiled inputs, then obtain
verified later public randomness and prepare the same four 4,224-token fixtures.
Start the actual named profile in fresh existing containment after stable 110 GiB
available memory. Verify exact argv/env, authentication, health and completed READY.
Run the existing startup retrieval smoke with the new configuration validator.

Before the expensive durability interval, run a bounded first-window falsifier:
four workers each submit up to five back-to-back requests using those fixtures,
with admission restricted to the first 300 seconds and the existing 600-second
per-request HTTP timeout. Each worker must actually start all five within 300s,
all 20 requests must finish normally with exact token counts and correct final
retrieval/negative controls, and all admitted work must drain by 900s. Start all
workers within 60s, retain every raw stream and timestamp, and require actual
four-way generated-token overlap. A recorded request error stops new submissions;
already admitted work drains or fails. Missing, duplicate, malformed, late or
unaccounted streams fail. Run the client in its own fresh zero-swap cgroup with
an external 960-second timeout. The existing engine watchdog continues sampling
identity and whole-host memory throughout. This is a necessary-condition
correctness/rate falsifier, not durability, context, fidelity or headline performance.

If the falsifier fails, stop safely, preserve FAIL/NO_RESULT and inspect the second
bounded configuration. If it passes, first verify no kernel OOM/Xid/new host swap,
no cgroup limit event, minimum whole-host memory >= 18 GiB, and unchanged prepared
compiled inputs. New compiled inputs make this a preparation result requiring a
separate freeze/replay; they cannot support confirmation. Bind the falsifier and
startup records before running the original 1800-second durability client with
the new launch validator. Its request-count, timing, overlap, native output,
memory/health and failure rules remain byte-for-byte unchanged. Additional startup
requests are declared here and prefix caching remains off.

Review candidate 2 closes the retained H1/M1 failures without changing scheduler
configuration or acceptance limits. Both public scoring and completed-run scoring
must check all 23 preliminary-window files against the admission receipt before
any read-only rescore. The exact input manifest, startup evidence and server launch
must match the full run, with window completion before receipt observation before
full admission. Missing or changed prerequisite bytes fail; scoring must not repair
the retained window summary. Preserve the unchanged durability component's result
alongside the combined client result. Catch transport exceptions inside each worker,
set the shared stop immediately, retain its failure row and drain admitted work.

Record whole-host swap totals and cumulative pswpin/pswpout counters before
startup, before timed admission and after shutdown; record cgroup current/peak/
events while it exists. Any increase in host used swap fails the no-new-swap
condition without unsupported attribution. Keep clean stop, no descendants,
>= 110 GiB recovery, closed model/runtime/cache verification and unchanged defaults.
This does not requalify direct context on a changed scheduler: after the candidate
passes these cheap falsifiers, direct aggregate 1M and required fidelity/switching
gates remain mandatory before production admission.

Native scheduler documentation corroborates the batching tradeoff but cannot
predict this host's result: [vLLM tuning](https://docs.vllm.ai/en/stable/configuration/optimization/).
The pinned source is authoritative for the actual 32-token clamp and kernel paths.
