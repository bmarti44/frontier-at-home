# Direct context with an explicit output instruction

Context-direct-006 processed four250128-token inputs, but the unchanged scorer
returned FAIL: slots0/2/3 exhausted2048 output tokens in reasoning and had empty
final content. Slot1 returned the required answer. Raw failed traces repeatedly
debate whether numeric order means line labels or identifier values, and whether
the final control field requires a comma. Preserve that failure unchanged.

The next bounded alternative changes only the fixture output instruction:

> Return exactly one line containing four comma-separated fields. The first
> three fields must be the exact identifier values after the colons on AUDIT
> RECORD lines labeled1,2,3, in that label order. The fourth field must be the
> literal NO_EXTRA_RECORD. Do not include labels, explanations, or other text.
> Do not invent any other record.

Use spaces between labels in the executable prompt (labeled 1, 2, 3). Do not
supply any actual record value in this instruction. Use the existing57 fixture
builder, record positions, absent-value negative control and48 scorer unchanged.
Render/tokenize the modified prompt and adjust only filler length until every
request is exactly250128 actual tokens. Retain2048 output tokens, temperature,
seed derivation, template and parser settings. No enable_thinking=False change:
the pinned template ignores it and the parser could relabel reasoning as content.

Before loading, commit and freeze this preparation source, unchanged scoring
sources, fixtures/tokenizer, model/runtime inventories, exact profile and all
prepared compiled inputs. Obtain independently verified post-freeze public
randomness, then prepare all inputs while the model is unloaded. Bind the actual
launch with the already reviewed binder. This is a new candidate, not a rescore
of006. No source, scorer, fixture or generated-cache mutation may be hidden.

Use the same named experimental1M aggregate/four-slot profile, native execution,
92/94GiB containment, swap0,18GiB whole-host kill floor, external sampling and
9000s wall timeout. Preserve authentication/default/guard state and enforce110GiB
available before loading. Native start/stop remains the already qualified
experimental lifecycle; record it without reopening unchanged implementation.

A short request may cheaply reject the new instruction before the direct run.
Use the same prepared instruction and deterministic fixture at a short length,
no more than2048 output tokens, and the unchanged final-answer scorer. Label this
as a correctness smoke, never context capability. Preserve it in startup history;
a failure stops this candidate before the expensive run. A passing short request
does not weaken any full-context gate.

The direct acceptance is unchanged: all four final replies must finish with stop
and match all three records plus the negative control; actual prompt/output IDs
and usage must agree; inputs total1000512; generated streams overlap; independent
metrics corroborate four running without preemption; full identity/memory/kernel
coverage and clean shutdown; unchanged prepared runtime/cache bytes. A failed
answer, short output, timeout, Xid, OOM, swap or invalid evidence is FAIL. Host
minimum must remain above the18GiB kill floor (and repository10GiB acceptance
minimum). No fidelity, performance, production switching or default promotion is
claimed. Keep the earlier failures and use the existing persistent reviewers.
