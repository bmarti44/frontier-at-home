# Direct aggregate-context preparation

The independent tokenizer/template path exactly matched the running server's
18 input token IDs on a short correctness request. This is not a context result.
The real media endpoint also rejected five images and mixed image/video inputs.

Use four separately seeded prompts, exactly250,128 independently counted tokens
each (1,000,512 total), with the existing three-position retrieval plus negative
control. Native --long-prefill-token-threshold32 permits four simultaneous
32-token chunks within batch128. Keep four slots, the full cache, chunked prefill,
APC off, default reserve-full-input behavior, and all existing host safeguards.
The persistent gap reviewer verified both native scheduler clamp sites and the
absence of a Mamba-alignment blocker in this profile.

Require completed correct outputs from all four, actual returned prompt IDs
matching the independently prepared inputs, matching usage, genuine generated
tokens from every request before the earliest terminal event, and corroborating
four-running/zero-waiting native metrics in that shared interval. Require no
increase in preemption counters, no truncated/error stream, and the existing
external memory/swap/process/freeze checks. Metrics or open connections alone
are insufficient. No forced ignore-EOS/minimum output length is used.

The new launch check fails genuinely against running media012 because the
native long-prefill threshold remains0 there. Its raw assertion is preserved.
Only the next opt-in context attempt selects32. No production default changes.
