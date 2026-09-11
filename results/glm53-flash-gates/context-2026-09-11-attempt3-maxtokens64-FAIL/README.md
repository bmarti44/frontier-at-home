Four-slot 1,000,560-token fill under expandable_segments:True. Memory PASS
(minimum 14.92 GiB, floor 10). Retrieval FAIL for a probe reason, not a model
reason: `50_probe_context.py` defaulted to `max_tokens 64` and GLM spent the
whole budget in `reasoning_content` (finish_reason `length`, empty content) —
the reasoning text names the correct codes (slot 0 "lynx code 43620311", slot 2
"otter 56410561"). The probe default is now 512 tokens; see the next run.
