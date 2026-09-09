# Native prompt metric adapter review

Candidate `0a903133`, gate candidate1, campaign-global round14. Both persistent
reviewers independently report zero high/critical findings in this reducer.
Six focused CPU regressions passed. The gap reviewer also constructed and
serialized rows using the pinned vLLM native prompt-logprob builder and verified
the reduction.

The adversarial reviewer found a medium native tie compatibility issue. Its
actual native builder reproduction has an equal-logprob truth with rank2 and
selected top1 with rank1. The genuine RED is committed at `12e5a439` in
prompt-tie-red-001. `dd75c6be` rejects only strict logprob order inversion;
the selected rank1 token determines correctness. All seven focused and121
combined CPU regressions pass. The adversarial reviewer independently repeated
the pinned native tie reproduction and attested closure without a new full
review cycle. Ambiguous multiple rank1 entries still fail closed.

The surrounding campaign MUST bind engine `logprobs_mode=raw_logprobs` explicitly.
Pinned vLLM can return raw logits in the same response field; negative logits
can pass numeric checks and cannot be distinguished by field naming. Bind the
actual engine configuration, source/runtime, tokenizer/vocabulary, exact request,
fixture and raw response before applying the fixed100-case paired formula.
This component is evidence-only and provides no standalone quality verdict.
No GPU, model or CUDA compilation was performed by this review.
