CPU indexer reference audit
===========================

Candidate 1, campaign review round 45. The complete scoped audit passed 240 tests. The expected Triton helper-load error in the raw output is an exercised negative control. The four initial tests and genuine missing-module RED were committed before implementation; a fifth test adds full 2,048-row prefill and valid-logit checks. Production code, defaults, previously frozen helpers and GPU runtime are unchanged. This is preparation for the actual post-projection GPU probe, not its execution result.
