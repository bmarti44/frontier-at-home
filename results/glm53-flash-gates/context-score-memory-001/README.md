# Context scorer memory correction

The interrupted attempt and genuine RED are preserved in context-direct-002.
The scorer now streams unchanged metrics JSONL through EOF and retains only
scalar checks plus the current row. All prior acceptance checks remain intact.
The fixed test limits extra traced peak allocation for a 16 MB metrics log to
5 MiB; the original implementation added 30,189,420 bytes. All 17 tests pass,
and both persistent reviewers found no remaining high/critical issue in this
correction. This does not establish model capacity or fidelity.
