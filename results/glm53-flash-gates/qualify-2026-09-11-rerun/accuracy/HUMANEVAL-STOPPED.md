# HumanEval: stopped by the owner, not scored

GSM8K (`acc-gsm8k.json`, 94/100 holdout) and MMLU-Pro (`acc-mmlu-pro.json`,
202/247 holdout) completed in one kit run (73 min total) and are ledger-recorded.

HumanEval was attempted twice and has no result:

1. First attempt failed pre-flight: the Docker daemon (pinned
   `python:3.12-slim` sandbox) was stopped on the host (traceback in
   `log.txt`). The kit's recorded `status_reason` for the cell comes from this
   attempt.
2. After starting Docker, the resumed run (`suites_kept_from_previous_run`
   for the two finished suites) completed 5 of 164 problems, 4 correct, at
   13 min 42 s each: in the raw-completion protocol (no stop sequences,
   16,384-token budget) GLM never emits end-of-text, so every problem
   generates the full budget (~37 h for the suite). The owner stopped the run
   on 2026-09-11; the six partial transcripts are kept in
   `transcripts/humaneval/`. The README cell is TBD.

Nothing in this directory was edited by hand except this note.
