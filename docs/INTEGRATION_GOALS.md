# Integration goal contract

Every claimed model/backend integration gets its own durable autonomous goal.
Create it in the draft claim PR before changing an engine or running a large
model. The goal turns a long investigation into resumable, auditable gates; it
does not imply that the integration will pass.

Use these paths, replacing the placeholders with the catalog slugs:

```text
scripts/<model-slug>_<backend>_goal.py
results/<model-slug>-<backend>-goal/state.json
results/<model-slug>-<backend>-goal/<gate>/attempt-<number>/
```

The controller must expose one stable interface:

```bash
scripts/<model-slug>_<backend>_goal.py run
scripts/<model-slug>_<backend>_goal.py resume
scripts/<model-slug>_<backend>_goal.py status --json
```

`run` initializes or continues the goal, `resume` continues an existing goal,
and `status --json` is read-only and machine-readable. Both execution commands
select the highest-value unfinished gate and continue without asking a human to
sequence routine work.

## Initial goal definition

The first goal commit must record:

- the exact model, public weight source, license, tokenizer, backend, hardware,
  and maximum target context;
- a measurable success condition for correctness/fidelity, context capability,
  prefill, decode, TTFT, memory headroom, service health, switching, and
  rollback;
- a clean baseline and the reference used for comparisons;
- model/backend-specific OOM containment and an explicit stop condition;
- the ordered gates and fixed scorer assigned to each gate;
- the evidence root and the persistent reviewer ledger.

At minimum, define gates for foundation, fidelity/accuracy, safe context,
performance, operator switching/rollback, and final review. Add architecture-
specific gates only when they test a real risk. Do not copy GLM thresholds,
CUDA memory limits, or a 1M context target to hardware or models where they do
not apply.

Each gate is exactly one of `PENDING`, `RED_CONFIRMED`, `PASS`, `FAIL`, or
`NO_RESULT`. Narrative text cannot produce `PASS`; only the committed scorer
can. Preserve every attempt, including failures and null results, with
`manifest.json`, `raw.jsonl`, and `summary.json`.

## Required workflow

1. Commit the controller, initial state schema, gate tests, and acceptance
   formulas before production implementation.
2. Capture a genuine RED result on unchanged production code.
3. Implement the smallest default-off diagnostic change and clean-build it.
4. Freeze source, binary, scorer, model, tokenizer/fixture, and configuration
   hashes before selecting fresh confirmation inputs.
5. Run in architecture-appropriate containment, score only preserved raw
   evidence, and mutation-test the scorer.
6. Reuse the same reviewers for the life of the goal. Continue until neither
   reports a verified high or critical issue; reviewers choose their own score.
7. Finish with reproducible terminal results for every gate, either a qualified
   integration or an evidence-backed `NO_GO`/`NO_RESULT` where appropriate.

GLM-5.2 on CUDA is the current concrete example in
[`scripts/glm52_goal.py`](../scripts/glm52_goal.py). Reuse its fail-closed
evidence ideas, not its model-specific implementation sequence.
