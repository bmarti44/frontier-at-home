# Track B gate 2 — DSpark equivalence + nvfp4-spec fidelity — 2026-08-20

## DSpark greedy-equivalence: FAIL (expected-behavior finding)
3/3 greedy prompts diverge between nvfp4 and nvfp4+DSpark (one from
char 2). Unlike llama.cpp draft-mtp (byte-exact), SGLang's DSpark
(VanillaMarkov head, gamma 7) is not output-preserving as implemented.
Outputs recorded (nospec-p*.txt vs spec-p*.txt).

## Accuracy suites on nvfp4-spec (ssm float32, low effort, 16384)
| suite | nvfp4-spec | llama.cpp Q4 baseline | delta |
|---|---|---|---|
| GSM8K holdout | 98.0 | 98.0 | 0 |
| MMLU-Pro holdout | 85.02 (2 invalid) | 85.02 | 0 |
| **HumanEval** | **74.39** | **79.27** | **-4.88** |

The -4.9 HumanEval drop (8/164 items) lands on the owner's primary
workload; unattributed between NVFP4 W4A4 and DSpark non-exactness
(an nvfp4-no-spec HumanEval run would isolate it — not yet spent).
Owner's conditional promotion ("if this all holds") is NOT satisfied.
Ledger namespace trackb-nvfp4.
