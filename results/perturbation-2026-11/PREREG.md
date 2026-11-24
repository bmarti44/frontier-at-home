# Perturbation experiment: does qwen38fn lose more than Inkling 2 Mini on lightly changed questions?

DIAGNOSTIC. This is not an acceptance gate and not an input to the promotion rule. It answers an
owner question (2026-11-23): if the evals are changed slightly and re-run, does the serving default
`qwen38fn` drop while `inkling2mini` holds? A drop that only one model shows on the same changed items
is evidence that the model has memorized those benchmark items. Everything below was fixed before any
model answered a changed item.

## Items
The rows both models were already scored on, by the accuracy bench's own selection
(`select_indices`):

| Suite | Rows | Excluded | Kept | Change | Variants per row |
|---|---|---|---|---|---|
| GSM8K | dev 100 + holdout 100 | 4 | 196 | new numbers and names, answer recomputed | 2 |
| MMLU-Pro | dev 253 + holdout 247 | 9 | 491 | options shuffled, the correct letter moves | 1 |
| HumanEval | 164 | 0 | 164 | function renamed to a same-informativeness synonym | 1 |

An excluded row leaves both arms.
- **GSM8K (4):**
  - row 672: every number is written as a word, so nothing can be varied;
  - row 749: the reference answer is wrong (the text's "six groups" vs the reference's "groups of six");
  - row 454: the reference answer is wrong (two people eat 4 apples a day each; the reference uses 4 + 1);
  - row 926: two readings agree only at the original values, so a variant can be scored against one reading only (review A-M1).
- **HumanEval:** a row whose renamed canonical solution fails its renamed test in the sandbox leaves both arms (review A-L4).
- **MMLU-Pro (9):** an option refers to the other options by position ("none of the above", "all of these"), so a shuffle changes its meaning.

## How each change was made and checked
**GSM8K: templates in the GSM-Symbolic style.**
- Each row has a template: the question with its free numbers and person names as variables, plus value ranges, constraints and an exact-arithmetic solver.
- Every template passed these mechanical checks (`gsm8k_template_lib.check_entry`):
  - at the original values it renders the original question character for character;
  - its solver returns the official answer;
  - each range stays within 4× of the original value.
- A second solver for every row was written blind, by a different agent that saw only the template (never the first solver or the reference solution). The two solvers must agree on the original values and on 300 random acceptable samples per row: 0 disagreements.
- Each variant is drawn from a per-row RNG keyed by the public seed, and must satisfy all of these:
  - the constraints hold;
  - at least one number differs from the original;
  - the answer differs from the original answer, so a memorized answer is wrong;
  - the answer is whole where the original was whole;
  - the answer is positive where the original was positive;
  - magnitude matching (review A-H1): every integer keeps its original number of digits, and the answer stays within 0.5–2× the original.
- One variant has a smaller answer than the original and one a larger answer, where the template allows both. Variants are therefore not systematically harder.
  - With a placeholder seed the variant/original answer ratio had median 1.03 and quartiles 0.75 / 1.50.
  - The digit count of the answer grew in 33 variants and shrank in 23.
  - 11 rows have both variants on one side.
- The ranges the template authors wrote allowed up to 4× upward. Before matching, the median ratio was 1.89, which the review flagged.
- Numbers written as words, unit and calendar facts, and structural counts stay fixed.

**MMLU-Pro.**
- Each item's options are permuted by a per-item RNG keyed by the public seed, redrawing until the correct option changes letter.
- The question text and option texts are unchanged.

**HumanEval.**
- The entry-point function is renamed to a synonym of similar informativeness: an opaque original gets an opaque name (`skjkasdkd` → `qpwlrmzx`, `f` → `g`). The table is `humaneval-names.json`.
- Names that added task information were replaced with neutral ones (review A-L2).
- These occurrences are renamed:
  - call-like uses (`name(`) in the prompt, the canonical solution and the test;
  - `function <name>`;
  - prose mentions of identifier-like names.
- English words used as prose ("return the longest one") stay.
- Validation:
  - the renamed prompt plus the renamed canonical solution passes the renamed test in the bench's sandbox, for all 164;
  - the new name is not already used in the row;
  - the old name no longer appears as a code identifier.

## Protocol
**Rendering, budgets and scoring.**
- Every job is rendered, sent and scored by `scripts/31_bench_accuracy.py`'s own functions (`render_item`, `Client.complete`, `score_gsm8k`, `score_mmlu`, `run_humaneval`). Settings match each model's holdout draw:
  - official encoder, thinking mode, reasoning effort `low`;
  - HumanEval raw;
  - 16,384 tokens;
  - no HumanEval stops;
  - temperature 0, seed 42.
- The runner is `run_perturb.py`. HumanEval completions are executed afterwards, one sandbox at a time, each start waiting for MemAvailable ≥ 10 GiB.

**Arms and order.**
- Each model answers both arms in one session: the original (O) and the changed (P).
- The jobs are one seeded shuffle of both arms (`make_items.py`), so server drift hits both arms alike.
- Concurrency is 3 for both models, which leaves one of the 4 slots free on production.

**Servers.**
- `inkling2mini` is served through the switch (production path, profile `inkling-2-mini/cuda-spark-128g-1m`) during the M4 switch round trip.
- `qwen38fn` is the serving production instance after the switch back.
- The runner refuses a server whose model id is not the expected one.

**Failures.**
- A transport error (`invalid:`) other than a timeout is retried once (`--retry-invalid`).
- A timeout is final and scored wrong, as in the holdout contract; the request clock is the holdout's 2,700 s.
- A HumanEval sandbox infrastructure failure (launch failure, exit 125–127) is re-executed once (`score-humaneval`).
- What remains counts as an infrastructure failure.
- A scoring exception is a bug and stops the run; it is never recorded as a transport error (review A-M3).

## Analysis (`analyze.py`, fixed now)
**Per model and suite**, over the items whose arms both completed:
- O_i = 1 if the original was answered correctly; P_i = the mean over the item's variants.
- **drop** = mean(O_i − P_i), in percentage points, with a 95% paired bootstrap interval over items (10,000 resamples, seed 20261123).

**Between models**, on the items both completed:
- **difference** = drop(qwen38fn) − drop(inkling2mini), bootstrapping items jointly.

**Readings:**
- A model **drops** when its drop interval lies above 0.
- **qwen38fn drops more** when the difference interval lies above 0; **inkling2mini drops more** when it lies below 0.
- Otherwise there is **no detectable difference**, and the interval width is the sensitivity.

**Suite priority.** GSM8K is the primary suite, because a changed number makes a memorized answer wrong. MMLU-Pro and HumanEval are secondary; their changes break memorized letters and names but not memorized content. With three suites, a single secondary interval that barely excludes 0 is reported as weak evidence.

**NO_RESULT.** A suite is NO_RESULT for a model if any job is missing, or more than 2% of its jobs end in an infrastructure failure. Below that, an item with any infrastructure-failed job leaves both arms and the between-model set, and is listed (review A-M2). A timeout is not an infrastructure failure: it stays in the data, scored wrong.

**Also reported, not read:**
- each model's O-arm accuracy next to its official holdout and dev scores (run-to-run variation under concurrency);
- median completion tokens per arm (a model that recognizes an item may answer it more briefly);
- MMLU-Pro accuracy by the correct option's letter in each arm. A model with a letter preference can change accuracy under a shuffle for that reason alone (review A-L3).

## What this cannot show
- A drop on changed items in only one model is evidence of memorized benchmark items. It can also come from brittleness to surface form, which this design cannot fully separate from memorization.
- No difference does not mean "not bench-maxed" in the broader sense of heavy training on benchmark-style data. That needs problems written after both models' training cutoffs.
- The GSM8K variants are magnitude-matched and balanced around the original answer, but a new number can still be less "round" than the original. Any residual difficulty shift is common to both models' items. It cancels in the difference only to the extent the two models are equally sensitive to it, so a small difference is read with that caveat.

## Seed
The variant draws and the job order are keyed by a public drand randomness value. It is fetched after the templates, solvers, builders, runner and this file were frozen, after the HumanEval rename check, and before any model run. Its round, randomness and signature, read from two independent relays that must agree, are in `seed.json`; the seed string is `drand-<round>-<randomness>`.

## Review
Reviewer A (measurement) reviewed the design once before the seed, as a diagnostic:
- 1 high: GSM8K magnitude inflation;
- 3 medium: an ambiguous row, infrastructure items, scoring errors;
- 4 low: the timeout, informative names, letter position, the rename-check rule.

All are closed as described above.

## Rebuild after the power loss (2026-11-24)
A site power loss on 2026-11-23 erased this preparation, which had lived in a scratch directory. Nothing had been run against a model.
- **How it was rebuilt.** The exact file writes and commands recorded in the session transcripts (the coordinator's and the 20 template and solver agents') were replayed in time order into this directory.
  - Every replayed command printed what it printed the first time. The only differences were the thread order of a mock dry run, one truncated print, and timestamps.
  - With the placeholder seed, the builders reproduce the figures above: 196 GSM8K rows, answer ratio median 1.033 with quartiles 0.75 / 1.50, 33 answers with more digits and 23 with fewer, 11 one-sided rows; MMLU-Pro 491 kept; HumanEval 164/164 renamed cleanly.
  - All ten GSM8K batches show 0 solver disagreements on 300 samples per row.
- **Added after the review, plumbing only.** None of these changes a rule above:
  - `check_humaneval_renames.py` is the sandbox check the HumanEval section already requires. It runs each original canonical solution as a control, then the renamed one, one container at a time with MemAvailable ≥ 10 GiB. Its output is `rename-check.json`.
  - `build_all.sh` runs the three builders and `make_items.py` with the seed, drops any task the rename check excludes, and hashes the outputs into `BUILT.sha256`.
  - `run-arm.sh` runs one model: two `generate` passes (the second retries non-timeout transport errors once) and two `score-humaneval` passes (the second re-executes sandbox infrastructure failures once). It records 1 Hz memory samples and the kernel log, and refuses a tree that differs from `FROZEN.sha256` or `BUILT.sha256`.
- **Endpoint.** Both models are reached at the production engine port, `http://127.0.0.1:8013`, with the bearer token from `/etc/deepseek-v4-flash/api-key`. `inkling2mini` runs first, while it is switched in; `qwen38fn` runs after the switch back.
