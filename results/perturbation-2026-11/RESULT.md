# Perturbed evals: is qwen38fn bench-maxed? (DIAGNOSTIC)

**Answer: no sign of it.** On changed versions of the benchmark questions,
qwen38fn scored the same as on the originals in all three suites. So did
Inkling 2 Mini, and no suite shows a detectable difference between the two
models.

The sharpest test is GSM8K, the primary suite, because a memorized answer
becomes wrong once the numbers change:
- **qwen38fn:** 99.5% on the originals and 99.2% on 392 new-number variants.
  The drop is +0.3 pp, with a 95% interval of [0.0, +0.8]. That rules out a
  memorization drop larger than about 0.8 pp.
- **Its one variant miss:** a wording that allows simple or compound interest.
  qwen38fn chose compound.

This is a diagnostic, as the owner asked. It does not feed any promotion
decision. The rules were fixed in `PREREG.md` before the seed was drawn and
before any model run.

## Results

Here, O is the original item and P is the changed version. For GSM8K, P is
the mean of two variants. The drop is O − P in percentage points, with a 95%
paired bootstrap interval over items (10,000 resamples, seed 20261123). Each
model's change is applied to that model's own runs of the same items.

| Suite | Model | O | P | Drop (pp) [95% CI] | Reading |
|---|---|---|---|---|---|
| GSM8K (196 rows, 2 variants each) | qwen38fn | 99.5% | 99.2% | +0.3 [0.0, +0.8] | no detectable drop |
| | inkling2mini | 97.4% | 97.7% | −0.3 [−1.8, +1.0] | no detectable drop |
| | **difference** | | | +0.5 [−0.8, +2.0] | no detectable difference |
| MMLU-Pro (491 rows, options shuffled) | qwen38fn | 87.2% | 88.2% | −1.0 [−2.9, +0.8] | no detectable drop |
| | inkling2mini | 83.9% | 82.7% | +1.2 [−1.2, +3.7] | no detectable drop |
| | **difference** | | | −2.2 [−5.1, +0.6] | no detectable difference |
| HumanEval (164, entry point renamed) | qwen38fn | 84.8% | 87.8% | −3.0 [−9.1, +3.0] | no detectable drop |
| | inkling2mini | 68.9% | 72.6% | −3.7 [−10.4, +3.0] | no detectable drop |
| | **difference** | | | +0.6 [−7.9, +9.1] | no detectable difference |

The difference is drop(qwen38fn) − drop(inkling2mini) on the same items, with
items bootstrapped jointly. A positive difference would mean qwen38fn loses
more. Unrounded values are in `summary.json`, and the console output is in
`analyze.out`.

Both runs are complete:
- every job exists (1,898 per model);
- 0 transport errors and 0 sandbox infrastructure failures;
- no items were excluded, so no suite is `NO_RESULT`.

## Reading it

**GSM8K (primary).** qwen38fn missed one original and three variant answers
out of 588 jobs.
- **Row 187 is the whole drop.** Variant 0 asks what Priya owes after
  "monthly interest of 1%" on $100 for 3 months. qwen38fn reasoned that
  compound interest is "the more natural interpretation" and answered 103.03.
  The reference is 103, and Inkling answered 103. This is one ambiguous item,
  not a memorization pattern.
- **Row 368 looks like a reference problem.** Both models miss the original
  and both variants, and in all six answers both models are off by exactly 8
  in the same direction. The template inherits whatever reading of the
  original makes the reference differ. It affects both arms and both models,
  so it cancels in the drops.
- **Inkling's misses** are spread across arms: 5 originals and 9 variants,
  with no excess on the changed items.

**MMLU-Pro (secondary).** Shuffling the options changed neither model's score
detectably.
- qwen38fn moved up 1.0 pp and Inkling moved down 1.2 pp.
- The difference interval (−5.1, +0.6) leans toward Inkling being the more
  order-sensitive of the two, but it includes 0.
- Accuracy by the correct option's letter (in `summary.json`) shows no
  consistent letter preference in either model. There are 29–64 items per
  letter, so every letter-level number is noisy.

**HumanEval (secondary).** Both models scored slightly higher with renamed
entry points, but within noise: the intervals are about ±6 pp wide with 164
items.

Inkling never ends a raw HumanEval completion. All 328 runs hit the
16,384-token cap, writing numbered copies of the function (`rolling_max_2`
… `rolling_max_100`) after the answer. The bench sends no stop strings
(protocol v4). Its extractor applies the classic stop list after the fact,
and only when the plain splice does not parse (protocol v5). So when the
cut-off text happens to parse, the extra code runs together with the answer.

An **exploratory, post-hoc** check (`explore_humaneval_runaway.py`) tested
whether that extra code causes failures:
- It took every failed job whose plain splice parsed: 39 for Inkling and 37
  for qwen38fn.
- It re-ran each in the bench's own sandbox with the completion cut to the
  first function.
- **Inkling:** 3 of 39 pass that way, 2 originals and 1 variant.
- **qwen38fn:** 0 of 37.

So Inkling's failures are genuine (for example, its `rolling_max` crashes on
an empty list). The runaway costs it at most 3 items. It shifts Inkling's
HumanEval drop by 0.6 pp and changes no reading. The check is in
`explore-humaneval-runaway.json` and is not part of the preregistered
reading.

**Answer length.** Median completion tokens are within 1–5 tokens between O
and P for both models in GSM8K and MMLU-Pro. A model that recognized a
memorized item might answer it more briefly; that does not show up here.
Inkling's HumanEval median is the 16,384 cap in both arms, as explained
above.

## Also reported (not read)

Original-arm accuracy next to each model's earlier scores. All are at
thinking, effort low, with a 16,384-token budget. Run-to-run variation under
concurrency is about 1–2 pp.

| Model | Suite | This run (O arm) | Holdout | Dev split |
|---|---|---|---|---|
| qwen38fn | GSM8K | 195/196 (99.5%) | 98/100 | 99/100 (c4) |
| qwen38fn | MMLU-Pro | 428/491 (87.2%) | 218/247 (88.3%) | 221/253 (87.4%, c4) |
| qwen38fn | HumanEval | 139/164 (84.8%) | 138/164 (84.1%) | 139/164 (c4) |
| inkling2mini | GSM8K | 191/196 (97.4%) | 94/100 | 96/100 (effort pick, a6) |
| inkling2mini | MMLU-Pro | 412/491 (83.9%) | 204/247 (82.6%) | 213/253 (84.2%, effort pick, a6) |
| inkling2mini | HumanEval | 113/164 (68.9%) | 76/96 when the power loss cut the draw | — |

## What this cannot show

- **A drop in only one model** would be evidence of memorized items, but it
  could also be brittleness to surface form. This design cannot fully
  separate the two. No such drop appeared.
- **"No drop" does not mean "not bench-maxed"** in the broader sense of heavy
  training on benchmark-style data. A model can learn the style and
  difficulty of GSM8K-like problems without memorizing them. Changed numbers
  test memorized answers, not that. Settling the broader question needs
  problems written after both models' training cutoffs.
- **Residual difficulty shifts.** The GSM8K variants are magnitude-matched
  and balanced around the original answer, but any residual difficulty shift
  cancels in the difference only to the extent both models are equally
  sensitive to it.

What it does show: qwen38fn's GSM8K lead over Inkling (99.5% vs 97.4% here)
survives new numbers intact. Its MMLU-Pro and HumanEval leads (87.2% vs
83.9%, and 84.8% vs 68.9%) survive shuffled options and renamed functions.

## How it ran

**Endpoint.** Both models were served by the qualified production profile on
`127.0.0.1:8013` with the production key, at concurrency 3. Both arms of a
model ran in one seeded interleave (drand round 6670473, `seed.json`). The
runner, the builders and `31_bench_accuracy.py` match `FROZEN.sha256`, and
the items and variants match `BUILT.sha256`. `run-arm.sh` checked both before
each run.

**inkling2mini** ran from 2026-11-24 18:34Z to 2026-11-25 03:28Z, while
production was switched to it. The round trip is in
`results/inkling2-gates/switch-2026-11-24/` (PR #26).
- At 02:17Z, a Claude Code session teardown killed the background shell
  running the arm, at 1,627 of 1,898 jobs.
- Transcripts are written atomically. All 1,627 read back intact, and the
  resumed run skipped them (`arm.log` records the restart).
- The engine process was untouched.

**qwen38fn** ran from 2026-11-25 03:41Z to 09:22Z, after the switch back,
against the production default.

**Kernel.** 0 NVRM, Xid, OOM or cgroup-kill lines in either run. Besides the
routine snap AppArmor denials, the only kernel lines were four loop-device
mounts at 09:20Z, from a snapd refresh (`core26`, `mesa-2604` and others).

**Memory.**
- **inkling2mini run:** MemAvailable low 25.1 GiB, and swap unchanged.
- **qwen38fn run:** MemAvailable stayed at 11.4–12.2 GiB, which is
  production's normal level. From 06:56Z, and mostly after 08:13Z, the
  kernel paged out 48 MiB of idle daemon memory, mostly the Proton VPN daemon (30 MiB), plus tailscaled and
  fwupd. The engine was not touched, and MemAvailable did not move. This was
  recorded, not treated as a failure: this is a diagnostic, and the swap came
  from the host's normal reclaim at production headroom, not from the run's
  own memory.

## Files

- `PREREG.md`: design, analysis rules, review and rebuild record.
- `FROZEN.sha256`, `BUILT.sha256`, `seed.json`: the freeze and the seed.
- `items.jsonl`, `variants/`: the 1,898 jobs and the changed items.
- `runs/<model>/`, per model:
  - `arm.log`, `generate.log`, `score.log`;
  - `run-generate-*.json`: the served model card and hashes;
  - `transcripts/`: one per job, with the rendered-prompt hash, completion,
    usage and score;
  - `observe/`: 1 Hz meminfo and the kernel log.
- `analyze.py` produces `summary.json` and `analyze.out` (the preregistered
  reading).
- `explore_humaneval_runaway.py` produces `explore-humaneval-runaway.json`
  and `.log` (exploratory, post hoc).
