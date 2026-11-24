# Holdout result: interrupted by a power loss; qwen38fn stays the default

A site power loss at about 14:47Z on 2026-11-23 stopped the host 96 items into HumanEval, the last of the three suites. The frozen verdict (`holdout_verdict.py`, freeze `3aa7cd00`, run 2026-11-24 before any later commit) returns **`NO_RESULT`** in `holdout-summary.json`: the driver died with the host and never wrote its closing lines.

The interruption does not decide anything. The preregistered rule could not return `PROMOTE` however the remaining HumanEval items had gone, for two independent reasons:
- **MMLU-Pro is significantly worse.** 204/247 against the default's 218/247 gives z = −1.785, past the −1.645 limit. That suite finished and is ledgered.
- **The sum cannot exceed 454.** GSM8K 94 + MMLU-Pro 204 + HumanEval 76 of the first 96 is 374. With all 68 remaining items correct the sum would be 442, below the 455 the rule needs.

So under the README's rule (`KEEP_DEFAULT` or `NO_RESULT`), `qwen38fn` stays the serving default. M4 lands `inkling2mini` as a qualified, switchable engine. No second draw is made: GSM8K and MMLU-Pro are spent in the ledger, and finishing HumanEval cannot change the outcome.

## The run (2026-11-23, one fresh start, head `3aa7cd00`, clean)

- **Start:**
  - production `qwen38fn` stopped through the switch at 10:07:37Z;
  - artifact identity matched (interpreter, vLLM tree `5bcb020bc1e801e3`, patches 0001–0004, package freeze, 14 weight files);
  - ready after 191 s, idle MemAvailable 26.8 GiB, KV capacity 1,238,000 tokens at 262,144 per request.
- **Restart probe:** byte-identical to M3 candidate 2 on all three hashes, so the holdout was drawn at effort low (10:11:49Z).
- **Safety until the outage** (16,624 one-second samples, 10:07:37Z to 14:47:18Z):
  - 0 kill lines, and no NVRM, Xid or OOM line in the run's kernel log or the previous boot's journal;
  - low point 25.51 GiB after the production stop. The first two samples, 11.84 and 14.74 GiB, are `qwen38fn` still serving before its stop completed;
  - swap growth 0.00 GiB;
  - 0 transport errors.
- **Outage:** the host's journal ends at 14:47:16Z with no kernel fault or shutdown sequence, and the meminfo sampler's last line is 14:47:18Z, at 26.99 GiB available. The host came back at 14:10Z on 2026-11-24. `dsv4-engine-restore.service` restored `qwen38fn` (exit 0), which was serving on :8013 with its memwatch armed when this was written. The new boot's kernel log has only the driver's load banner.
- **Ledger:** `results/holdout-ledger.json` has a `started` and a `completed` entry for each of GSM8K and MMLU-Pro. Stack `inkling-2-mini-cuda-spark-128g-1m`, namespace `""`. HumanEval is not a ledgered split.

## Scores (thinking, effort low, 16,384-token budget; the same rows as qwen38fn's draw)

| Suite | Inkling 2 Mini | Default (`qwen38fn`) | Delta | z (one-sided worse if < −1.645) |
|---|---|---|---|---|
| GSM8K holdout | 94/100 (0 invalid) | 98/100 | −4 | −1.443 |
| MMLU-Pro holdout | 204/247 (82.59%; 1 invalid) | 218/247 (88.26%) | −14 | **−1.785** |
| HumanEval, first 96 tasks (interrupted) | 76/96 | 83/96 on the same tasks | −7 | −1.339 |
| **Sum** | at most 442/511 | 454/511 | — | — |

The z values use the README's pooled one-sided formula. The HumanEval row is reported, not ruled: the suite did not finish.

Every HumanEval completion ran to the 16,384-token budget (`finish_reason: length`), about 140 s per item. The model writes the function, then keeps generating, because the raw-prompt protocol sets no stop sequence (`HUMANEVAL_STOPS = None`). The bench's extractor scores the first parseable program. All 20 misses are `sandbox failed with exit 1`, which is a wrong answer.

## The promotion legs

| Leg | Inkling 2 Mini | Rule | Met |
|---|---|---|---|
| Prefill at 28,672 | 3,332.5 tok/s | ≥ 2,711.5 | yes |
| Decode at 0 / 28,672 | 36.14 / 44.16 tok/s | ≥ 34.07 / 40.92 | yes |
| MMMU-val-100 | 77 | ≥ 64 | yes |
| Tool-call | 20/20 | ≥ 19 | yes |
| Holdout sum > 454 | at most 442 | > 454 | **no** |
| No suite significantly worse | MMLU-Pro z −1.785 | z ≥ −1.645 | **no** |

Inkling 2 Mini is faster on every leg (M3 candidate 2, `../qualify-2026-11-23-c2/RESULT.md`), but it is not more accurate on text.

## Files
- `holdout-summary.json`: the frozen verdict's output (`NO_RESULT`).
- `accuracy/`: the two finished suites, the 96 HumanEval transcripts and the bench log.
- `observe/`: the driver log, kernel log, cgroup log and meminfo samples. `driver-console.log` is the driver's console log copied from `~/.frontier-at-home/logs/`.
- `manifest.json`, `probe-holdout.json`: the kit's manifest and the restart probe.
