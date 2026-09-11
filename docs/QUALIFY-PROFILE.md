# Qualifying a profile: one command, one bundle

```
scripts/94_qualify_profile.py --profile <model>/<file> [--host FILE] [--port 8015]
    [--baseline <model>/<file> | --baseline-results <dir>]
    [--out results/<model>-gates/qualify-<YYYY-MM-DD>]
    [--cells speed,toolcall,vision,media,teacher,accuracy,context,soak] [--skip <cells>]
    [--with-soak] [--no-launch] [--served-model NAME] [--dry-run] [--resummarize]
    [--reasoning-effort low]
```

The kit resolves the profile (`92_resolve_profile.py render` + `check`),
starts it on the dev port with `scripts/93_profile_serve.sh` (unless
`--no-launch`), waits until `/v1/models` lists the served model, runs the
cells in order, and always stops the profile again (also on failure or
Ctrl-C). It measures nothing itself: every number is parsed from the composed
script's own output file. `--dry-run` resolves everything, writes
`manifest.json` with the planned argv of each cell, and exits 0 with no server
and no model on disk.

## Cells

| cell | proves | script (exact argv in `manifest.json`) | evidence |
|---|---|---|---|
| speed | decode tok/s, TTFT, prefill tok/s at ctx 0 and 28,672 (2 reps + 1 warm-up, seed 42, `min_tokens` 320, output-token validation against the model tokenizer) | `30_bench_speed.py` | `speed/speed.json` |
| toolcall | deterministic tool-call probe, temperature 0 | `39_bench_toolcall.py` | `toolcall/toolcall.json` |
| vision | MMMU-val-100 accuracy through chat completions, chat (non-thinking) template mode like the README rows, unless the profile's `qualification_options.vision_thinking_mode` or `--vision-thinking-mode` selects `thinking` | `38_bench_vision.py` | `vision/summary.json` |
| media | the profile's declared `--limit-mm-per-prompt` maximum: N solid-colour images at WxH named in order, one F-frame video's motion direction, and N+1 images rejected with 400 | `51_probe_media_max.py` | `media/summary.json` |
| teacher | token-weighted delta-NLL / top-1 loss vs BF16 teacher logits (AGENTS.md fidelity limits) | `49_score_teacher_windows.py` | `teacher/summary.json` |
| accuracy | gsm8k / mmlu-pro / humaneval (holdout by default; `--accuracy-split dev`) | `31_bench_accuracy.py` x3 | `accuracy/acc-<suite>.json` |
| context | direct full fill: `serving.parallel_slots` x (tokens-per-slot - 12,016 headroom for template + answer) with needles, controls, memory floor = `safety.kill_floor_gib` | `50_probe_context.py --phase both` | `context/summary.json` |
| soak | 30-minute sustained load (off by default, `--with-soak`) | `35_soak.py` | `soak/soak.json` |

Context runs last because it takes about an hour. Every cell's stdout and
stderr go to `<out>/<cell>/log.txt`; `manifest.json` records host facts, git
HEAD, the sha256 of every composed script, the resolved profile (served model,
tokenizer sha256, slots, context cap, kill floor, digest checks, engine binary
sha256), and start/end times.

## Reading a FAIL

`SUMMARY.md` has one row per metric: measured value, target (or `-`),
baseline column when one was given, and a status of `PASS`, `FAIL`,
`measured`, or `SKIPPED`. The overall verdict is `PASS` only if every
non-skipped cell with a target passes and no cell errored; `FAIL` otherwise;
`MEASURED` when the profile declares no targets at all.

- A cell whose script exited non-zero, or wrote no evidence, is `FAIL` with
  the tail of its `log.txt` under "Failed cells". Read the full log, fix the
  cause, re-run only that cell with `--cells <cell> --out <same bundle>`
  (move the old cell directory aside first: scripts refuse to overwrite their
  evidence). The other cells' rows are kept from the bundle's previous
  manifest and re-parsed from their evidence on disk; `manifest.json` lists
  them under `cells_kept_from_previous_run`. `--resummarize --out <bundle>`
  rebuilds `summary.json`/`SUMMARY.md` from the evidence alone (no server).
- Gate scripts (context, teacher) exit 1 on a `FAIL` verdict but still write
  `summary.json`; the row keeps the parsed verdict so you can see which check
  failed (`context/summary.json` `checks`, `teacher/summary.json`
  `fail_reasons`).
- `SKIPPED` rows carry the reason (no encoder, no `reference_logits`, no
  tokenizer, soak off). A skip never counts as a pass.
- Never edit a bundle by hand; re-run and keep both.

## Targets and baselines

Targets live in the profile JSON as an optional `qualification_targets`
block (docs/PROFILE-SCHEMA.md), or in a file passed with `--targets`:

```json
"qualification_targets": {
  "decode_tok_s_min": {"0": 17.46, "28672": 26.71},
  "prefill_tok_s_min": {"28672": 698.7},
  "ttft_s_max": {"0": 0.5},
  "context_pass": true, "toolcall_min": 19, "vision_min": 0.64,
  "delta_nll_max": 0.01, "top1_loss_pp_max": 0.5,
  "accuracy_min": {"gsm8k": 0.95}, "soak_pass": true
}
```

`--baseline <model>/<file>` reads that profile's `status.evidence` directory
and uses the newest `qualify-*/summary.json` written by this kit;
`--baseline-results <dir>` points at any prior bundle. The baseline value is
rendered side by side; it never gates.

## What a new model needs before the kit is useful

1. A profile JSON with `serving` (`parallel_slots` x `request_context_cap`)
   and `safety` (`kill_floor_gib`), a `--served-model-name`/`--alias` in
   its argv, and `bench.stack_label`.
2. `tokenizer.json` in the rendered `--model` directory (speed validates
   output token ids against it; the context probe needs it to fill slots
   exactly).
3. Optional: a `reference_logits` block in `model.json` (teacher cell) and a
   chat encoder registered in `scripts/31_bench_accuracy.py` `ENCODER_PATHS`
   (accuracy cell; `--encoder NAME`). Without them those cells are SKIPPED
   with a reason.
