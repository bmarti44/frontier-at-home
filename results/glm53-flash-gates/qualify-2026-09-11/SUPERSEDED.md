# Superseded by qualify-2026-09-11-rerun

This bundle's `SUMMARY.md` reports the vision cell as 0.73 PASS, but the
cell's own evidence in `vision/summary.json` records 0.19 with 77 unparseable
answers (chat template mode, `--thinking-mode chat`). The 0.73 figure came
from `../qualify-2026-09-11-cells/vision/` (the model's own thinking mode,
12 unparseable) and was carried into this SUMMARY.md by an earlier partial
re-run of the kit. Nothing here was edited; the files are left as written.

`qualify-2026-09-11-rerun/` is a full fresh run of every cell on the same
profile (2026-09-11 14:37Z onward) with the vision cell measured in the
mode the profile now declares (`qualification_options.vision_thinking_mode`)
and the chat-mode transcripts kept beside it in `vision-chat-mode/`.
Use the rerun bundle for the README row and `STATUS.md`.
