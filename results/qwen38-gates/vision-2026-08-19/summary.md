# Qwen3.8-27B vision gate — 2026-08-19

Q4_K_M + mmproj-f16, b10488, n8p6 MTP, chat (non-thinking) rendering,
max_tokens 512, pinned evalset evalsets/mmmu-val-100 (MMMU validation,
rev 98e6ac0c, 100 single-image MC cases, 30 subjects, pins verified
before the run).

**64/100 correct, 3 unparseable (scored invalid, not guessed), 0
transport errors.** In line with published MMMU expectations for this
model class at Q4; thinking mode may score higher (not measured — the
qualification target was the production chat rendering). Full parsed/
expected digest in digest.json; raw transcripts retained in session
scratchpad.
