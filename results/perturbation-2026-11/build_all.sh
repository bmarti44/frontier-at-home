#!/usr/bin/env bash
# Build the variant sets and the job list from the public seed (PREREG "Seed").
# Usage: build_all.sh   (reads seed.json and rename-check.json next to this file; writes variants/ and items.jsonl)
set -euo pipefail
cd "$(dirname "$0")"
seed=$(python3 -c 'import json; print(json.load(open("seed.json"))["seed"])')
mkdir -p variants
python3 -B build_humaneval.py rows/humaneval.jsonl humaneval-names.json variants/humaneval-all.jsonl
# A renamed task whose canonical solution failed the sandbox check leaves both arms (review A-L4).
python3 - <<'PY'
import json
check = json.load(open("rename-check.json"))
drop = set(check["excluded_by_rename"])
assert not check["original_failed"] and not check["static_problems"], "rename check did not complete cleanly"
rows = [json.loads(l) for l in open("variants/humaneval-all.jsonl")]
with open("variants/humaneval-renamed.jsonl", "w") as f:
    for r in rows:
        if r["task_id"] not in drop:
            f.write(json.dumps(r) + "\n")
print(f"humaneval kept {len(rows) - len(drop)}, excluded by the rename check {sorted(drop)}")
PY
rm -f variants/humaneval-all.jsonl
python3 -B build_mmlu.py rows/mmlu-pro.jsonl "$seed" variants/mmlu-pro-shuffled.jsonl
python3 -B build_gsm8k.py rows/gsm8k.jsonl gsm8k-templates gsm8k-solvers2 "$seed" variants/gsm8k-variants.jsonl
python3 -B make_items.py . "$seed" items.jsonl
sha256sum items.jsonl variants/*.jsonl > BUILT.sha256
