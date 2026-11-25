"""Validate a batch of GSM8K templates against its source rows.

Usage: python3 -B validate_templates.py <batch-XX.json source rows> <templates-XX.json>
Prints one line per row and exits 0 only when every row in the batch has a valid template.
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_template_lib as lib  # noqa: E402

source = {r["row"]: r for r in json.loads(Path(sys.argv[1]).read_text())}
templates = json.loads(Path(sys.argv[2]).read_text())
by_row = {}
bad = 0
for entry in templates:
    if entry.get("row") in by_row:
        print(f"row {entry.get('row')}: duplicate entry")
        bad += 1
    by_row[entry.get("row")] = entry
for row, src in source.items():
    if row not in by_row:
        print(f"row {row}: MISSING")
        bad += 1
        continue
    if by_row[row].get("exclude"):
        print(f"row {row}: EXCLUDED: {by_row[row]['exclude']}")
        continue
    problems = lib.check_entry(by_row[row], src["question"], src["reference_solution"], random.Random(row))
    if problems:
        bad += 1
        print(f"row {row}: FAIL")
        for p in problems:
            print(f"    - {p}")
    else:
        print(f"row {row}: ok")
extra = set(by_row) - set(source)
if extra:
    print(f"rows not in this batch: {sorted(extra)}")
    bad += 1
excluded = sum(1 for r in source if by_row.get(r, {}).get("exclude"))
print(f"{len(source) - bad - excluded}/{len(source)} valid, {excluded} excluded")
sys.exit(1 if bad else 0)
