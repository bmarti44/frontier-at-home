"""Blind inputs for the second solver: the template without the first solver or the reference solution.

Usage: python3 -B make_blind.py <batch number, e.g. 03>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_template_lib as lib  # noqa: E402

here = Path(__file__).resolve().parent
n = sys.argv[1]
src = {r["row"]: r for r in json.loads((here / f"gsm8k-batches/batch-{n}.json").read_text())}
out = []
for e in json.loads((here / f"gsm8k-templates/templates-{n}.json").read_text()):
    if e.get("exclude"):
        continue
    out.append({"row": e["row"], "original_question": src[e["row"]]["question"], "template": e["template"],
                "vars": e["vars"], "constraints": e.get("constraints", []),
                "reference_answer_at_original_values": lib.format_answer(
                    lib.reference_answer(src[e["row"]]["reference_solution"]))})
(here / f"gsm8k-blind/blind-{n}.json").write_text(json.dumps(out, indent=1))
print(n, len(out))
