"""Cross-check the two solvers of a batch on random acceptable samples (the build's check, per batch).

Usage: python3 -B agree.py <batch number>
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_template_lib as lib  # noqa: E402

here = Path(__file__).resolve().parent
n = sys.argv[1]
src = {r["row"]: r for r in json.loads((here / f"gsm8k-batches/batch-{n}.json").read_text())}
s2 = {e["row"]: e for e in json.loads((here / f"gsm8k-solvers2/solvers-{n}.json").read_text())}
bad = 0
for e in json.loads((here / f"gsm8k-templates/templates-{n}.json").read_text()):
    if e.get("exclude"):
        continue
    row = e["row"]
    a = lib.compile_solver(e["solver"]); b = lib.compile_solver(s2[row]["solver"])
    orig = lib.reference_answer(src[row]["reference_solution"])
    rng = random.Random(f"agree:{row}")
    checked = tries = 0
    dis = []
    while checked < 300 and tries < 15000:
        tries += 1
        v = lib.sample(e, rng)
        ok, _, x = lib.accept(e, a, v, orig)
        if not ok:
            continue
        checked += 1
        try:
            y = lib.solve(b, e, v)
        except Exception as error:  # noqa: BLE001
            dis.append((v, repr(error))); continue
        if x != y:
            dis.append((v, f"{x} vs {y}"))
    if dis:
        bad += 1
        print(f"row {row}: {len(dis)}/{checked} disagree, e.g. {dis[0]}")
print(f"batch {n}: {bad} rows disagree")
