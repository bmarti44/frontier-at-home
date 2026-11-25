"""Check a second-solver file: each solver runs and matches the reference answer at the original values.

Usage: python3 -B check_solvers2.py <gsm8k-blind/blind-XX.json> <gsm8k-solvers2/solvers-XX.json>
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_template_lib as lib  # noqa: E402

blind = {e["row"]: e for e in json.loads(Path(sys.argv[1]).read_text())}
sol = {e["row"]: e for e in json.loads(Path(sys.argv[2]).read_text())}
bad = 0
for row, e in blind.items():
    if row not in sol:
        print(f"row {row}: MISSING"); bad += 1; continue
    try:
        s = lib.compile_solver(sol[row]["solver"])
        got = lib.solve(s, e, lib.originals(e))
    except Exception as error:  # noqa: BLE001
        print(f"row {row}: solver fails at the original values: {error!r}"); bad += 1; continue
    want = lib.reference_answer("#### " + e["reference_answer_at_original_values"])
    if got != want:
        print(f"row {row}: gives {got} at the original values, expected {want}"); bad += 1; continue
    rng = random.Random(row)
    crashes = 0
    for _ in range(200):
        values = lib.sample(e, rng)
        try:
            lib.solve(s, e, values)
        except (AssertionError, ZeroDivisionError, ValueError):
            pass
        except Exception as error:  # noqa: BLE001
            crashes += 1
            last = error
    if crashes:
        print(f"row {row}: {crashes}/200 random samples raise {last!r} (only assert/ZeroDivision/ValueError may reject)"); bad += 1; continue
    print(f"row {row}: ok")
print(f"{len(blind) - bad}/{len(blind)} ok")
sys.exit(1 if bad else 0)
