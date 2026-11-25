"""GSM8K variants from the validated templates: new numbers and names, answers recomputed.

Usage: python3 -B build_gsm8k.py <rows/gsm8k.jsonl> <templates dir> <solvers2 dir> <seed> <out.jsonl>

For every row with a template, draw 2 distinct acceptable (gsm8k_template_lib.accept) and
magnitude-matched value sets from a per-row RNG keyed by the public seed. Each drawn variant's answer must agree with the second,
independently written solver; before drawing, the two solvers must also agree on 300 random
acceptable samples. A row whose solvers disagree is reported and the build fails.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gsm8k_template_lib as lib  # noqa: E402

VARIANTS = 2
AGREEMENT_SAMPLES = 300
DRAW_TRIES = 200000
ANSWER_RATIO = (Fraction(1, 2), Fraction(2))


def magnitude_matched(entry: dict, values: dict, answer: Fraction, original: Fraction) -> bool:
    """Review A-H1: variants keep the originals' size, so the drop is not a test of bigger arithmetic.
    Every int var keeps its original number of digits and the answer stays within 0.5-2x the original."""
    for k, spec in entry["vars"].items():
        if spec["kind"] == "int" and len(str(abs(values[k]))) != len(str(abs(spec["original"]))):
            return False
    if original == 0:
        return answer == 0
    ratio = answer / original
    return ANSWER_RATIO[0] <= ratio <= ANSWER_RATIO[1]


def load_dir(path: Path, prefix: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for p in sorted(path.glob(f"{prefix}-*.json")):
        for entry in json.loads(p.read_text()):
            assert entry["row"] not in out, ("duplicate row", entry["row"], p.name)
            out[entry["row"]] = entry
    return out


def main() -> int:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    templates = load_dir(Path(sys.argv[2]), "templates")
    solvers2 = load_dir(Path(sys.argv[3]), "solvers")
    seed = sys.argv[4]
    failures = 0
    written = excluded = 0
    ratios: list[float] = []
    one_sided: list[int] = []
    digit_shift: list[int] = []
    with open(sys.argv[5], "w") as f:
        for r in rows:
            entry = templates.get(r["row"])
            if entry is None:
                print("MISSING template", r["row"])
                failures += 1
                continue
            if entry.get("exclude"):
                print("excluded", r["row"], entry["exclude"])
                excluded += 1
                continue
            problems = lib.check_entry(entry, r["question"], r["answer"], random.Random(r["row"]))
            if problems:
                print("INVALID template", r["row"], problems)
                failures += 1
                continue
            if r["row"] not in solvers2:
                print("MISSING second solver", r["row"])
                failures += 1
                continue
            s1 = lib.compile_solver(entry["solver"])
            s2 = lib.compile_solver(solvers2[r["row"]]["solver"])
            original = lib.reference_answer(r["answer"])
            # Agreement on the original values and on random acceptable samples.
            disagreements = []
            try:
                if lib.solve(s2, entry, lib.originals(entry)) != original:
                    disagreements.append(("original", None))
            except Exception as error:  # noqa: BLE001
                disagreements.append(("original", repr(error)))
            check_rng = random.Random(f"agree:{r['row']}")
            checked = tries = 0
            while checked < AGREEMENT_SAMPLES and tries < 50 * AGREEMENT_SAMPLES:
                tries += 1
                values = lib.sample(entry, check_rng)
                ok, _, a1 = lib.accept(entry, s1, values, original)
                if not ok:
                    continue
                checked += 1
                try:
                    a2 = lib.solve(s2, entry, values)
                except Exception as error:  # noqa: BLE001
                    disagreements.append((values, repr(error)))
                    continue
                if a1 != a2:
                    disagreements.append((values, f"{a1} vs {a2}"))
            if disagreements:
                print("DISAGREE", r["row"], len(disagreements), disagreements[:3])
                failures += 1
                continue
            rng = random.Random(hashlib.sha256(f"{seed}:gsm8k:{r['row']}".encode()).hexdigest())
            # One variant with a smaller answer and one with a larger answer when the template allows it,
            # so the variants are not harder on average; otherwise both from the side that exists.
            pools: dict[str, list[tuple[dict, str, object]]] = {"lower": [], "higher": []}
            for _ in range(DRAW_TRIES):
                values = lib.sample(entry, rng)
                ok, _, answer = lib.accept(entry, s1, values, original)
                if not ok or not magnitude_matched(entry, values, answer, original):
                    continue
                question = lib.render(entry, values)
                side = "lower" if answer < original else "higher"
                if any(question == q for _, q, _ in pools[side]):
                    continue
                if len(pools[side]) < VARIANTS:
                    pools[side].append((values, question, answer))
                if all(len(v) >= 1 for v in pools.values()):
                    break
            if pools["lower"] and pools["higher"]:
                drawn = [pools["lower"][0], pools["higher"][0]]
            else:
                drawn = (pools["lower"] or pools["higher"])[:VARIANTS]
                one_sided.append(r["row"])
            if len(drawn) < VARIANTS:
                print("TOO FEW variants", r["row"], len(drawn))
                failures += 1
                continue
            for k, (values, question, answer) in enumerate(drawn):
                ratios.append(float(answer / original) if original else 1.0)
                digit_shift.append(len(str(lib.format_answer(answer))) - len(str(lib.format_answer(original))))
                f.write(json.dumps({"split": r["split"], "row": r["row"], "variant": k,
                                    "perturbation": {"kind": "template_resample", "values": values},
                                    "item": {"question": question,
                                             "answer": "#### " + lib.format_answer(answer)}}) + "\n")
                written += 1
    print(f"variants {written}, excluded rows {excluded}, failures {failures}")
    if ratios:
        print(f"answer ratio variant/original: median {statistics.median(ratios):.3f}, "
              f"quartiles {statistics.quantiles(ratios, n=4)}, "
              f"answers with more digits {sum(d > 0 for d in digit_shift)}, fewer {sum(d < 0 for d in digit_shift)}; "
              f"rows with both variants on one side: {len(one_sided)} {one_sided}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
