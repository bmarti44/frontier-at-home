"""MMLU-Pro variants: the options shuffled so the correct option moves to another letter.

Usage: python3 -B build_mmlu.py <rows/mmlu-pro.jsonl> <seed> <out.jsonl>

Excluded (from both arms): items with an option that refers to the other options by position
("none of the above", "all of these"), since a shuffle changes their meaning. Letters used as
content (logic variables D, E, F; atoms A and B) are not references and stay.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from pathlib import Path

POSITIONAL = re.compile(
    r"\b(all|none|both|neither) of (the )?(above|below|these|those|the (above|following|options|choices|answers))\b",
    re.I,
)


def item_rng(seed: str, row: int) -> random.Random:
    return random.Random(hashlib.sha256(f"{seed}:mmlu-pro:{row}".encode()).hexdigest())


def main() -> int:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    seed = sys.argv[2]
    kept = excluded = 0
    with open(sys.argv[3], "w") as f:
        for r in rows:
            if any(POSITIONAL.search(o) for o in r["options"]):
                excluded += 1
                print("excluded", r["row"], [o for o in r["options"] if POSITIONAL.search(o)])
                continue
            n = len(r["options"])
            old = ord(r["answer"].upper()) - ord("A")
            rng = item_rng(seed, r["row"])
            while True:
                order = list(range(n))
                rng.shuffle(order)
                if order.index(old) != old:
                    break
            new_answer = chr(ord("A") + order.index(old))
            variant = {k: r[k] for k in ("question_id", "question", "category")}
            variant["options"] = [r["options"][i] for i in order]
            variant["answer"] = new_answer
            assert variant["options"][ord(new_answer) - 65] == r["options"][old]
            f.write(json.dumps({"split": r["split"], "row": r["row"], "variant": 0,
                                "perturbation": {"kind": "shuffle_options", "order": order,
                                                 "answer_from": r["answer"].upper(), "answer_to": new_answer},
                                "item": variant}) + "\n")
            kept += 1
    print(f"kept {kept}, excluded {excluded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
