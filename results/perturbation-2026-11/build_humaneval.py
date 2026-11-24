"""HumanEval variants: the entry-point function renamed to a synonym of the same informativeness.

Usage: python3 -B build_humaneval.py <rows/humaneval.jsonl> <humaneval-names.json> <out.jsonl>

Rename rule (mechanical, no other edit):
- every call-like use `name(` in the prompt, the canonical solution and the test;
- prose mentions too when the name is an identifier (contains "_" or a digit), or is written
  "function <name>"; plain English words used as prose ("return the longest one") stay.
The test calls `candidate`, so only HumanEval/33, whose test calls `sort_third` itself, changes there.
Static checks: the new name is unused in the original row, prompt + canonical solution parses, and the
old name no longer appears as a code identifier. The sandbox check (renamed canonical passes the renamed
test) runs separately, when no other sandbox work is running.
"""
from __future__ import annotations

import ast
import io
import json
import keyword
import re
import sys
import tokenize
from pathlib import Path


def rename(text: str, old: str, new: str, prose: bool) -> str:
    text = re.sub(rf"\b{re.escape(old)}\b(?=\s*\()", new, text)
    text = re.sub(rf"(?<=function ){re.escape(old)}\b", new, text)
    if prose:
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def code_names(source: str) -> set[str]:
    names = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.NAME:
            names.add(tok.string)
    return names


def build(row: dict, new: str) -> tuple[dict, list[str]]:
    old = row["entry_point"]
    problems = []
    if not new.isidentifier() or keyword.iskeyword(new) or new in dir(__builtins__):
        problems.append(f"{new!r} is not a usable identifier")
    if re.search(rf"\b{re.escape(new)}\b", row["prompt"] + row["canonical_solution"] + row["test"]):
        problems.append(f"{new!r} already appears in the row")
    # Identifier-like names (and "fibfib", not an English word) are renamed in prose as well.
    prose = "_" in old or any(c.isdigit() for c in old) or old == "fibfib"
    out = dict(row)
    out["prompt"] = rename(row["prompt"], old, new, prose)
    out["canonical_solution"] = rename(row["canonical_solution"], old, new, prose)
    out["test"] = rename(row["test"], old, new, prose)
    out["entry_point"] = new
    program = out["prompt"] + out["canonical_solution"]
    try:
        ast.parse(program)
    except SyntaxError as error:
        problems.append(f"renamed program does not parse: {error}")
        return out, problems
    if old in code_names(program) or old in code_names(out["test"]):
        problems.append(f"old name {old!r} is still a code identifier")
    if f"def {new}(" not in out["prompt"]:
        problems.append("renamed prompt does not define the new name")
    left = [m.start() for m in re.finditer(rf"\b{re.escape(old)}\b", out["prompt"])]
    out["perturbation"] = {"kind": "rename_entry_point", "from": old, "to": new,
                           "old_name_left_in_prose": len(left)}
    return out, problems


def main() -> int:
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text().splitlines()]
    names = json.loads(Path(sys.argv[2]).read_text())
    bad = 0
    with open(sys.argv[3], "w") as f:
        for row in rows:
            number = row["task_id"].split("/")[1]
            out, problems = build(row, names[number])
            if problems:
                bad += 1
                print(row["task_id"], problems)
            f.write(json.dumps(out) + "\n")
    print(f"{len(rows) - bad}/{len(rows)} renamed cleanly")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
