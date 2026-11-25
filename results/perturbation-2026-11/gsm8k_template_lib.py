"""GSM8K variant templates: rendering, sampling and validation.

A template entry (one per GSM8K row):
{
  "row": <GSM8K test index>,
  "template": "<the original question with each varied value replaced by {var}; literal braces doubled>",
  "vars": {
     "<var>": {"kind": "int", "original": 48, "min": 20, "max": 90, "step": 2, "fmt": "plain"|"comma"},
     "<var>": {"kind": "num_choice", "original": "2.50", "values": ["1.50", "2.25", "3.75"]},
     "<var>": {"kind": "name", "original": "Natalia", "pool": ["Priya", "Maria", ...]}
  },
  "constraints": ["<python boolean expression over v['<var>']>", ...],
  "solver": "def solve(v):\n    ...\n    return <answer>"
}
solve(v) gets a dict: int vars as int, num_choice vars as Fraction, name vars as str. It returns
an int or a Fraction (use Fraction, never float). It may `assert` that intermediate quantities are
sensible (whole people, non-negative stock); a failed assert rejects that sample.
"""
from __future__ import annotations

import random
import re
from decimal import Decimal
from fractions import Fraction
from typing import Any

MIN_ACCEPT_RATE = 0.02


def reference_answer(solution: str) -> Fraction:
    return Fraction(Decimal(solution.rsplit("#### ", 1)[1].strip().replace(",", "")))


def render_value(spec: dict, value: Any) -> str:
    if spec["kind"] == "int":
        return f"{value:,}" if spec.get("fmt") == "comma" else str(value)
    return str(value)


def solver_value(spec: dict, value: Any) -> Any:
    if spec["kind"] == "num_choice":
        return Fraction(Decimal(str(value).replace(",", "")))
    return value


def render(entry: dict, values: dict) -> str:
    return entry["template"].format(**{k: render_value(entry["vars"][k], v) for k, v in values.items()})


def originals(entry: dict) -> dict:
    return {k: s["original"] for k, s in entry["vars"].items()}


def compile_solver(source: str):
    namespace: dict[str, Any] = {"Fraction": Fraction, "math": __import__("math")}
    exec(source, namespace)  # noqa: S102 - our own reviewed template code
    return namespace["solve"]


def solve(solver, entry: dict, values: dict) -> Fraction:
    result = solver({k: solver_value(entry["vars"][k], v) for k, v in values.items()})
    if isinstance(result, bool) or not isinstance(result, (int, Fraction)):
        raise TypeError(f"solver returned {type(result).__name__}; use int or Fraction")
    return Fraction(result)


def sample(entry: dict, rng: random.Random) -> dict:
    values: dict[str, Any] = {}
    used_names: set[str] = set()
    for k in sorted(entry["vars"]):
        s = entry["vars"][k]
        if s["kind"] == "int":
            values[k] = rng.randrange(s["min"], s["max"] + 1, s.get("step", 1))
        elif s["kind"] == "num_choice":
            values[k] = rng.choice(s["values"])
        else:
            choices = [n for n in s["pool"] if n not in used_names]
            values[k] = rng.choice(choices)
            used_names.add(values[k])
    return values


def accept(entry: dict, solver, values: dict, original_answer: Fraction) -> tuple[bool, str, Fraction | None]:
    env = {"v": {k: solver_value(entry["vars"][k], x) for k, x in values.items()}, "Fraction": Fraction}
    for c in entry.get("constraints", []):
        if not eval(c, env):  # noqa: S307 - our own reviewed template code
            return False, f"constraint {c}", None
    numeric = [k for k, s in entry["vars"].items() if s["kind"] != "name"]
    if all(values[k] == entry["vars"][k]["original"] for k in numeric):
        return False, "numbers unchanged", None
    try:
        answer = solve(solver, entry, values)
    except (AssertionError, ZeroDivisionError, ValueError) as error:
        return False, f"solver rejected: {type(error).__name__}", None
    if answer == original_answer:
        return False, "answer unchanged", None
    if original_answer.denominator == 1 and answer.denominator != 1:
        return False, "non-integer answer", None
    if original_answer.denominator != 1 and (answer * 100).denominator != 1:
        return False, "more than 2 decimal places", None
    if original_answer > 0 and answer <= 0:
        return False, "non-positive answer", None
    return True, "ok", answer


def format_answer(answer: Fraction) -> str:
    if answer.denominator == 1:
        return str(answer.numerator)
    return format(Decimal(answer.numerator) / Decimal(answer.denominator), "f").rstrip("0").rstrip(".")


def check_entry(entry: dict, question: str, solution: str, rng: random.Random, tries: int = 200) -> list[str]:
    """Return a list of problems; empty means the entry is valid."""
    problems = []
    vars_ = entry.get("vars", {})
    if not vars_:
        return ["no vars"]
    for k, s in vars_.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", k):
            problems.append(f"{k}: var names must be lowercase identifiers")
        kind = s.get("kind")
        if kind == "int":
            if not all(isinstance(s.get(f), int) for f in ("original", "min", "max")):
                problems.append(f"{k}: int needs integer original/min/max")
                continue
            if s["min"] >= s["max"]:
                problems.append(f"{k}: empty range")
            if s["original"] > 0 and (s["max"] > 4 * s["original"] or s["min"] < s["original"] / 4):
                problems.append(f"{k}: range {s['min']}..{s['max']} is not within 4x of the original {s['original']}")
        elif kind == "num_choice":
            if not isinstance(s.get("original"), str) or not s.get("values") or s["original"] in s["values"]:
                problems.append(f"{k}: num_choice needs a string original and values that exclude it")
        elif kind == "name":
            pool = s.get("pool") or []
            if len(pool) < 6 or s.get("original") in pool:
                problems.append(f"{k}: name pool needs at least 6 names, excluding the original")
        else:
            problems.append(f"{k}: unknown kind {kind!r}")
    if not any(s.get("kind") in ("int", "num_choice") for s in vars_.values()):
        problems.append("no numeric var: a memorized answer would still be right")
    if problems:
        return problems
    try:
        rendered = render(entry, originals(entry))
    except (KeyError, IndexError, ValueError) as error:
        return [f"template does not render: {error!r}"]
    if rendered != question:
        i = next((n for n, (a, b) in enumerate(zip(rendered, question)) if a != b), min(len(rendered), len(question)))
        return [f"template at original values differs from the question at char {i}: "
                f"{rendered[max(0, i - 30):i + 30]!r} vs {question[max(0, i - 30):i + 30]!r}"]
    placeholders = set(re.findall(r"(?<!\{)\{([a-z][a-z0-9_]*)\}(?!\})", entry["template"]))
    if placeholders != set(vars_):
        problems.append(f"placeholders {sorted(placeholders)} != vars {sorted(vars_)}")
    try:
        solver = compile_solver(entry["solver"])
        got = solve(solver, entry, originals(entry))
    except Exception as error:  # noqa: BLE001 - report any solver failure
        return problems + [f"solver fails at the original values: {error!r}"]
    expected = reference_answer(solution)
    if got != expected:
        problems.append(f"solver at original values gives {got}, the reference answer is {expected}")
    env = {"v": {k: solver_value(s, s["original"]) for k, s in vars_.items()}, "Fraction": Fraction}
    for c in entry.get("constraints", []):
        try:
            if not eval(c, env):  # noqa: S307
                problems.append(f"constraint fails at the original values: {c}")
        except Exception as error:  # noqa: BLE001
            problems.append(f"constraint {c!r} raises {error!r}")
    if problems:
        return problems
    accepted = 0
    for _ in range(tries):
        try:
            ok, _, _ = accept(entry, solver, sample(entry, rng), expected)
        except Exception as error:  # noqa: BLE001
            return [f"sampling raised {error!r}"]
        accepted += ok
    if accepted < MIN_ACCEPT_RATE * tries:
        problems.append(f"only {accepted}/{tries} samples are acceptable; widen ranges or fix constraints")
    return problems
