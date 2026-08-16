#!/usr/bin/env python3
"""Build the committed-shape bundle for the decisive matched-32K campaign."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable


SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from glm52_goal import score_registered_gate  # noqa: E402


TOPIC = "lossless-plateau-candidate15-matched32k"
SCORER_ID = "parity.performance.v1"
Scorer = Callable[[str, str, Iterable[dict[str, Any]]], dict[str, Any]]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_json_bytes(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is invalid: {exc}") from exc


def _load_rows(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"raw.jsonl line {line_number} is invalid: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"raw.jsonl line {line_number} is not a JSON object")
        rows.append(row)
    return rows


def _drand_round(receipt: dict[str, Any]) -> int:
    value = receipt.get("round")
    if value is None:
        inner = receipt.get("receipt")
        if isinstance(inner, dict):
            value = inner.get("round")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("drand round is absent or invalid")
    return value


def _fixture_digest(rows: list[dict[str, Any]]) -> str:
    if not rows or any("fixture_sha256" not in row for row in rows):
        raise ValueError("fixture_sha256 is absent from one or more raw rows")
    digests = {row["fixture_sha256"] for row in rows}
    if len(digests) != 1:
        raise ValueError("raw rows disagree on fixture_sha256")
    digest = next(iter(digests))
    if not isinstance(digest, str) or not digest:
        raise ValueError("fixture_sha256 is invalid")
    return digest


def _scorer_source(scorer: Scorer) -> tuple[str, Path]:
    module_name = scorer.__module__
    source = inspect.getsourcefile(scorer)
    if source is None:
        module = sys.modules.get(module_name)
        source = getattr(module, "__file__", None)
    if source is None:
        raise ValueError(f"cannot locate scorer module file: {module_name}")
    return module_name, Path(source).resolve()


def build_bundle(
    out_dir: Path,
    repo: Path,
    *,
    scorer: Scorer = score_registered_gate,
) -> tuple[Path, str]:
    """Re-score one completed campaign and write its immutable three-file bundle."""
    out_dir = out_dir.resolve()
    repo = repo.resolve()
    if not (repo / "scripts" / "glm52_goal.py").is_file():
        raise ValueError("repo does not contain scripts/glm52_goal.py")

    raw = (out_dir / "raw.jsonl").read_bytes()
    rows = _load_rows(raw)
    identity_raw, identity = _read_json_bytes(out_dir / "raw.jsonl.identity.json")
    terminal_memory = _read_json(out_dir / "terminal-memory.json")
    retained_raw, retained = _read_json_bytes(out_dir / "retained-manifest.json")
    frozen_scorer_path = out_dir / "retained" / "scripts" / "glm52_goal.py"
    if not frozen_scorer_path.is_file():
        raise ValueError(
            f"retained frozen scorer is missing: {frozen_scorer_path}"
        )
    frozen_scorer_sha256 = _sha256(frozen_scorer_path.read_bytes())
    preflight_raw, preflight = _read_json_bytes(out_dir / "campaign-preflight.json")
    for label, document in (
        ("raw.jsonl.identity.json", identity),
        ("terminal-memory.json", terminal_memory),
        ("retained-manifest.json", retained),
        ("campaign-preflight.json", preflight),
    ):
        if not isinstance(document, dict):
            raise ValueError(f"{label} must contain a JSON object")
    if (
        identity.get("record_type") != "matched_campaign_identity"
        or identity.get("schema_version") != 2
    ):
        raise ValueError("raw.jsonl.identity.json is not matched_campaign_identity v2")
    freeze_commit = identity.get("freeze_commit")
    if freeze_commit is None or freeze_commit != retained.get("freeze_commit"):
        raise ValueError("identity and retained manifest freeze_commit mismatch")
    receipt_sha256 = identity.get("randomness_receipt_sha256")
    if receipt_sha256 is None or receipt_sha256 != retained.get(
        "randomness_receipt_sha256"
    ):
        raise ValueError(
            "identity and retained manifest randomness_receipt_sha256 mismatch"
        )
    receipt_raw, receipt = _read_json_bytes(
        out_dir / "retained" / "randomness-receipt.json"
    )
    if not isinstance(receipt, dict):
        raise ValueError("randomness-receipt.json must contain a JSON object")
    if _sha256(receipt_raw) != receipt_sha256:
        raise ValueError("randomness receipt digest mismatch")
    for key in ("candidate_hash", "freeze_commit"):
        if key in receipt and receipt[key] != identity.get(key):
            raise ValueError(f"randomness receipt {key} disagrees with identity")

    fixture_sha256 = _fixture_digest(rows)
    shards = preflight.get("dsv4_shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("preflight dsv4_shards must not be empty")
    if any(not isinstance(shard, dict) or "sha256" not in shard for shard in shards):
        raise ValueError("every preflight dsv4_shard must carry sha256")
    shard_digests = [shard["sha256"] for shard in shards]

    scorer_module, scorer_path = _scorer_source(scorer)
    scorer_sha256 = _sha256(scorer_path.read_bytes())
    if scorer is score_registered_gate and scorer_sha256 != frozen_scorer_sha256:
        raise ValueError(
            "post-freeze scorer drift: current glm52_goal.py does not match "
            "the retained frozen scorer"
        )

    # Deliberately allow scorer exceptions to propagate: a bundle must never turn
    # malformed or incomplete campaign evidence into a narrated result.
    summary = scorer("parity", SCORER_ID, rows)
    if not isinstance(summary, dict):
        raise ValueError("scorer output must be a JSON object")
    verdict = summary.get("verdict")
    if not isinstance(verdict, str) or not verdict:
        raise ValueError("scorer output has no verdict")
    verdict_label = verdict.lower()
    if not re.fullmatch(r"[a-z0-9_-]+", verdict_label):
        raise ValueError(f"scorer verdict cannot name a bundle: {verdict!r}")

    summary_bytes = (
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "gate": "parity",
        "topic": TOPIC,
        "scorer_id": SCORER_ID,
        "candidate_hash": identity["candidate_hash"],
        "freeze_commit": freeze_commit,
        "drand_round": _drand_round(receipt),
        "randomness_receipt_sha256": receipt_sha256,
        "identity_sha256": _sha256(identity_raw),
        "retained_manifest_sha256": _sha256(retained_raw),
        "preflight_sha256": _sha256(preflight_raw),
        "glm_binary_sha256": identity["glm_binary_sha256"],
        "glm_model_sha256": identity["glm_model_sha256"],
        "glm_profile_sha256": identity["glm_profile_sha256"],
        "dsv4_binary_sha256": identity.get(
            "dsv4_binary_sha256", preflight.get("dsv4_binary_sha256")
        ),
        "dsv4_configuration_sha256": identity["dsv4_configuration_sha256"],
        "dsv4_serving_weights_manifest_sha256": identity[
            "dsv4_serving_weights_manifest_sha256"
        ],
        "dsv4_model_shards_sha256": shard_digests,
        "fixture_sha256": fixture_sha256,
        "scorer_module": scorer_module,
        "scorer_sha256": scorer_sha256,
        "frozen_scorer_sha256": frozen_scorer_sha256,
        "raw_sha256": _sha256(raw),
        "summary_sha256": _sha256(summary_bytes),
        "terminal_memory": terminal_memory,
        "source_state_dir": str(out_dir),
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    bundle = (
        repo / "results" / "glm52-gates" / f"{TOPIC}-{verdict_label}"
    ).resolve()
    if bundle.is_relative_to(out_dir):
        raise ValueError(f"bundle path must not be inside OUT_DIR: {bundle}")
    if bundle.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle.name}-", dir=bundle.parent))
    try:
        (temporary / "raw.jsonl").write_bytes(raw)
        (temporary / "summary.json").write_bytes(summary_bytes)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        if bundle.exists():
            raise FileExistsError(f"refusing to overwrite existing bundle: {bundle}")
        os.rename(temporary, bundle)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return bundle, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    bundle, verdict = build_bundle(args.out_dir, args.repo)
    print(f"bundle={bundle}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
