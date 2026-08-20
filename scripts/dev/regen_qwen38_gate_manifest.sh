#!/usr/bin/env bash
# Regenerate the Track C 1M evidence inventory after adding gate evidence.
set -Eeuo pipefail

readonly REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
readonly EVIDENCE_REL=results/qwen38-gates/trackc-1m-np4-2026-08-19
readonly EVIDENCE_DIR=$REPO/$EVIDENCE_REL
readonly MANIFEST=$EVIDENCE_DIR/manifest.json

/usr/bin/python3 - "$EVIDENCE_DIR" "$EVIDENCE_REL" "$MANIFEST" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

evidence_dir = Path(sys.argv[1])
evidence_rel = sys.argv[2]
manifest_path = Path(sys.argv[3])

files = []
for path in sorted(evidence_dir.iterdir(), key=lambda item: item.name):
    if not path.is_file() or path == manifest_path:
        continue
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            byte_count += len(chunk)
    files.append({
        "path": path.name,
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
    })

manifest = {
    "schema_version": 1,
    "evidence_directory": evidence_rel,
    "scope": (
        "Every regular file directly in the evidence directory except "
        "manifest.json itself, whose self-hash cannot be embedded in its own bytes."
    ),
    "regeneration": "scripts/dev/regen_qwen38_gate_manifest.sh",
    "pending_evidence_note": (
        "Gate-4 full-load measurement is running separately. Add its results here "
        "and rerun the regeneration script so every added evidence file is covered."
    ),
    "files": files,
}

temporary = manifest_path.with_suffix(".json.tmp")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(manifest, stream, indent=2, sort_keys=True)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
os.replace(temporary, manifest_path)
PY
