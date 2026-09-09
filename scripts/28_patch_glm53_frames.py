#!/usr/bin/env python3
"""Apply the reviewed sixteen-frame cap to the pinned GLM processor only."""
import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import strict_json

RELATIVE = "vllm/transformers_utils/processors/glm5next.py"
OLD = 'max_frame_count=kwargs.get("max_frames") or self.max_frame_count_dynamic,'
NEW = 'max_frame_count=min(16, kwargs.get("max_frames") or self.max_frame_count_dynamic),'


def patch(source):
    lock = strict_json(ROOT / "configs/build-manifests/glm53-flash-sources.json")
    expected = next(row["sha256"] for row in lock["sources"]["vllm"]["files"]
                    if row["path"] == RELATIVE)
    path = source / RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("processor must match the pristine locked digest; patch exactly once")
    text = raw.decode()
    if text.count(OLD) != 1:
        raise ValueError("frame-sampler anchor count is not one")
    path.write_text(text.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    patch(args.source)
