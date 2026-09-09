#!/usr/bin/env python3
"""Read pinned safetensors headers and calculate the dense overlay, without weights.

This is a metadata-only audit, not an artifact, fidelity or memory-fit gate.
Headers are bounded ranged reads; complete indexes are independently hashed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/lib"))
from glm53_contract import sha256_file, strict_json


def read_url(url, size, byte_range=None, total=None):
    headers = {"User-Agent": "frontier-at-home-glm53-layout-audit/1"}
    if byte_range is not None:
        first, last = byte_range
        headers["Range"] = f"bytes={first}-{last}"
        url += f"?layout_range={first}-{last}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
        if byte_range is not None:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {first}-{last}/{total}":
                raise ValueError("server did not return the exact requested byte range")
        elif response.status != 200:
            raise ValueError("full metadata download did not return 200")
        data = response.read(size + 1)
        if len(data) != size:
            raise ValueError("metadata byte count mismatch")
        return data


def shard_header(url, entry, destination):
    size = entry["size"]
    prefix = read_url(url, 8, (0, 7), size)
    length = struct.unpack("<Q", prefix)[0]
    if not 2 <= length <= min(16 * 1024 * 1024, size - 8):
        raise ValueError("invalid or excessive safetensors header size")
    blob = read_url(url, length, (8, 7 + length), size)
    destination.write_bytes(blob)
    header = strict_json(destination)
    intervals = []
    widths = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
              "F8_E4M3FN": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
              "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(tensor, dict) or set(tensor) != {"dtype", "shape", "data_offsets"}:
            raise ValueError("unexpected tensor header schema")
        shape, offsets = tensor["shape"], tensor["data_offsets"]
        if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("invalid tensor shape")
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(x) is not int for x in offsets):
            raise ValueError("invalid tensor offsets")
        count = widths[tensor["dtype"]]
        for dimension in shape:
            count *= dimension
        first, last = offsets
        if not 0 <= first <= last <= size - 8 - length or last - first != count:
            raise ValueError("tensor offsets disagree with shape or shard size")
        intervals.append((first, last))
    end = 0
    for first, last in sorted(intervals):
        if first != end:
            raise ValueError("tensor header has a gap or overlapping data")
        end = last
    if end != size - 8 - length:
        raise ValueError("tensor header does not cover complete shard")
    return length, header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    lock_path = ROOT / "configs/build-manifests/glm53-flash-sources.json"
    lock = strict_json(lock_path)
    overlay = args.prepared.resolve() / "vllm-exl3/tools/dense_overlay.py"
    pin = next(row["sha256"] for row in lock["sources"]["vllm-exl3"]["files"] if row["path"] == "tools/dense_overlay.py")
    if sha256_file(overlay) != pin:
        raise ValueError("overlay source identity mismatch")
    spec = importlib.util.spec_from_file_location("audited_dense_overlay", overlay)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = {"schema_version": 1, "qualification": "metadata_only", "model_loaded": False,
                "source_lock_sha256": sha256_file(lock_path), "scorer_sha256": sha256_file(Path(__file__)),
                "overlay_sha256": pin, "artifacts": lock["artifacts"], "start_unix": time.time()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    packs = {}
    with (output / "raw.jsonl").open("w") as raw:
        try:
            for label, key in (("k2", "exl3_k2"), ("dense", "dense_overlay_source")):
                artifact = lock["artifacts"][key]
                api_path = args.metadata / (label + "-api.json")
                api = strict_json(api_path)
                if api["sha"] != artifact["revision"] or api["id"] != artifact["repo"]:
                    raise ValueError("API metadata identity mismatch")
                pack_dir = output / label
                pack_dir.mkdir()
                (pack_dir / "api.json").write_bytes(api_path.read_bytes())
                base = f"https://huggingface.co/{artifact['repo']}/resolve/{artifact['revision']}/"
                entries = {row["rfilename"]: row for row in api["siblings"]}
                index_name = "model.safetensors.index.json"
                entry = entries[index_name]
                if entry["size"] > 32 * 1024 * 1024:
                    raise ValueError("index exceeds metadata budget")
                index_path = pack_dir / index_name
                index_path.write_bytes(read_url(base + index_name, entry["size"]))
                if sha256_file(index_path) != entry["lfs"]["sha256"]:
                    raise ValueError("full index hash mismatch")
                index = strict_json(index_path)
                names = sorted(set(index["weight_map"].values()))
                if any(not re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", name) for name in names):
                    raise ValueError("unexpected shard path")
                headers = {}
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs = {pool.submit(shard_header, base + name, entries[name], pack_dir / (name + ".header.json")): name for name in names}
                    for job in as_completed(jobs):
                        name = jobs[job]
                        headers[name] = job.result()
                        raw.write(json.dumps({"time_unix": time.time(), "pack": label, "file": name,
                                              "header_bytes": headers[name][0], "header_sha256": sha256_file(pack_dir / (name + ".header.json")),
                                              "shard_bytes": entries[name]["size"], "shard_sha256_expected": entries[name]["lfs"]["sha256"]}) + "\n")
                        raw.flush()
                for name, filename in index["weight_map"].items():
                    if name not in headers[filename][1]:
                        raise ValueError("index references absent tensor")
                packs[label] = {"index": index, "headers": headers}
            options = SimpleNamespace(root="model.language_model.", skip_layers={45}, draft_layers=set(),
                                      prefix_rewrite=("model.language_model.", "language_model.model."), draft_prefix_rewrite=None)
            plan, keys, problems = module.build_plan(options, packs["k2"]["index"], packs["k2"]["headers"],
                                                    {"weight_map": packs["dense"]["index"]["weight_map"], "headers": packs["dense"]["headers"]})
            if problems or not plan:
                raise ValueError("overlay plan invalid: " + repr(problems))
            outputs = module.plan_outputs(plan)
            bytes_out = sum(module.nbytes({"dtype": dtype, "shape": shape}) for _, dtype, shape, _, _ in outputs)
            replaced = sum(module.nbytes(packs["k2"]["headers"][packs["k2"]["index"]["weight_map"][e["name"]]][1][e["name"]]) for e in plan)
            (output / "overlay-plan.json").write_text(json.dumps({"plan": plan, "fork_keys": keys}, indent=2) + "\n")
            summary = {"verdict": "PASS", "qualification": "metadata_only", "model_loaded": False,
                       "replaced_tensors": len(plan), "overlay_tensors": len(outputs), "module_keys": len(keys),
                       "overlay_data_bytes": bytes_out, "replaced_bf16_bytes": replaced,
                       "resident_reduction_bytes_estimate": replaced - bytes_out,
                       "memory_fit": "not measured", "weights_downloaded": False}
        except Exception as exc:
            raw.write(json.dumps({"time_unix": time.time(), "failure": repr(exc)}) + "\n")
            summary = {"verdict": "FAIL", "qualification": "metadata_only", "failure": repr(exc)}
    summary["raw_sha256"] = sha256_file(output / "raw.jsonl")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
