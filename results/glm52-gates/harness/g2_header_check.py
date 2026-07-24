#!/usr/bin/env python3
"""G2 header check: GGUF magic/version, general.architecture, tensor count,
block range, MTP/nextn presence. Derived from scripts/dev/gguf-tensors.py.

Report mode:  g2_header_check.py FILE
Assert mode:  g2_header_check.py FILE --assert ARCH N_TENSORS BLK_LO BLK_HI
  exits 0 only if: magic ok, version 3, architecture == ARCH, tensor count ==
  N_TENSORS, block range == [BLK_LO, BLK_HI], no duplicate KV keys, every
  tensor offset is non-negative and the tensor-data region start lies within
  the file, and the whole header walk completes without hitting EOF.
"""
import struct, sys, re, json, os

path = sys.argv[1]
do_assert = len(sys.argv) > 2 and sys.argv[2] == "--assert"
exp = sys.argv[3:7] if do_assert else None
fsize = os.path.getsize(path)
f = open(path, "rb")
errors = []

def rd(n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"short read at offset {f.tell()}")
    return b

def u32(): return struct.unpack("<I", rd(4))[0]
def u64(): return struct.unpack("<Q", rd(8))[0]
def s():
    n = u64()
    if n > 1 << 20: raise ValueError(f"implausible string length {n}")
    return rd(n).decode("utf-8", "replace")

SCALAR = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def read_value(t):
    if t == 8:
        return s()
    if t == 9:
        et = u32(); n = u64()
        if et == 8:
            for _ in range(n): f.seek(u64(), 1)
        elif et == 9:
            for _ in range(n): read_value(9)
        else:
            f.seek(SCALAR[et] * n, 1)
        return None
    f.seek(SCALAR[t], 1)
    return None

try:
    magic = rd(4)
    version = u32()
    n_tensors = u64()
    n_kv = u64()
    if magic != b"GGUF": errors.append(f"bad magic {magic!r}")
    if version != 3: errors.append(f"unexpected gguf version {version}")

    kvs, seen, dups = {}, set(), []
    for _ in range(n_kv):
        key = s(); t = u32()
        if key in seen: dups.append(key)
        seen.add(key)
        v = read_value(t)
        if v is not None and len(kvs) < 64:
            kvs[key] = v
    if dups: errors.append(f"duplicate KV keys: {dups[:5]}")

    names, offs = [], []
    for _ in range(n_tensors):
        name = s()
        n_dims = u32()
        if n_dims > 8: raise ValueError(f"implausible n_dims {n_dims} for {name}")
        f.seek(8 * n_dims, 1)
        f.seek(4, 1)          # dtype
        offs.append(u64())    # offset within tensor-data region
        names.append(name)
    header_end = f.tell()
    if header_end >= fsize: errors.append("header extends past EOF")
    bad_offs = [o for o in offs if o < 0 or header_end + o > fsize]
    if bad_offs: errors.append(f"{len(bad_offs)} tensor offsets outside file bounds")
except (EOFError, ValueError, struct.error) as e:
    errors.append(f"header walk failed: {e}")
    names, kvs, version, n_tensors, n_kv = [], {}, None, None, None

blk = sorted({int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.", n))})
special = sorted({n for n in names if re.search(r"nextn|mtp|eh_proj|shared_head", n, re.I)})
out = {
    "file_bytes": fsize,
    "magic_ok": not any("magic" in e for e in errors),
    "version": version,
    "n_tensors": n_tensors,
    "n_kv": n_kv,
    "architecture": kvs.get("general.architecture"),
    "name": kvs.get("general.name"),
    "block_range": [blk[0], blk[-1]] if blk else None,
    "n_blocks": len(blk),
    "mtp_nextn_tensor_count": len(special),
    "mtp_nextn_sample": special[:8],
    "errors": errors,
}

if do_assert:
    arch, nt, lo, hi = exp[0], int(exp[1]), int(exp[2]), int(exp[3])
    if out["architecture"] != arch: errors.append(f"arch {out['architecture']} != {arch}")
    if out["n_tensors"] != nt: errors.append(f"n_tensors {out['n_tensors']} != {nt}")
    if out["block_range"] != [lo, hi]: errors.append(f"block_range {out['block_range']} != [{lo},{hi}]")
    out["assert_mode"] = {"expected": {"arch": arch, "n_tensors": nt, "blocks": [lo, hi]}, "errors": errors}

print(json.dumps(out, indent=2))
sys.exit(1 if errors else 0)
