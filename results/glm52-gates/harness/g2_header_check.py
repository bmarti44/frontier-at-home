#!/usr/bin/env python3
"""G2 header check: GGUF magic/version, general.architecture, tensor count,
block range, MTP/nextn tensor presence. Derived from scripts/dev/gguf-tensors.py
(same header walk), extended to decode string KVs so the gate can assert arch."""
import struct, sys, re, json

path = sys.argv[1]
f = open(path, "rb")

def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s():
    n = u64()
    return f.read(n).decode("utf-8", "replace")

magic = f.read(4)
version = u32()
n_tensors = u64()
n_kv = u64()

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

kvs = {}
for _ in range(n_kv):
    key = s(); t = u32()
    v = read_value(t)
    if v is not None and len(kvs) < 64:
        kvs[key] = v

names = []
for _ in range(n_tensors):
    name = s()
    n_dims = u32()
    f.seek(8 * n_dims, 1)
    f.seek(4 + 8, 1)
    names.append(name)

blk = sorted({int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.", n))})
special = sorted({n for n in names if re.search(r"nextn|mtp|eh_proj|shared_head", n, re.I)})
out = {
    "magic_ok": magic == b"GGUF",
    "version": version,
    "n_tensors": n_tensors,
    "n_kv": n_kv,
    "architecture": kvs.get("general.architecture"),
    "name": kvs.get("general.name"),
    "block_range": [blk[0], blk[-1]] if blk else None,
    "n_blocks": len(blk),
    "mtp_nextn_tensor_count": len(special),
    "mtp_nextn_sample": special[:8],
}
print(json.dumps(out, indent=2))
