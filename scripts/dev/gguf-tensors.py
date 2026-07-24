#!/usr/bin/env python3
"""Minimal GGUF header parser: list tensor names (no data read)."""
import struct, sys, re

path = sys.argv[1]
f = open(path, "rb")

def u32(): return struct.unpack("<I", f.read(4))[0]
def u64(): return struct.unpack("<Q", f.read(8))[0]
def s():
    n = u64()
    return f.read(n).decode("utf-8", "replace")

magic = f.read(4)
assert magic == b"GGUF", magic
version = u32()
n_tensors = u64()
n_kv = u64()

SCALAR = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}

def skip_value(t):
    if t == 8:      # string
        f.seek(u64(), 1)
    elif t == 9:    # array
        et = u32(); n = u64()
        if et == 8:
            for _ in range(n): f.seek(u64(), 1)
        elif et == 9:
            for _ in range(n): skip_value(9)
        else:
            f.seek(SCALAR[et] * n, 1)
    else:
        f.seek(SCALAR[t], 1)

for _ in range(n_kv):
    key = s(); t = u32(); skip_value(t)

names = []
for _ in range(n_tensors):
    name = s()
    n_dims = u32()
    f.seek(8 * n_dims, 1)  # dims
    f.seek(4 + 8, 1)       # dtype + offset
    names.append(name)

print(f"version={version} n_tensors={n_tensors}")
blk = sorted({int(m.group(1)) for n in names if (m := re.match(r"blk\.(\d+)\.", n))})
print(f"block range: {blk[0]}..{blk[-1]} ({len(blk)} blocks)")
special = [n for n in names if re.search(r"nextn|mtp|eh_proj|embed_tokens_mtp|shared_head", n, re.I)]
print("MTP-related tensors:", special if special else "NONE")
top = [n for n in names if not n.startswith("blk.")]
print("non-blk tensors:", top)
