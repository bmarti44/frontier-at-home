#!/usr/bin/env python3
"""Print absolute file offset + size for GGUF tensors matching a substring.
Usage: gguf_tensor_offset.py FILE SUBSTR [MAXN]"""
import sys, struct

f = open(sys.argv[1], 'rb')
sub, maxn = sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5
def rd(n):
    b = f.read(n)
    assert len(b) == n
    return b
def u32(): return struct.unpack('<I', rd(4))[0]
def u64(): return struct.unpack('<Q', rd(8))[0]
def s(): return rd(u64()).decode('utf-8', 'replace')
SZ = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def skip_val(t):
    if t == 8: rd(u64())
    elif t == 9:
        it, n = u32(), u64()
        for _ in range(n): skip_val(it)
    else: rd(SZ[t])
assert rd(4) == b'GGUF'
ver, n_t, n_kv = u32(), u64(), u64()
align = 32
for _ in range(n_kv):
    k = s(); t = u32()
    if k == 'general.alignment':
        align = struct.unpack('<I', rd(4))[0] if t == 4 else (skip_val(t) or align)
    else:
        skip_val(t)
ents = []
for _ in range(n_t):
    name = s(); nd = u32()
    dims = [u64() for _ in range(nd)]
    dt = u32(); off = u64()
    ents.append((name, dims, dt, off))
end = f.tell()
data_start = (end + align - 1) // align * align
ents.sort(key=lambda e: e[3])
hits = 0
for i, (name, dims, dt, off) in enumerate(ents):
    if sub in name and hits < maxn:
        nxt = ents[i+1][3] if i + 1 < len(ents) else None
        size = (nxt - off) if nxt else -1
        print(f"{name} dims={dims} dtype={dt} abs_off={data_start + off} size={size}")
        hits += 1
