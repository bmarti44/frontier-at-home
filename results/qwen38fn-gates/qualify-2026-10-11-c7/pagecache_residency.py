"""Page-cache residency per file or directory via mmap + mincore; reads no data.

Usage: python3 -I pagecache_residency.py PATH...  (diagnostic for RESULT.md)
"""
import ctypes, os, sys
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
PAGE = os.sysconf("SC_PAGE_SIZE")
PROT_READ, MAP_SHARED, FAILED = 1, 1, ctypes.c_void_p(-1).value
def resident(path):
    size = os.path.getsize(path)
    if size == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
        if addr in (None, FAILED):
            return 0
        try:
            pages = (size + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * pages)()
            if libc.mincore(addr, size, vec) != 0:
                return 0
            return sum(v & 1 for v in vec) * PAGE
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)
for root in sys.argv[1:]:
    if os.path.isfile(root):
        print(f"{resident(root) / 2**30:8.2f} GiB  {root}")
        continue
    total = 0
    for d, _, files in os.walk(root):
        for n in files:
            p = os.path.join(d, n)
            if os.path.isfile(p) and not os.path.islink(p):
                try:
                    total += resident(p)
                except OSError:
                    pass
    print(f"{total / 2**30:8.2f} GiB  {root}")
