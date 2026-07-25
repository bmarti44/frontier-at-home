#!/usr/bin/env python3
"""O_DIRECT random-read QD sweep (fio substitute). Read-only."""
import os, mmap, threading, time, random, sys
path = sys.argv[1]
fsize = os.path.getsize(path)
ALIGN = 4096
def worker(fd, bs, stop, counter, idx):
    buf = mmap.mmap(-1, bs)
    mv = memoryview(buf)
    rng = random.Random(1234 + idx)
    n = 0
    while not stop.is_set():
        off = (rng.randrange(0, fsize - bs) // ALIGN) * ALIGN
        got = os.preadv(fd, [mv], off)
        n += got
    counter[idx] = n
for bs_label, bs in (("9.7MiB", 10171392), ("1MiB", 1048576)):
    for qd in (1, 4, 8, 16, 32):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        stop = threading.Event()
        counter = [0] * qd
        ts = [threading.Thread(target=worker, args=(fd, bs, stop, counter, i)) for i in range(qd)]
        t0 = time.time()
        for t in ts: t.start()
        time.sleep(8)
        stop.set()
        for t in ts: t.join()
        dt = time.time() - t0
        os.close(fd)
        print(f"bs={bs_label} qd={qd:2d}: {sum(counter)/dt/1e9:.2f} GB/s")
