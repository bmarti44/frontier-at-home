"""Resolve and hash malloc_trim's native provider without invoking malloc_trim."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


def describe():
    class DlInfo(ctypes.Structure):
        _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]

    process = ctypes.CDLL(None)
    dladdr = process.dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    address = ctypes.cast(process.malloc_trim, ctypes.c_void_p)
    if dladdr(address, ctypes.byref(info)) != 1 or not info.filename:
        raise RuntimeError('malloc_trim provider could not be resolved')
    provider = Path(info.filename.decode()).resolve(strict=True)
    stat = provider.stat()
    with provider.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'scope': 'Native provider identity only; malloc_trim not invoked',
            'symbol': 'malloc_trim', 'python': str(Path(sys.executable).resolve()),
            'provider': {'path': str(provider), 'size_bytes': stat.st_size,
                         'sha256': digest, 'device': stat.st_dev, 'inode': stat.st_ino}}


def verify_mapping(pid, provider):
    path = Path(provider['path']).resolve(strict=True)
    stat = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (stat.st_size, stat.st_dev, stat.st_ino, digest) != (
            provider['size_bytes'], provider['device'], provider['inode'], provider['sha256']):
        raise ValueError('native provider file changed')
    matches = []
    for line in (Path('/proc') / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5] == str(path):
            device = os.makedev(*(int(x, 16) for x in fields[3].split(':')))
            if device != stat.st_dev or int(fields[4]) != stat.st_ino:
                raise ValueError('native provider mapped identity mismatch')
            matches.append(line)
    if not matches:
        raise ValueError('frozen native provider is not mapped')
    return {'pid': pid, 'provider': provider, 'maps': matches}


if __name__ == '__main__':
    print(json.dumps(describe(), allow_nan=False))
