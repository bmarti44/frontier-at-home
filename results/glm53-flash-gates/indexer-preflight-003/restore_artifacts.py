"""Restore captured bytes from this committed archive into a fresh directory."""
import argparse
import hashlib
import json
import lzma
from pathlib import Path, PurePosixPath
import stat


def relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or str(path) != name or name == '.':
        raise ValueError(f'invalid archive path: {name!r}')
    return path


def regular(root, name):
    path = root
    for part in relative(name).parts:
        path = path / part
        if path.is_symlink():
            raise ValueError(f'symlink in archive path: {name}')
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f'nonregular archive file: {name}')
    return path


def verify(path, record):
    def identity(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
    before = path.stat()
    with path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    after = path.stat()
    if identity(before) != identity(after) or after.st_size != record['size_bytes'] or digest != record['sha256']:
        raise ValueError(f'archive size, identity or digest mismatch: {path}')


def restore(root, destination):
    root = Path(root).resolve(strict=True)
    manifest_path = regular(root, 'archive.json')
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    jobs = []
    for name, record in manifest['copied_files'].items():
        jobs.append((name, name, record, record, False))
    for name, record in manifest['compressed_files'].items():
        if record['encoding'] != 'xz':
            raise ValueError('unsupported compression')
        archive_record = record['archive']
        jobs.append((name, record['file'], record, archive_record, True))
    for record in manifest['bundles'].values():
        jobs.append((record['file'], record['file'], record, record, False))
    outputs = [str(relative(job[0])) for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError('duplicate output path')
    for _, name, _, record, _ in jobs:
        verify(regular(root, name), record)
    destination = Path(destination)
    destination.mkdir(exist_ok=False)
    for name, archived, original_record, archive_record, compressed in jobs:
        source_path = regular(root, archived)
        target = destination / relative(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        opener = lzma.open if compressed else open
        size = 0
        with opener(source_path, 'rb') as source, target.open('xb') as sink:
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > original_record['size_bytes']:
                    raise ValueError(f'expanded size exceeds inventory: {name}')
                sink.write(chunk)
        verify(target, original_record)
        verify(regular(root, archived), archive_record)
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError('archive manifest changed')
    return {'verdict': 'PASS', 'restored_files': len(jobs), 'destination': str(destination)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(restore(Path(__file__).parent, args.destination)))
