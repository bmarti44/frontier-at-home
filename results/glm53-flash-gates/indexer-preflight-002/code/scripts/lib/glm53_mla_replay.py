"""Prepare and select a frozen, model-free MLA replay bundle."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from glm53_contract import sha256_file, strict_json, verify_inventory


def require(value, message):
    if not value: raise ValueError(message)


def file_inventory(root):
    return {'schema_version': 1, 'files': [
        {'path': str(p.relative_to(root)), 'size_bytes': p.stat().st_size, 'sha256': sha256_file(p)}
        for p in sorted(root.rglob('*')) if p.is_file()]}


def prepare_bundle(source, target, record):
    """Relocate only group metadata; preserve every compiled artifact byte."""
    source, target = Path(source).absolute(), Path(target).absolute()
    require(record['root'] == str(source), 'preparation root mismatch')
    require(all(row['type'] in ('file', 'directory') for row in record['entries']), 'invalid preparation file type')
    original = {'schema_version': 1, 'files': [
        {key: row[key] for key in ('path', 'size_bytes', 'sha256')}
        for row in record['entries'] if row['type'] == 'file']}
    before = verify_inventory(source, original)
    libraries = [name for name in before if name.endswith('/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so')]
    require(len(libraries) == 1 and any(name.startswith('triton/') for name in before), 'prepared MLA kernels missing')
    require(not target.exists() and source not in target.parents and target not in source.parents, 'invalid replay destination')
    target.mkdir()
    for name in before:
        if name.startswith('triton/'):
            destination = target / name
        elif name == libraries[0]:
            destination = target / 'flashinfer/sparse_mla_sm120/sparse_mla_sm120.so'
        else:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
        require(sha256_file(destination) == sha256_file(source / name), 'compiled artifact copy mismatch')
    for group in (target / 'triton').rglob('__grp__*.json'):
        data = strict_json(group)
        require(set(data) == {'child_paths'} and isinstance(data['child_paths'], dict) and data['child_paths'], 'invalid Triton group')
        children = {}
        original_group = source / group.relative_to(target)
        for name, path in data['child_paths'].items():
            require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..') and
                    path == str(original_group.parent / name) and
                    str((original_group.parent / name).relative_to(source)) in before, 'unbound Triton group child')
            children[name] = str(group.parent / name)
        group.write_text(json.dumps({'child_paths': children}, indent=2) + '\n')
    require(verify_inventory(source, original) == before, 'preparation changed during relocation')
    manifest = {'schema_version': 1, 'qualification': 'frozen_MLA_replay_bundle_only',
        'source_inventory': {'sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()},
        'triton': file_inventory(target / 'triton'), 'flashinfer': file_inventory(target / 'flashinfer')}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for path in sorted(target.rglob('*'), reverse=True): path.chmod(0o555 if path.is_dir() else 0o444)
    target.chmod(0o555)
    binding = {'root': str(target), 'sha256': sha256_file(target / 'manifest.json')}
    verify_bundle(target, binding)
    return binding


def verify_bundle(root, binding):
    root = Path(root).absolute()
    require(set(binding) == {'root', 'sha256'} and binding['root'] == str(root), 'replay binding path mismatch')
    require(sha256_file(root / 'manifest.json') == binding['sha256'], 'replay manifest digest mismatch')
    require({p.name for p in root.iterdir()} == {'manifest.json', 'triton', 'flashinfer'}, 'replay bundle coverage mismatch')
    def readonly(path):
        value = path.lstat()
        require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode), 'nonregular replay artifact')
        require(value.st_mode & 0o222 == 0, 'writable replay artifact')
        if stat.S_ISDIR(value.st_mode):
            with os.scandir(path) as entries: children = sorted(entry.name for entry in entries)
            for name in children: readonly(path / name)
    readonly(root)
    manifest = strict_json(root / 'manifest.json')
    require(set(manifest) == {'schema_version', 'qualification', 'source_inventory', 'triton', 'flashinfer'} and
            type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and
            manifest['qualification'] == 'frozen_MLA_replay_bundle_only', 'invalid replay manifest')
    for name in ('triton', 'flashinfer'): verify_inventory(root / name, manifest[name])
    return manifest


def activate(root, binding):
    """Explicit startup-only selection; caller writes the returned receipt."""
    from glm53_runtime_jit import activate_flashinfer, activate_triton, sealed_cache_class
    root = Path(root)
    manifest = verify_bundle(root, binding)
    cache = sealed_cache_class(root / 'triton', manifest['triton'])
    flashinfer = activate_flashinfer(root / 'flashinfer', manifest['flashinfer'],
        {'sparse_mla_sm120': 'sparse_mla_sm120/sparse_mla_sm120.so'}, enabled=True)
    activate_triton(cache, enabled=True)
    return {'selection': 'sealed_MLA_replay', 'bundle': binding,
            'triton_cache_root': str(root / 'triton'), 'flashinfer': flashinfer}
