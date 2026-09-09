"""Prepare and select a frozen, model-free KDA replay bundle."""
import hashlib
import json
import os
import math
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
    require(any(name.endswith('.autotune.json') for name in before), 'prepared KDA tuning inputs missing')
    require(not target.exists() and source not in target.parents and target not in source.parents, 'invalid replay destination')
    target.mkdir()
    for name in before:
        if name.startswith('triton/'):
            destination = target / name
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
    manifest = {'schema_version': 1, 'qualification': 'frozen_KDA_replay_bundle_only',
        'source_inventory': {'sha256': hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()},
        'triton': file_inventory(target / 'triton'), 'autotune': tuning_inventory(target / 'triton')}
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
    require({p.name for p in root.iterdir()} == {'manifest.json', 'triton'}, 'replay bundle coverage mismatch')
    def readonly(path):
        value = path.lstat()
        require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode), 'nonregular replay artifact')
        require(value.st_mode & 0o222 == 0, 'writable replay artifact')
        if stat.S_ISDIR(value.st_mode):
            with os.scandir(path) as entries: children = sorted(entry.name for entry in entries)
            for name in children: readonly(path / name)
    readonly(root)
    manifest = strict_json(root / 'manifest.json')
    require(set(manifest) == {'schema_version', 'qualification', 'source_inventory', 'triton', 'autotune'} and
            type(manifest['schema_version']) is int and manifest['schema_version'] == 1 and
            manifest['qualification'] == 'frozen_KDA_replay_bundle_only', 'invalid replay manifest')
    verify_inventory(root / 'triton', manifest['triton'])
    require(manifest['autotune'] == tuning_inventory(root / 'triton'), 'autotune sidecar mismatch')
    return manifest



def autotune_receipt(path, root):
    """Parse historical cache inputs only; global evidence JSON stays strict."""
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'duplicate autotune JSON key')
            result[key] = value
        return result
    data = json.loads(path.read_bytes(), object_pairs_hook=pairs)
    require(isinstance(data, dict) and set(data) == {'key', 'configs_timings'}, 'invalid autotune schema')
    require(isinstance(data['key'], list) and data['key'] and all(type(v) in (str, int, bool) for v in data['key']), 'invalid tuning key')
    trials = data['configs_timings']
    require(isinstance(trials, list) and trials, 'missing tuning trials')
    seen, statuses, timings = set(), [], []
    for trial in trials:
        require(isinstance(trial, list) and len(trial) == 2, 'invalid tuning trial')
        config, timing = trial
        require(isinstance(config, dict) and set(config) == {'kwargs', 'num_warps', 'num_ctas', 'num_stages', 'maxnreg', 'pre_hook', 'ir_override'}, 'invalid tuning config')
        require(isinstance(config['kwargs'], dict) and all(isinstance(k, str) and type(v) in (int, bool) for k, v in config['kwargs'].items()), 'invalid tuning kwargs')
        require(all(type(config[k]) is int and config[k] > 0 for k in ('num_warps', 'num_ctas', 'num_stages')) and
                all(config[k] is None for k in ('maxnreg', 'pre_hook', 'ir_override')), 'unsupported tuning config')
        identity = json.dumps(config, sort_keys=True, allow_nan=False)
        require(identity not in seen, 'duplicate tuning configuration'); seen.add(identity)
        require(isinstance(timing, list) and len(timing) == 3 and all(type(t) in (int, float) for t in timing), 'invalid tuning timing')
        finite = all(math.isfinite(t) and t > 0 for t in timing)
        failed = all(t == float('inf') for t in timing)
        require(finite or failed, 'invalid tuning failure sentinel or timing')
        statuses.append('finite' if finite else 'failed_infinite_sentinel'); timings.append(timing)
    winner = min(range(len(trials)), key=lambda i: timings[i])
    require(statuses[winner] == 'finite', 'nonfinite tuning winner')
    return {'path': str(path.relative_to(root)), 'sha256': sha256_file(path), 'key': data['key'],
            'winner_index': winner, 'winner_config': trials[winner][0], 'trial_statuses': statuses,
            'failed_trial_indices': [i for i, status in enumerate(statuses) if status != 'finite']}


def tuning_inventory(root):
    paths = sorted(root.rglob('*.autotune.json'))
    require(paths, 'missing autotune inputs')
    return [autotune_receipt(path, root) for path in paths]


def reject_retuning(*, enabled=False):
    """Evidence-only startup selection; disabled mode performs no imports."""
    require(type(enabled) is bool, 'retuning selection must be boolean')
    if not enabled: return
    from triton.runtime.autotuner import Autotuner
    def reject(*args, **kwargs):
        raise ValueError('sealed KDA retuning is unavailable')
    Autotuner._bench = reject


def activate(root, binding):
    """Select before importing vLLM or constructing its autotuners."""
    from glm53_runtime_jit import activate_triton, sealed_cache_class
    import triton.knobs as knobs
    root = Path(root)
    manifest = verify_bundle(root, binding)
    cache = sealed_cache_class(root / 'triton', manifest['triton'])
    knobs.autotuning.cache = True
    reject_retuning(enabled=True)
    activate_triton(cache, enabled=True)
    return {'selection': 'sealed_KDA_replay', 'bundle': binding,
            'triton_cache_root': str(root / 'triton'), 'retuning': 'rejected', 'disk_autotune_cache': True}
