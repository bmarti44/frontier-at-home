"""Explicit passive host diagnostic; no imports into a serving process."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'swap-observation-001/observe.py'
spec = importlib.util.spec_from_file_location('closed_global_observer', PARENT)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
ROOT = Path('/sys/fs/cgroup')
GROUPS = ['.', 'init.scope', 'system.slice', 'user.slice',
          'system.slice/docker.service', 'system.slice/snapd.service']


def counters(raw):
    result = {}
    for line in raw.splitlines():
        key, value = line.split()
        if key in result or not value.isascii() or not value.isdecimal():
            raise ValueError('duplicate or malformed memory.stat counter')
        result[key] = int(value)
    return {key: result[key] for key in ['pswpin', 'pswpout']}


def snapshot():
    left = previous.swap_counters()
    groups = {}
    for name in GROUPS:
        path = ROOT / name
        before = path.stat()
        raw = (path / 'memory.stat').read_text()
        after = path.stat()
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError('cgroup identity changed during read')
        groups[name] = {'device': after.st_dev, 'inode': after.st_ino,
                        'raw': raw, 'counters': counters(raw)}
    return {'before': left, 'groups': groups, 'after': previous.swap_counters(),
            'observer': previous.stat_fields(Path('/proc/self/stat').read_text()),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def score(rows, seconds):
    if len(rows) != seconds + 2 or rows[-1].get('kind') != 'end':
        raise ValueError('missing samples or terminal record')
    samples = rows[:-1]
    if any(type(row.get('index')) is not int for row in samples) or [
            row['index'] for row in samples] != list(range(seconds + 1)):
        raise ValueError('missing or duplicate sample index')
    first = samples[0]
    globals_ = []
    prior_groups = None
    for row in samples:
        if row.get('kind') != 'sample' or set(row['groups']) != set(GROUPS):
            raise ValueError('malformed sample coverage')
        if type(row['boot_id']) is not str:
            raise ValueError('invalid boot identity')
        boot = uuid.UUID(row['boot_id'])
        if not boot.int or str(boot) != row['boot_id'] or any(
                type(row['observer'][k]) is not int or row['observer'][k] <= 0
                for k in ['pid', 'start_ticks']):
            raise ValueError('invalid observer identity')
        if row['boot_id'] != first['boot_id'] or any(
                row['observer'][k] != first['observer'][k] for k in ['pid', 'start_ticks']):
            raise ValueError('observer or boot identity changed')
        for name, group in row['groups'].items():
            if any(type(group[k]) is not int or group[k] <= 0 for k in ['device', 'inode']):
                raise ValueError('invalid cgroup identity')
            if group['counters'] != counters(group['raw']):
                raise ValueError('reported cgroup counters differ from raw')
            if any(group[k] != first['groups'][name][k] for k in ['device', 'inode']):
                raise ValueError('cgroup recreated')
            if prior_groups and any(group['counters'][k] < prior_groups[name]['counters'][k]
                                    for k in ['pswpin', 'pswpout']):
                raise ValueError('cgroup counter reset')
        prior_groups = row['groups']
        globals_.extend([row['before'], row['after']])
    for value in globals_:
        if type(value['time_unix']) not in (int, float) or not math.isfinite(value['time_unix']):
            raise ValueError('invalid timestamp')
        for key in ['observed_ns', 'pswpin', 'pswpout', 'SwapTotal_kib', 'SwapFree_kib', 'MemAvailable_kib']:
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError('invalid global counter')
        if value['SwapFree_kib'] > value['SwapTotal_kib']:
            raise ValueError('invalid swap capacity')
    for a, b in zip(globals_, globals_[1:]):
        if a['observed_ns'] >= b['observed_ns'] or any(a[k] > b[k] for k in ['pswpin', 'pswpout']):
            raise ValueError('global time or counter reset')
    if type(rows[-1].get('observed_ns')) is not int or rows[-1]['observed_ns'] < globals_[-1]['observed_ns']:
        raise ValueError('invalid terminal timestamp')
    elapsed = (globals_[-1]['observed_ns'] - globals_[0]['observed_ns']) / 1e9
    if elapsed < seconds or elapsed > seconds + 10:
        raise ValueError('capture duration outside fixed bounds')
    initial = globals_[0]
    quiet = all(all(value[k] == initial[k] for k in ['pswpin', 'pswpout']) and
                value['SwapTotal_kib'] - value['SwapFree_kib'] ==
                initial['SwapTotal_kib'] - initial['SwapFree_kib'] for value in globals_)
    return {'verdict': 'NO_RESULT', 'observation_integrity': 'PASS',
            'scope': 'Passive host accounting only; no causal or model qualification',
            'formula': 'All ordered raw samples and identities valid; quiet requires every global swap counter and used-swap value equal to initial observation',
            'samples': len(samples), 'elapsed_seconds': elapsed, 'quiet_interval': quiet,
            'global_delta': {k: globals_[-1][k] - initial[k] for k in ['pswpin', 'pswpout']},
            'cgroup_deltas_not_additive': {name: {k: samples[-1]['groups'][name]['counters'][k] -
                first['groups'][name]['counters'][k] for k in ['pswpin', 'pswpout']} for name in GROUPS}}


def run(out, seconds):
    out.mkdir(parents=True, exist_ok=False)
    config = {'seconds': seconds, 'groups': GROUPS, 'interval_seconds': 1}
    files = [Path(__file__).resolve(), PARENT, HERE / 'PROTOCOL.md', Path(sys.executable).resolve()]
    manifest = {'scope': 'Passive observation; no model/randomized arm', 'configuration': config,
                'configuration_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'kernel': os.uname().release, 'files': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    started = None
    with (out / 'raw.jsonl').open('x') as stream:
        def write(row):
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        try:
            for index in range(seconds + 1):
                delay = 0 if started is None else started + index - time.monotonic()
                if delay > 0: time.sleep(delay)
                data = snapshot()
                if started is None: started = data['before']['observed_ns'] / 1e9
                write({'kind': 'sample', 'index': index, **data})
        except BaseException as error:
            write({'kind': 'error', 'error': repr(error)})
        finally:
            write({'kind': 'end', 'observed_ns': time.monotonic_ns()})
    rows = [json.loads(line) for line in (out / 'raw.jsonl').read_text().splitlines()]
    try:
        result = score(rows, seconds)
    except (ValueError, KeyError, TypeError) as error:
        result = {'verdict': 'FAIL', 'observation_integrity': 'FAIL',
                  'scope': 'Incomplete or malformed passive capture', 'error': repr(error)}
    result['raw_sha256'] = hashlib.sha256((out / 'raw.jsonl').read_bytes()).hexdigest()
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    return result['observation_integrity'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900: parser.error('duration must be 1..900 seconds')
    sys.exit(0 if run(args.output, args.seconds) else 1)
