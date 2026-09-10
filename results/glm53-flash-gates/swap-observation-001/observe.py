"""Explicit external diagnostic only: process counters, never serving imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROC = Path('/proc')

def swap_counters():
    values = dict(line.split() for line in (PROC / 'vmstat').read_text().splitlines())
    memory = dict(line.split(':', 1) for line in (PROC / 'meminfo').read_text().splitlines())
    return {'observed_ns': time.monotonic_ns(), 'time_unix': time.time(),
        **{key: int(values[key]) for key in ['pswpin', 'pswpout']},
        **{key + '_kib': int(memory[key].split()[0]) for key in ['MemAvailable', 'SwapTotal', 'SwapFree', 'SwapCached']}}

def stat_fields(text):
    pid, rest = text.split(' (', 1)
    name, tail = rest.rsplit(')', 1)
    fields = tail.split()
    return {'pid': int(pid), 'name': name, 'start_ticks': int(fields[19]),
            'major_faults': int(fields[9]), 'minor_faults': int(fields[7])}

def process(path):
    before = stat_fields((path / 'stat').read_text())
    status = dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines())
    group = (path / 'cgroup').read_text().splitlines()
    after = stat_fields((path / 'stat').read_text())
    if (before['pid'], before['start_ticks'], before['name']) != (after['pid'], after['start_ticks'], after['name']):
        raise ValueError('process identity changed during census')
    raw_swap = status.get('VmSwap')
    if raw_swap is not None and raw_swap.split()[1:] != ['kB']:
        raise ValueError('unexpected VmSwap units')
    return {**after, 'uid': [int(value) for value in status['Uid'].split()],
        'vm_swap_kib': None if raw_swap is None else int(raw_swap.split()[0]), 'cgroup': group}

def run(output, seconds, phase):
    output.mkdir(parents=True, exist_ok=False)
    identity = stat_fields((PROC / 'self/stat').read_text())
    manifest = {'scope': 'External read-only swap diagnostic; major-fault correlation is not proof of swap causation or model qualification',
        'phase': phase, 'duration_seconds': seconds, 'interval_seconds': 1,
        'observer': identity, 'boot_id': (PROC / 'sys/kernel/random/boot_id').read_text().strip(),
        'code': {'path': str(Path(__file__).resolve()), 'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'binary': {'path': str(Path(sys.executable).resolve()), 'sha256': hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    begin = swap_counters()
    started = time.monotonic()
    count = 0
    with (output / 'raw.jsonl').open('x') as stream:
        def event(**row):
            stream.write(json.dumps({'observed_ns': time.monotonic_ns(), **row}, allow_nan=False) + '\n')
            stream.flush()
        event(kind='start', counters=begin)
        try:
            while time.monotonic() - started < seconds:
                left = swap_counters()
                own = stat_fields((PROC / 'self/stat').read_text())
                if (own['pid'], own['start_ticks']) != (identity['pid'], identity['start_ticks']):
                    raise ValueError('observer identity changed')
                rows = []
                for path in sorted(PROC.iterdir(), key=lambda p: p.name):
                    if not path.name.isdigit(): continue
                    try: rows.append(process(path))
                    except (OSError, ValueError, KeyError, IndexError) as error:
                        rows.append({'pid': int(path.name), 'read_error': repr(error)})
                right = swap_counters()
                event(kind='census', index=count, before=left, after=right, processes=rows)
                count += 1
                delay = min(started + seconds, started + count) - time.monotonic()
                if delay > 0: time.sleep(delay)
        except BaseException as error:
            event(kind='error', error=repr(error))
            raise
        finally:
            event(kind='end', counters=swap_counters(), completed_censuses=count)
    with (output / 'raw.jsonl').open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps({'scope': manifest['scope'], 'censuses': count, 'raw_sha256': digest}), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=int, required=True)
    parser.add_argument('--phase', required=True)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 900: parser.error('duration must be1..900seconds')
    run(args.output, args.seconds, args.phase)
