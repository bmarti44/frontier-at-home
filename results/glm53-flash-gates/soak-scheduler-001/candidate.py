"""Evidence-only 256/64 configuration adapter and bounded window falsifier."""
import argparse
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('glm_scheduler_base_soak', HERE.parent / 'soak-native-001/run.py')
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)
require = runner.require

def check_launch(launch):
    argv = launch['arguments']
    require(type(argv) is list and all(type(arg) is str for arg in argv), 'malformed candidate argv')
    for flag, value in [('--max-model-len', '262144'), ('--max-num-seqs', '4'),
                        ('--max-num-batched-tokens', '256'), ('--long-prefill-token-threshold', '64')]:
        require([arg for arg in argv if arg.split('=', 1)[0] == flag] == [flag], 'missing or duplicate option: ' + flag)
        index = argv.index(flag)
        require(index + 1 < len(argv) and argv[index + 1] == value, 'wrong candidate value: ' + flag)
    require(argv.count('--no-enable-prefix-caching') == 1 and
            not any(arg.startswith('--enable-prefix-caching') or arg.startswith('--no-enable-prefix-caching=') for arg in argv),
            'prefix cache configuration changed')
    return launch

# This adapter selects only the candidate's launch validator. The inherited
# preparation, stream/retrieval parser and complete durability scorer stay intact.
runner.probe.check_launch = check_launch
runner.SOURCES = runner.SOURCES + [Path(__file__).resolve(), HERE / 'PROTOCOL.md']

def score_first_window(out):
    window = out / 'first-window'
    try:
        binding = runner.verify(out)
        rows = [runner.decode(line) for line in (window / 'raw.jsonl').read_text().splitlines()]
        require(rows[0]['kind'] == 'start' and rows[-1]['kind'] == 'end', 'window endpoints')
        initial, final = rows[0], rows[-1]
        start, deadline, end = initial['start_ns'], initial['deadline_ns'], final['ended_ns']
        require(all(type(value) is int and value > 0 for value in [start, deadline, end]), 'window time types')
        require(deadline == start + 300 * 10**9 and start < end <= start + 900 * 10**9, 'window duration/drain')
        require(initial['manifest_sha256'] == binding and
                initial['launch_sha256'] == runner.probe.sha(window / 'launch.json'), 'window input/launch binding')
        check_launch(runner.read(window / 'launch.json'))
        require(initial['smoke'] == runner.validate_smoke(out, binding, start), 'window startup binding')
        observed = [row['observed_ns'] for row in rows]
        require(all(type(value) is int and value > 0 for value in observed)
                and all(a < b for a, b in zip(observed, observed[1:])), 'window journal order')
        require(not final['stopped_on_failure'] and end <= final['observed_ns'], 'window failed stop')
        entries = rows[1:-1]
        require(len(entries) == 20 and all(row['kind'] == 'request' and row['verdict'] == 'PASS' for row in entries), 'window request coverage')
        names = [row['raw_path'] for row in entries]
        require(len(set(names)) == 20 and set(names) == {path.name for path in window.glob('*-raw.jsonl')}, 'window raw coverage')
        requests = []
        for row in entries:
            worker, index = row['worker'], row['index']
            require(type(worker) is int and worker in range(4) and type(index) is int and index in range(5), 'window worker/index')
            require(row['raw_path'] == f'w{worker}-{index:05d}-raw.jsonl', 'window path identity')
            result = runner.score_request(window / row['raw_path'], runner.read(out / f'{worker}-input-token-ids.json'), runner.read(out / f'{worker}-fixture.json'))
            require(all(row[key] == value for key, value in result.items()), 'window stored reduction differs')
            require(start <= result['start_ns'] < deadline and result['end_ns'] <= end, 'window request time bounds')
            require(result['end_ns'] - result['start_ns'] <= 600 * 10**9 and result['end_ns'] <= row['observed_ns'], 'window request timeout/journal')
            requests.append({**result, 'worker': worker, 'index': index})
        require(len({row['response_id'] for row in requests}) == 20, 'window duplicate response identity')
        require(start <= initial['observed_ns'] <= min(row['start_ns'] for row in requests), 'window admission journal')
        intervals = []
        for worker in range(4):
            own = sorted((row for row in requests if row['worker'] == worker), key=lambda row: row['index'])
            require([row['index'] for row in own] == list(range(5)), 'window five admissions per worker')
            require(own[0]['start_ns'] < start + 60 * 10**9, 'window worker started late')
            require(all(0 < b['start_ns'] - a['end_ns'] <= 5 * 10**9 for a, b in zip(own, own[1:])), 'window worker chronology/idle gap')
            intervals.extend([(row['first_ns'], 1, worker) for row in own] + [(row['terminal_ns'], -1, worker) for row in own])
        active, overlap = set(), False
        for _, change, worker in sorted(intervals):
            if change == 1: active.add(worker)
            else: active.discard(worker)
            overlap |= len(active) == 4
        require(overlap, 'window four generated streams never overlap')
        result = {'scope': 'First-window necessary-condition falsifier only; host checks required separately; not durability/context/fidelity/performance qualification',
                  'verdict': 'PASS', 'requests': 20, 'admissions_per_worker': [5] * 4,
                  'drain_seconds': (end - start) / 1e9, 'four_generated_streams_overlap': True,
                  'qualified_production_performance': None}
    except Exception as error:
        result = {'scope': 'First-window necessary-condition falsifier only', 'verdict': 'FAIL', 'error': repr(error)}
    runner.probe.write(window / 'summary.json', result)
    return result

def first_window(out, server):
    binding = runner.verify(out)
    launch = runner.probe.launch_check(server)
    smoke = runner.validate_smoke(out, binding, time.monotonic_ns())
    window = out / 'first-window'
    window.mkdir(exist_ok=False)
    runner.probe.write(window / 'launch.json', launch)
    key = (server / 'api-key').read_text().strip()
    start = time.monotonic_ns()
    deadline = start + 300 * 10**9
    stop, barrier, lock = threading.Event(), threading.Barrier(4), threading.Lock()
    with (window / 'raw.jsonl').open('x') as journal:
        def event(**values):
            with lock:
                journal.write(json.dumps({'observed_ns': time.monotonic_ns(), **values}, allow_nan=False) + '\n')
                journal.flush()
        event(kind='start', start_ns=start, deadline_ns=deadline, manifest_sha256=binding,
              launch_sha256=runner.probe.sha(window / 'launch.json'), smoke=smoke)
        def worker(worker_id):
            body = runner.read(out / f'{worker_id}-request.json')
            ids = runner.read(out / f'{worker_id}-input-token-ids.json')
            fixture = runner.read(out / f'{worker_id}-fixture.json')
            barrier.wait(timeout=60)
            for index in range(5):
                if stop.is_set(): break
                name = f'w{worker_id}-{index:05d}-raw.jsonl'
                if not runner.stream(window / name, body, key, deadline): break
                try:
                    result = runner.score_request(window / name, ids, fixture)
                    event(kind='request', worker=worker_id, index=index, raw_path=name, verdict='PASS', **result)
                except Exception as error:
                    event(kind='request', worker=worker_id, index=index, raw_path=name, verdict='FAIL', error=repr(error))
                    stop.set()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(worker, range(4)))
        except Exception as error:
            stop.set()
            event(kind='runner_error', error=repr(error))
        finally:
            event(kind='end', ended_ns=time.monotonic_ns(), stopped_on_failure=stop.is_set())
    return score_first_window(out)

def run_full(out, server):
    result = score_first_window(out)
    require(result['verdict'] == 'PASS', 'first-window falsifier must pass before durability admission')
    window = out / 'first-window'
    binding = {'scope': 'Completed necessary-condition falsifier before unchanged durability admission',
               'observed_ns': time.monotonic_ns(), 'manifest_sha256': runner.verify(out),
               'files': {str(path.relative_to(out)): runner.probe.sha(path)
                         for path in sorted(window.iterdir()) if path.is_file()}}
    with (out / 'first-window-admission-binding.json').open('x') as stream:
        json.dump(binding, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return runner.run(out, server)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke', 'first-window', 'score-window', 'run', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed')
    parser.add_argument('--server', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': runner.prepare(args.output, args.seed)
    elif args.action in ['first-window', 'score-window']:
        result = first_window(args.output, args.server) if args.action == 'first-window' else score_first_window(args.output)
        print(json.dumps(result), flush=True)
        sys.exit(0 if result['verdict'] == 'PASS' else 1)
    else:
        if args.action == 'run': result = run_full(args.output, args.server)
        elif args.action == 'score': result = runner.score(args.output)
        else: result = runner.smoke(args.output, args.server)
        sys.exit(0 if result else 1)
