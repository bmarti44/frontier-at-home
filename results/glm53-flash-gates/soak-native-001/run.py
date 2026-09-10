"""Evidence-only four-client native GLM durability runner; no serving imports."""
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import math
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

probe = module('glm_durability_probe', ROOT / 'scripts/48_probe_glm53_context.py')
soak = module('glm_durability_threads', ROOT / 'scripts/35_soak.py')
preparation = module('glm_durability_instruction', HERE.parent / 'context-clear-instruction-001/prepare_inputs.py')
DURATION = 1800
DRAIN_LIMIT = 2400
WORKERS = 4
INPUT_TOKENS = 4224
CONFIG = {'admission_seconds': DURATION, 'drain_limit_seconds': DRAIN_LIMIT,
          'workers': WORKERS, 'input_tokens': INPUT_TOKENS, 'max_tokens': 2048}
SOURCES = [Path(__file__).resolve(), HERE / 'PROTOCOL.md',
           Path(preparation.__file__), ROOT / 'scripts/35_soak.py',
           ROOT / 'scripts/48_probe_glm53_context.py', probe.DS, probe.BINDING,
           probe.MODEL / 'tokenizer.json', probe.MODEL / 'chat_template.jinja',
           ROOT / 'fixtures/ctx-32k.txt']
FILES = {f'{worker}-{kind}.json' for worker in range(WORKERS)
         for kind in ('request', 'fixture', 'input-token-ids')}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def finite(value):
    return type(value) in (float, int) and math.isfinite(value)

def prepare(out, seed):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    require(re.fullmatch('[0-9a-f]{64}', seed), 'invalid public seed')
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(probe.MODEL / 'tokenizer.json'))
    require(probe.sha(probe.MODEL / 'tokenizer.json') == probe.read(probe.BINDING)['tokenizer']['sha256'], 'tokenizer binding')
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
        extensions=['jinja2.ext.loopcontrols']).from_string((probe.MODEL / 'chat_template.jinja').read_text())
    for worker in range(WORKERS):
        case_seed = hashlib.sha256(f'{seed}:soak-worker:{worker}'.encode()).hexdigest()
        target = INPUT_TOKENS - 128
        for _ in range(8):
            art = probe.retrieval.build_request_artifacts(tokenizer, target=target, seed_sha256=case_seed)
            body = art['payload']
            body['messages'][0]['content'] = art['fixture']['text'] + preparation.INSTRUCTION
            body.update(model='glm-5.3-flash', max_tokens=2048, return_token_ids=True,
                chat_template_kwargs={'reasoning_effort': 'low', 'clear_thinking': True})
            rendered = template.render(messages=body['messages'], tools=[], add_generation_prompt=True,
                reasoning_effort='low', clear_thinking=True)
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            if len(ids) == INPUT_TOKENS:
                break
            target += INPUT_TOKENS - len(ids)
        require(len(ids) == INPUT_TOKENS, 'exact prompt length did not converge')
        fixture = {k: v for k, v in art['fixture'].items() if k != 'text'}
        fixture.update(worker=worker, seed=case_seed, input_tokens=len(ids))
        require(fixture['absent_value'] not in body['messages'][0]['content'], 'negative marker present')
        for kind, value in [('request', body), ('fixture', fixture), ('input-token-ids', ids)]:
            probe.write(out / f'{worker}-{kind}.json', value)
    probe.write(out / 'manifest.json', {'scope': 'native short-prompt durability inputs',
        'seed': seed, 'configuration': CONFIG, 'files': {p: probe.sha(out / p) for p in sorted(FILES)},
        'sources': {str(p): probe.sha(p) for p in SOURCES}, 'prepared_unix': time.time()})
    verify(out)

def verify(out):
    manifest = probe.read(out / 'manifest.json')
    require(manifest['configuration'] == CONFIG, 'configuration changed')
    require(set(manifest['files']) == FILES, 'fixture file coverage')
    require(set(manifest['sources']) == {str(p) for p in SOURCES}, 'source coverage')
    require(re.fullmatch('[0-9a-f]{64}', manifest['seed']), 'invalid seed')
    for name, digest in manifest['files'].items():
        require(probe.sha(out / name) == digest, 'fixture digest: ' + name)
    for name, digest in manifest['sources'].items():
        require(probe.sha(name) == digest, 'source digest: ' + name)
    for worker in range(WORKERS):
        ids = probe.read(out / f'{worker}-input-token-ids.json')
        require(len(ids) == INPUT_TOKENS and all(type(t) is int and t >= 0 for t in ids), 'input geometry')
    return probe.sha(out / 'manifest.json')

def stream(path, body, key, deadline_ns=None):
    request = urllib.request.Request('http://127.0.0.1:8015/v1/chat/completions',
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
    admitted_ns = time.monotonic_ns()
    if deadline_ns is not None and admitted_ns >= deadline_ns:
        return False
    with path.open('x') as raw:
        def event(**data):
            raw.write(json.dumps({'monotonic_ns': time.monotonic_ns(), 'time_unix': time.time(), **data}, allow_nan=False) + '\n')
            raw.flush()
        event(kind='start', monotonic_ns=admitted_ns)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                event(kind='http', status=response.status)
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    data = line[5:].strip()
                    if data == b'[DONE]':
                        event(kind='done')
                        break
                    event(kind='chunk', chunk=probe.decode(data))
        except Exception as error:
            event(kind='error', error=repr(error), body=error.read().decode(errors='replace') if hasattr(error, 'read') else None)
        event(kind='end')
    return True

def score_request(path, ids, fixture):
    rows = [probe.decode(line) for line in path.read_text().splitlines()]
    content, reasoning, usage, finish, first, terminal, response_id, outputs = probe.parse_stream(rows, ids)
    require(all(row['chunk'].get('model') == 'glm-5.3-flash' for row in rows if row['kind'] == 'chunk'), 'wrong response model')
    result = probe.retrieval.validate_completion(content=content, reasoning_content=reasoning,
        finish_reason=finish, done=True, records=fixture['records'], absent_value=fixture['absent_value'])
    require(result['pass'], 'final retrieval or negative control failed')
    require(type(usage.get('total_tokens')) is int and usage['total_tokens'] == len(ids) + len(outputs), 'total usage mismatch')
    return {'response_id': response_id, 'first_ns': first, 'terminal_ns': terminal,
            'start_ns': rows[0]['monotonic_ns'], 'end_ns': rows[-1]['monotonic_ns'],
            'input_tokens': len(ids), 'output_tokens': len(outputs), 'final_answer': content}

class HealthClient(soak.Client):
    def __init__(self, key):
        super().__init__('http://127.0.0.1:8015', key, 30)
        self.records = []
    def health(self):
        row = {'start_ns': time.monotonic_ns(), 'time_unix': time.time()}
        request = urllib.request.Request(self.base_url + '/v1/models', headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                row.update(status=response.status, body=response.read().decode())
        except Exception as error:
            row.update(status=0, error=repr(error))
        row['end_ns'] = time.monotonic_ns()
        self.records.append(row)
        return row['status']

def smoke(out, server):
    binding = verify(out)
    probe.launch_check(server)
    path = out / 'smoke'
    path.mkdir(exist_ok=False)
    stream(path / 'raw.jsonl', probe.read(out / '0-request.json'), (server / 'api-key').read_text().strip())
    try:
        result = score_request(path / 'raw.jsonl', probe.read(out / '0-input-token-ids.json'), probe.read(out / '0-fixture.json'))
        require(verify(out) == binding, 'smoke binding changed')
        result.update(verdict='PASS', scope='startup correctness only', manifest_sha256=binding)
    except Exception as error:
        result = {'verdict': 'FAIL', 'error': repr(error), 'manifest_sha256': binding}
    probe.write(path / 'summary.json', result)
    print(json.dumps({'smoke_verdict': result['verdict']}), flush=True)
    return result['verdict'] == 'PASS'

def run(out, server):
    binding = verify(out)
    launch = probe.launch_check(server)
    require(probe.read(out / 'smoke/summary.json')['verdict'] == 'PASS', 'startup correctness failed')
    require(probe.read(out / 'smoke/summary.json')['manifest_sha256'] == binding, 'smoke input binding')
    probe.write(out / 'server-launch.json', launch)
    key = (server / 'api-key').read_text().strip()
    monitor = soak.MemorySampler(interval=1.0)
    health = HealthClient(key)
    start = time.monotonic_ns()
    deadline = start + DURATION * 10**9
    health_thread = soak.HealthProber(health, start / 1e9)
    stop = threading.Event()
    barrier = threading.Barrier(WORKERS)
    journal_lock = threading.Lock()
    with (out / 'raw.jsonl').open('x') as journal:
        def event(**data):
            with journal_lock:
                journal.write(json.dumps({'observed_ns': time.monotonic_ns(), **data}, allow_nan=False) + '\n')
                journal.flush()
        event(kind='start', start_ns=start, deadline_ns=deadline, manifest_sha256=binding, launch_sha256=probe.sha(out / 'server-launch.json'))
        monitor.start()
        health_thread.start()
        def worker(worker_id):
            body = probe.read(out / f'{worker_id}-request.json')
            ids = probe.read(out / f'{worker_id}-input-token-ids.json')
            fixture = probe.read(out / f'{worker_id}-fixture.json')
            barrier.wait(timeout=60)
            count = 0
            while not stop.is_set() and time.monotonic_ns() < deadline:
                name = f'w{worker_id}-{count:05d}-raw.jsonl'
                if not stream(out / name, body, key, deadline):
                    break
                try:
                    result = score_request(out / name, ids, fixture)
                    event(kind='request', worker=worker_id, index=count, raw_path=name, verdict='PASS', **result)
                except Exception as error:
                    event(kind='request', worker=worker_id, index=count, raw_path=name, verdict='FAIL', error=repr(error))
                    stop.set()
                count += 1
            return count
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
                counts = list(pool.map(worker, range(WORKERS)))
            require(time.monotonic_ns() >= deadline or stop.is_set(), 'early admission stop')
        except Exception as error:
            stop.set()
            event(kind='runner_error', error=repr(error))
        finally:
            drained_ns = time.monotonic_ns()
            health_thread.stop()
            health_thread.join(timeout=35)
            health.health()
            sampler_alive = monitor.is_alive()
            monitor.stop()
            monitor.join(timeout=5)
            ended_ns = time.monotonic_ns()
            probe.write(out / 'monitor.json', {'memory': monitor.samples, 'memory_error': monitor.error,
                'sampler_alive_at_stop': sampler_alive, 'sampler_stopped': not monitor.is_alive(),
                'health_thread_stopped': not health_thread.is_alive(),
                'health': sorted(health.records, key=lambda r: r['start_ns']),
                'memory_clock_basis': 'seconds from sampler thread start; absolute watchdog evidence required separately'})
            event(kind='end', drained_ns=drained_ns, ended_ns=ended_ns, stopped_on_failure=stop.is_set())
    return score(out)

def score(out):
    try:
        binding = verify(out)
        journal = [probe.decode(line) for line in (out / 'raw.jsonl').read_text().splitlines()]
        require(journal[0]['kind'] == 'start' and journal[-1]['kind'] == 'end', 'journal endpoints')
        initial, final = journal[0], journal[-1]
        require(initial['manifest_sha256'] == binding, 'run input binding')
        require(initial['launch_sha256'] == probe.sha(out / 'server-launch.json'), 'launch binding')
        probe.check_launch(probe.read(out / 'server-launch.json'))
        start, deadline = initial['start_ns'], initial['deadline_ns']
        require(type(start) is int and type(deadline) is int and deadline == start + DURATION * 10**9, 'admission interval')
        require(type(final['drained_ns']) is int and type(final['ended_ns']) is int and deadline <= final['drained_ns'] <= start + DRAIN_LIMIT * 10**9, 'duration or drain bound')
        require(final['ended_ns'] >= final['drained_ns'] and not final['stopped_on_failure'], 'failed or invalid stop')
        observed = [row['observed_ns'] for row in journal]
        require(all(type(x) is int and x > 0 for x in observed) and all(a < b for a, b in zip(observed, observed[1:])), 'journal order')
        entries = journal[1:-1]
        require(len(entries) >= 30 and all(r['kind'] == 'request' and r['verdict'] == 'PASS' for r in entries), 'missing or failed requests')
        names = [r['raw_path'] for r in entries]
        require(len(set(names)) == len(names) and set(names) == {p.name for p in out.glob('*-raw.jsonl')}, 'raw file coverage')
        requests = []
        for row in entries:
            worker, index = row['worker'], row['index']
            require(type(worker) is int and worker in range(WORKERS) and type(index) is int and index >= 0, 'worker identity')
            require(row['raw_path'] == f'w{worker}-{index:05d}-raw.jsonl', 'request path identity')
            result = score_request(out / row['raw_path'], probe.read(out / f'{worker}-input-token-ids.json'), probe.read(out / f'{worker}-fixture.json'))
            require(all(row[k] == v for k, v in result.items()), 'stored request reduction differs')
            require(start <= result['start_ns'] < deadline and result['end_ns'] <= final['drained_ns'], 'request outside interval')
            require(result['end_ns'] - result['start_ns'] <= 600 * 10**9, 'request timeout')
            require(result['end_ns'] <= row['observed_ns'], 'request journal timestamp precedes completion')
            requests.append({**result, 'worker': worker, 'index': index})
        require(len({r['response_id'] for r in requests}) == len(requests), 'duplicate native response identity')
        require(start <= initial['observed_ns'] <= min(r['start_ns'] for r in requests), 'admission journal timestamp')
        require(final['ended_ns'] <= final['observed_ns'], 'terminal journal timestamp')
        intervals = []
        for worker in range(WORKERS):
            own = sorted((r for r in requests if r['worker'] == worker), key=lambda r: r['index'])
            require([r['index'] for r in own] == list(range(len(own))) and own, 'worker request coverage')
            require(own[0]['start_ns'] < start + 60 * 10**9, 'worker started late')
            require(all(a['end_ns'] < b['start_ns'] for a, b in zip(own, own[1:])), 'worker chronology')
            require(all(b['start_ns'] - a['end_ns'] <= 5 * 10**9 for a, b in zip(own, own[1:])), 'worker idle gap')
            require(own[-1]['end_ns'] >= deadline - 5 * 10**9, 'worker stopped early')
            require(sum(r['start_ns'] < start + 300 * 10**9 for r in own) >= 5, 'first window worker coverage')
            require(sum(r['start_ns'] >= deadline - 300 * 10**9 for r in own) >= 5, 'last window worker coverage')
            intervals.extend([(r['first_ns'], 1, worker) for r in own] + [(r['terminal_ns'], -1, worker) for r in own])
        active = set()
        overlap = False
        for _, change, worker in sorted(intervals):
            if change == 1:
                active.add(worker)
            else:
                active.discard(worker)
            overlap |= len(active) == WORKERS
        require(overlap, 'four generated streams never overlap')
        monitor = probe.read(out / 'monitor.json')
        elapsed = (final['ended_ns'] - start) / 1e9
        memory = monitor['memory']
        require(monitor['memory_error'] is None and monitor['sampler_alive_at_stop'] is True and monitor['sampler_stopped'] is True and monitor['health_thread_stopped'] is True, 'monitor failure')
        require(len(memory) >= 0.8 * elapsed and all(finite(r['t']) and finite(r['gib']) and r['gib'] >= 18 for r in memory), 'memory floor or density')
        require(0 <= memory[0]['t'] <= 2 and abs(memory[-1]['t'] - elapsed) <= 2, 'memory endpoint coverage')
        require(all(0 < b['t'] - a['t'] <= 2 for a, b in zip(memory, memory[1:])), 'memory sampling gap')
        health = monitor['health']
        require(len(health) >= 30 and all(type(r['status']) is int and r['status'] == 200 and type(r['start_ns']) is int and type(r['end_ns']) is int and start <= r['start_ns'] < r['end_ns'] <= final['ended_ns'] for r in health), 'health coverage or response')
        require(health[0]['start_ns'] <= start + 5 * 10**9 and final['drained_ns'] - health[-1]['start_ns'] <= 60 * 10**9, 'health endpoint coverage')
        require(all(0 < b['start_ns'] - a['start_ns'] <= 60 * 10**9 for a, b in zip(health, health[1:])), 'health gap')
        for row in health:
            require(any(m['id'] == 'glm-5.3-flash' and m['max_model_len'] == 262144 for m in probe.decode(row['body'])['data']), 'wrong health model')
        result = {'scope': 'short-prompt sustained operation client checks; separate host/freeze/lifecycle checks required', 'verdict': 'PASS',
            'admission_seconds': DURATION, 'elapsed_seconds': elapsed, 'requests': len(requests),
            'requests_per_worker': [sum(r['worker'] == w for r in requests) for w in range(WORKERS)],
            'minimum_available_gib': min(r['gib'] for r in memory), 'memory_samples': len(memory), 'health_probes': len(health),
            'four_generated_streams_overlap': overlap, 'qualified_production_performance': None}
    except Exception as error:
        result = {'scope': 'short-prompt sustained operation client checks', 'verdict': 'FAIL', 'error': repr(error)}
    probe.write(out / 'summary.json', result)
    print(json.dumps(result), flush=True)
    return result['verdict'] == 'PASS'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke', 'run', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed')
    parser.add_argument('--server', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.output, args.seed)
    else:
        result = score(args.output) if args.action == 'score' else globals()[args.action](args.output, args.server)
        sys.exit(0 if result else 1)
