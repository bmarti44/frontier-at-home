#!/usr/bin/env python3
"""Run a frozen model-free native/cache probe; never qualifies a serving model."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json, verify_inventory
from glm53_host_evidence import object_text, read, require, score_host_observations
from glm53_probe_capture import capture_wrapper, inference_lock, write

CONTROL_ENV = {'HOME': '/home/bmarti44', 'USER': 'bmarti44', 'LOGNAME': 'bmarti44',
               'LANG': 'C.UTF-8', 'PATH': '/usr/bin:/bin', 'XDG_RUNTIME_DIR': '/run/user/1000',
               'DBUS_SESSION_BUS_ADDRESS': 'unix:path=/run/user/1000/bus'}


def native_ids():
    return ['test_exl3_linear_basic', 'test_exl3_linear_mixed_mul1', 'test_exl3_linear_tp_slicing'] + [
        f'native_moe_k{bits}_width{width}' for bits in (2, 3, 4) for width in (1024, 2048)] + [
        'reject_' + bad for bad in ('rows', 'width', 'negative_limit', 'nan_limit', 'strided_pointers')]


def score_inner(output, kind, seed, binding):
    root = output / 'checks'
    summary, manifest = strict_json(root / 'summary.json'), strict_json(root / 'manifest.json')
    require(summary['verdict'] == 'PASS' and summary['model_loaded'] is False and manifest['seed'] == seed, 'inner verdict or seed mismatch')
    require(all(manifest.get(key) == value for key, value in binding.items() if key not in ('native_extensions', 'cache_layer_types')), 'frozen inner proof binding mismatch')
    raw = read(root / 'raw.jsonl')
    require(hashlib.sha256(raw).hexdigest() == summary['raw_sha256'], 'inner raw digest mismatch')
    rows = [object_text(line) for line in raw.splitlines()]
    require(not any(r.get('event') == 'failure' for r in rows), 'inner failure record')
    times = [r.get('time_unix') for r in rows]
    require(times and all(type(t) in (int, float) and math.isfinite(t) for t in times) and
            all(b > a for a, b in zip(times, times[1:])), 'invalid inner timestamps')
    if kind == 'native':
        require(summary['qualification'] == 'synthetic_native_smoke_only' and summary['checks_completed'] == 14 and
                summary['test_output_sha256'] == sha256_file(root / 'assertion-output.log'), 'native completion mismatch')
        require(len(rows) == 31, 'native raw record count mismatch')
        extensions = binding['native_extensions']
        require(set(extensions) == {'exllamav3_ext', 'vllm_exl3_c'}, 'native extension binding missing')
        for row, name in zip(rows[:2], ('exllamav3_ext', 'vllm_exl3_c')):
            require(set(row) == {'time_unix', 'native_extension', 'path', 'sha256'} and
                    row['native_extension'] == name and {'path': row['path'], 'sha256': row['sha256']} == extensions[name],
                    'native extension provenance mismatch')
        require(set(rows[2]) == {'time_unix', 'check_order'}, 'native order schema mismatch')
        for row in rows[3:]:
            keys = {'time_unix', 'event', 'check_id'}
            if row.get('event') == 'start': keys.add('global_torch_seed')
            require(set(row) == keys, 'native raw schema mismatch')
            if row.get('event') == 'start':
                value = int.from_bytes(hashlib.sha256(json.dumps([seed, row['check_id']], separators=(',', ':')).encode()).digest()[:8], 'big') % 2**63
                require(type(row['global_torch_seed']) is int and row['global_torch_seed'] == value, 'native fixture RNG mismatch')
        order = native_ids(); random.Random(seed).shuffle(order)
        require([r['check_order'] for r in rows if 'check_order' in r] == [order], 'native fixture order mismatch')
        require([(r.get('event'), r.get('check_id')) for r in rows if 'event' in r] ==
                [(event, name) for name in order for event in ('start', 'pass')], 'native check coverage mismatch')
    elif kind == 'cache':
        require(summary['qualification'] == 'model_free_cache_allocation_only' and summary['actual_input_tokens_processed'] == 0, 'cache qualification mismatch')
        validate_cache_rows(rows, seed, binding['cache_layer_types'])
    else:
        raise ValueError('unknown probe kind')
    return summary


def validate_cache_rows(rows, seed, layer_types):
    expected_events = ['normalized', 'allocated_and_zeroed', 'scheduler_normalized'] + ['reserve'] * 4 + ['fifth_rejected', 'restored']
    require([r.get('event') for r in rows] == expected_events, 'cache event coverage mismatch')
    def fields(row, names):
        require(set(row) == {'time_unix', 'event', *names}, 'cache raw schema mismatch')
    normalized, allocation, scheduler = rows[:3]
    fields(normalized, {'attention_block_tokens', 'mamba_block_tokens', 'mamba_shapes', 'mamba_dtypes', 'layout', 'groups', 'shared_pool_blocks'})
    require(normalized['attention_block_tokens'] == 8704 and normalized['mamba_block_tokens'] == 262144 and
            normalized['mamba_shapes'] == [[3, 24576], [64, 128, 128]] and normalized['mamba_dtypes'] == ['torch.bfloat16', 'torch.float32'] and
            normalized['layout'] == 'LBHNC' and normalized['shared_pool_blocks'] == 145, 'cache normalized geometry mismatch')
    require(len(layer_types) == 45 and layer_types.count('linear_attention') == 34 and layer_types.count('deepseek_sparse_attention') == 11, 'cache model layer coverage mismatch')
    mla, tail, mamba = set(), set(), set()
    for i, kind in enumerate(layer_types):
        prefix = f'model.language_model.layers.{i}.self_attn'
        if kind == 'linear_attention': mamba.add(prefix)
        else: mla.update((prefix + '.attn', prefix + '.indexer.k_cache')); tail.add(prefix + '.indexer.tail_cache')
    groups = normalized['groups']
    require(isinstance(groups, list) and len(groups) == 6 and
            all(set(g) == {'layers', 'spec'} and isinstance(g['spec'], str) and g['spec'] and isinstance(g['layers'], list) for g in groups), 'cache group schema mismatch')
    require([len(g['layers']) for g in groups] == [22, 11, 9, 9, 8, 8] and set(groups[0]['layers']) == mla and set(groups[1]['layers']) == tail and
            set(name for g in groups[2:] for name in g['layers']) == mamba, 'cache group layer coverage mismatch')
    fields(allocation, {'unique_backing_bytes', 'layer_views', 'cuda_memory_allocated', 'cuda_memory_reserved', 'cuda_peak_allocated'})
    require(all(type(allocation[k]) is int and allocation[k] >= 0 for k in ('unique_backing_bytes', 'layer_views', 'cuda_memory_allocated', 'cuda_memory_reserved', 'cuda_peak_allocated')) and
            allocation['unique_backing_bytes'] == 9565304320 and allocation['layer_views'] == 67 and
            9565304320 <= allocation['cuda_memory_allocated'] <= min(allocation['cuda_memory_reserved'], allocation['cuda_peak_allocated']), 'cache backing observation mismatch')
    fields(scheduler, {'scheduler_block_tokens', 'hash_block_tokens'})
    require(scheduler['scheduler_block_tokens'] == scheduler['hash_block_tokens'] == 4456448, 'cache scheduler normalization mismatch')
    order = [f'cache-preflight-{i}' for i in range(5)]; random.Random(seed).shuffle(order)
    all_ids = set()
    for i, row in enumerate(rows[3:7]):
        fields(row, {'request_id', 'block_ids', 'free_blocks'})
        require(row['request_id'] == order[i] and type(row['free_blocks']) is int and row['free_blocks'] == 144 - 36 * (i + 1), 'cache request order or accounting mismatch')
        groups = row['block_ids']
        require(isinstance(groups, list) and all(isinstance(g, list) for g in groups) and [len(g) for g in groups] == [31, 1, 1, 1, 1, 1], 'cache reservation geometry mismatch')
        ids = [value for group in groups for value in group]
        require(all(type(value) is int and 0 < value < 145 for value in ids) and len(set(ids)) == 36 and not all_ids.intersection(ids), 'cache physical ID coverage mismatch')
        all_ids.update(ids)
    require(all_ids == set(range(1, 145)), 'cache physical coverage incomplete')
    fields(rows[7], {'live_requests', 'distinct_blocks'}); fields(rows[8], {'free_blocks'})
    require(rows[7]['live_requests'] == 4 and rows[7]['distinct_blocks'] == 144 and rows[8]['free_blocks'] == 144, 'cache completion accounting mismatch')


def wrapper_environment(lock_environment):
    return {**CONTROL_ENV, **lock_environment, 'GLM_SAFE_RUN_AS_CURRENT_USER': '1', 'GLM_SAFE_MIN_START_GIB': '110',
            'GLM_SAFE_KILL_FLOOR_GIB': '40', 'GLM_SAFE_TIMEOUT_S': '600', 'GLM_SAFE_DONE_DIGESTS': '1'}


def verify_accepted_inputs(output, accepted):
    require(all(sha256_file(output / name) == digest for name, digest in accepted.items()), 'accepted manifest or randomness changed')


CODE_FILES = (
    'scripts/39_run_glm53_probe.py', 'scripts/38_guard_glm53_probe.py',
    'scripts/35_smoke_glm53_native.py', 'scripts/37_probe_glm53_cache.py',
    'scripts/lib/glm53_contract.py', 'scripts/lib/glm53_host_evidence.py', 'scripts/lib/glm53_probe_capture.py',
    'configs/build-manifests/glm53-flash-sources.json', 'configs/decision-specs/glm53-cache-preflight.json',
    'scripts/103_verify_drand_receipt_bundle.mjs')


def verify_frozen(output, manifest):
    require(set(manifest['code']) == set(CODE_FILES), 'incomplete code hash coverage')
    require({str(p.relative_to(output / 'code')) for p in (output / 'code').rglob('*') if p.is_file()} == set(CODE_FILES), 'unexpected frozen code files')
    for name, row in manifest['code'].items():
        require(sha256_file(output / 'code' / name) == row['sha256'], 'frozen code changed: ' + name)
    for name, row in manifest['tools'].items():
        require(sha256_file(Path(name)) == row['sha256'], 'frozen tool changed: ' + name)
    for name, row in manifest['orchestration'].items():
        require(sha256_file(output / name) == row['sha256'], 'frozen orchestration changed: ' + name)
    for name, row in manifest['external_files'].items():
        require(sha256_file(Path(name)) == row['sha256'], 'frozen fixture or metadata changed: ' + name)
    runtime = manifest['runtime']; inventory = Path(runtime['manifest'])
    require(sha256_file(inventory) == runtime['sha256'], 'runtime inventory changed')
    verify_inventory(Path(runtime['root']), strict_json(inventory))


def verify_beacon(output, manifest, receipt=None):
    if receipt is None: receipt = object_text(read(output / 'randomness.json'))
    require(type(receipt['round']) is int and receipt['round'] > 0, 'invalid public round')
    chosen_round = int((manifest['frozen_at_unix'] - 1595431050) // 30) + 2
    require(receipt['round'] == chosen_round and receipt['frozen_at_unix'] == manifest['frozen_at_unix'], 'beacon round or freeze reference changed')
    publication = 1595431050 + (receipt['round'] - 1) * 30
    require(publication > manifest['frozen_at_unix'] and receipt['publication_unix'] == publication, 'beacon precedes freeze')
    require(receipt['seed'] == int(receipt['randomness'][:16], 16), 'seed derivation mismatch')
    node = manifest['node']
    require(node in manifest['tools'] and 'scripts/103_verify_drand_receipt_bundle.mjs' in manifest['code'], 'verifier is not frozen')
    result = subprocess.run([node, str(output / 'code/scripts/103_verify_drand_receipt_bundle.mjs'), str(receipt['round']),
                             receipt['randomness'], receipt['signature'], receipt['previous_signature']],
                            env={'PATH': '/usr/bin:/bin', 'HOME': '/nonexistent'}, capture_output=True, text=True, timeout=30)
    write(output / 'launch-beacon-verification.json', {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr})
    require(result.returncode == 0 and result.stdout == 'DRAND_BLS_RECEIPT_OK\n', 'BLS verification failed')
    return receipt


def run(output):
    require(not sys.flags.optimize and sys.flags.isolated and sys.dont_write_bytecode, 'runner requires unoptimized isolated -I -B Python')
    require(dict(os.environ) == CONTROL_ENV, 'controller requires the declared clean environment')
    input_bytes = {name: read(output / name) for name in ('manifest.json', 'randomness.json')}
    accepted = {name: hashlib.sha256(data).hexdigest() for name, data in input_bytes.items()}
    manifest = object_text(input_bytes['manifest.json'])
    require(Path(__file__).resolve() == output / 'code/scripts/39_run_glm53_probe.py', 'run the frozen runner copy')
    expected_kind = {'native': '35_smoke_glm53_native.py', 'cache': '37_probe_glm53_cache.py'}
    kind = manifest['kind']; require(kind in expected_kind, 'unknown probe kind')
    python = Path(manifest['runtime']['root']) / 'bin/python3'
    require(Path(sys.executable).resolve() == python.resolve(), 'controller must use frozen packaged interpreter')
    guard = output / 'code/scripts/38_guard_glm53_probe.py'
    probe = output / 'code/scripts' / expected_kind[kind]
    safety = manifest['safety']
    require(safety == {'minimum_start_gib': 110, 'kill_floor_gib': 40, 'timeout_seconds': 600}, 'probe safety configuration changed')
    failure = None; host = None; inner = None
    with inference_lock() as lock_environment:
        try:
            verify_frozen(output, manifest)
            receipt = verify_beacon(output, manifest, object_text(input_bytes['randomness.json']))
            verify_accepted_inputs(output, accepted)
            arguments = [*manifest['probe_arguments_without_seed'], '--seed', str(receipt['seed'])]
            command = ['/usr/bin/env', '-i', *(k + '=' + v for k, v in sorted(manifest['environment'].items())),
                       str(python), '-I', '-B', str(guard), '--output', str(output / 'identity'), '--', str(probe), *arguments]
            wrapper = Path(manifest['wrapper'])
            require(str(wrapper) in manifest['tools'] and str(python) in manifest['tools'] and
                    'scripts/38_guard_glm53_probe.py' in manifest['code'] and 'scripts/' + probe.name in manifest['code'], 'launch inputs not frozen')
            environment = wrapper_environment(lock_environment)
            write(output / 'invocation.json', {'command': ['/usr/bin/bash', str(wrapper), '--tag', manifest['tag'], '--', *command],
                  'start_unix': time.time(), 'freeze': {'sha256': sha256_file(output / 'manifest.json')},
                  'randomness': {'sha256': sha256_file(output / 'randomness.json')}})
            capture_wrapper(output, wrapper, manifest['tag'], command, environment, 600)
            binding = {'scorer_sha256': manifest['code']['scripts/' + probe.name]['sha256'],
                       'binary_sha256': manifest['tools'][str(python)]['sha256']}
            if kind == 'native':
                binding.update(expected_checks=14, test_hashes=manifest['native_test_hashes'], native_extensions=manifest['native_extensions'],
                               source_revision=strict_json(output / 'code/configs/build-manifests/glm53-flash-sources.json')['sources']['vllm-exl3']['revision'])
            else:
                binding.update(decision=manifest['code']['configs/decision-specs/glm53-cache-preflight.json'],
                               metadata=manifest['cache_metadata_hashes'], cache_layer_types=manifest['cache_layer_types'])
            inner = score_inner(output, kind, receipt['seed'], binding)
            host = score_host_observations(output, {
                'binary_sha256': manifest['tools'][str(python)]['sha256'], 'executable': str(python.resolve()),
                'guard': str(guard), 'guard_sha256': manifest['code']['scripts/38_guard_glm53_probe.py']['sha256'],
                'probe': str(probe), 'probe_sha256': manifest['code']['scripts/' + probe.name]['sha256'],
                'probe_arguments': arguments,
                'environment_sha256': hashlib.sha256(b'\0'.join(sorted(os.fsencode(k + '=' + v) for k, v in manifest['environment'].items()))).hexdigest(),
                'unit_prefix': 'glm52-' + manifest['tag'] + '-', 'maximum_sample_gap_seconds': 2.0, **safety})
            identities = [object_text(line) for line in read(output / 'identity/raw.jsonl').splitlines()]
            inner_rows = [object_text(line) for line in read(output / 'checks/raw.jsonl').splitlines()]
            require(identities[0]['time_unix'] <= inner_rows[0]['time_unix'] <= inner_rows[-1]['time_unix'] <= identities[-2]['time_unix'], 'inner probe timestamps escape identity window')
        except BaseException as error:
            failure = repr(error)
        finally:
            try:
                verify_frozen(output, manifest)
                verify_accepted_inputs(output, accepted)
            except Exception as error: failure = failure or repr(error)
            summary = {'verdict': 'PASS' if failure is None else 'FAIL', 'qualification': 'model_free_' + kind + '_probe_only',
                       'failure': failure, 'host': host, 'inner': inner, 'model_loaded': False,
                       'model_fidelity': 'not measured', 'context_capability': 'not measured', 'performance': 'not measured', 'end_unix': time.time()}
            write(output / 'summary.json', summary)
    print(json.dumps(summary)); return 0 if failure is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    raise SystemExit(run(parser.parse_args().directory.resolve()))
