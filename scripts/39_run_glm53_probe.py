#!/usr/bin/env python3
"""Run a frozen model-free native/cache probe; never qualifies a serving model."""
import argparse
import hashlib
import json
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


def native_ids():
    return ['test_exl3_linear_basic', 'test_exl3_linear_mixed_mul1', 'test_exl3_linear_tp_slicing'] + [
        f'native_moe_k{bits}_width{width}' for bits in (2, 3, 4) for width in (1024, 2048)] + [
        'reject_' + bad for bad in ('rows', 'width', 'negative_limit', 'nan_limit', 'strided_pointers')]


def score_inner(output, kind, seed, binding):
    root = output / 'checks'
    summary, manifest = strict_json(root / 'summary.json'), strict_json(root / 'manifest.json')
    require(summary['verdict'] == 'PASS' and summary['model_loaded'] is False and manifest['seed'] == seed, 'inner verdict or seed mismatch')
    require(all(manifest.get(key) == value for key, value in binding.items()), 'frozen inner proof binding mismatch')
    raw = read(root / 'raw.jsonl')
    require(hashlib.sha256(raw).hexdigest() == summary['raw_sha256'], 'inner raw digest mismatch')
    rows = [object_text(line) for line in raw.splitlines()]
    require(not any(r.get('event') == 'failure' for r in rows), 'inner failure record')
    if kind == 'native':
        require(summary['qualification'] == 'synthetic_native_smoke_only' and summary['checks_completed'] == 14 and
                summary['test_output_sha256'] == sha256_file(root / 'assertion-output.log'), 'native completion mismatch')
        order = native_ids(); random.Random(seed).shuffle(order)
        require([r['check_order'] for r in rows if 'check_order' in r] == [order], 'native fixture order mismatch')
        require([(r.get('event'), r.get('check_id')) for r in rows if 'event' in r] ==
                [(event, name) for name in order for event in ('start', 'pass')], 'native check coverage mismatch')
    elif kind == 'cache':
        require(summary['qualification'] == 'model_free_cache_allocation_only' and summary['actual_input_tokens_processed'] == 0, 'cache qualification mismatch')
        events = [r.get('event') for r in rows]
        require(events[-6:] == ['reserve'] * 4 + ['fifth_rejected', 'restored'] and
                rows[-1]['free_blocks'] == 144 and rows[-2]['distinct_blocks'] == 144, 'cache completion coverage mismatch')
        require(sum(r.get('event') == 'allocated_and_zeroed' and r.get('unique_backing_bytes') == 9565304320 for r in rows) == 1, 'cache backing observation missing')
    else:
        raise ValueError('unknown probe kind')
    return summary


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


def verify_beacon(output, manifest):
    receipt = strict_json(output / 'randomness.json')
    require(type(receipt['round']) is int and receipt['round'] > 0, 'invalid public round')
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
    manifest = strict_json(output / 'manifest.json')
    require(Path(__file__).resolve() == output / 'code/scripts/39_run_glm53_probe.py', 'run the frozen runner copy')
    expected_kind = {'native': '35_smoke_glm53_native.py', 'cache': '37_probe_glm53_cache.py'}
    kind = manifest['kind']; require(kind in expected_kind, 'unknown probe kind')
    python = Path(manifest['runtime']['root']) / 'bin/python3'
    guard = output / 'code/scripts/38_guard_glm53_probe.py'
    probe = output / 'code/scripts' / expected_kind[kind]
    safety = manifest['safety']
    require(safety == {'minimum_start_gib': 110, 'kill_floor_gib': 40, 'timeout_seconds': 600}, 'probe safety configuration changed')
    failure = None; host = None; inner = None
    with inference_lock() as lock_environment:
        try:
            verify_frozen(output, manifest)
            receipt = verify_beacon(output, manifest)
            arguments = [*manifest['probe_arguments_without_seed'], '--seed', str(receipt['seed'])]
            command = ['/usr/bin/env', '-i', *(k + '=' + v for k, v in sorted(manifest['environment'].items())),
                       str(python), '-I', '-B', str(guard), '--output', str(output / 'identity'), '--', str(probe), *arguments]
            wrapper = Path(manifest['wrapper'])
            require(str(wrapper) in manifest['tools'] and str(python) in manifest['tools'] and
                    'scripts/38_guard_glm53_probe.py' in manifest['code'] and 'scripts/' + probe.name in manifest['code'], 'launch inputs not frozen')
            environment = {k: v for k, v in os.environ.items() if not k.startswith(('GLM_SAFE_', 'GLM_W1_'))}
            environment.update(lock_environment, GLM_SAFE_RUN_AS_CURRENT_USER='1', GLM_SAFE_MIN_START_GIB='110',
                               GLM_SAFE_KILL_FLOOR_GIB='40', GLM_SAFE_TIMEOUT_S='600', GLM_SAFE_DONE_DIGESTS='1')
            write(output / 'invocation.json', {'command': ['/usr/bin/bash', str(wrapper), '--tag', manifest['tag'], '--', *command],
                  'start_unix': time.time(), 'freeze': {'sha256': sha256_file(output / 'manifest.json')},
                  'randomness': {'sha256': sha256_file(output / 'randomness.json')}})
            capture_wrapper(output, wrapper, manifest['tag'], command, environment, 600)
            binding = {'scorer_sha256': manifest['code']['scripts/' + probe.name]['sha256'],
                       'binary_sha256': manifest['tools'][str(python)]['sha256']}
            if kind == 'native':
                binding.update(expected_checks=14, test_hashes=manifest['native_test_hashes'],
                               source_revision=strict_json(output / 'code/configs/build-manifests/glm53-flash-sources.json')['sources']['vllm-exl3']['revision'])
            else:
                binding.update(decision=manifest['code']['configs/decision-specs/glm53-cache-preflight.json'],
                               metadata=manifest['cache_metadata_hashes'])
            inner = score_inner(output, kind, receipt['seed'], binding)
            host = score_host_observations(output, {
                'binary_sha256': manifest['tools'][str(python)]['sha256'], 'executable': str(python.resolve()),
                'guard': str(guard), 'guard_sha256': manifest['code']['scripts/38_guard_glm53_probe.py']['sha256'],
                'probe': str(probe), 'probe_sha256': manifest['code']['scripts/' + probe.name]['sha256'],
                'probe_arguments': arguments,
                'environment_sha256': hashlib.sha256(b'\0'.join(sorted(os.fsencode(k + '=' + v) for k, v in manifest['environment'].items()))).hexdigest(),
                'unit_prefix': 'glm52-' + manifest['tag'] + '-', 'maximum_sample_gap_seconds': 2.0, **safety})
        except BaseException as error:
            failure = repr(error)
        finally:
            try: verify_frozen(output, manifest)
            except Exception as error: failure = failure or repr(error)
            summary = {'verdict': 'PASS' if failure is None else 'FAIL', 'qualification': 'model_free_' + kind + '_probe_only',
                       'failure': failure, 'host': host, 'inner': inner, 'model_loaded': False,
                       'model_fidelity': 'not measured', 'context_capability': 'not measured', 'performance': 'not measured', 'end_unix': time.time()}
            write(output / 'summary.json', summary)
    print(json.dumps(summary)); return 0 if failure is None else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    raise SystemExit(run(parser.parse_args().directory.resolve()))
