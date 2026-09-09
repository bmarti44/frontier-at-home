"""Freeze a fresh model-free probe; no download, model import or CUDA call."""
import argparse
import base64
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path('/home/bmarti44/spark-deepseek-v4-flash')
BASE = Path('/home/bmarti44/.cache/glm53-flash')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('kind', choices=('native', 'cache', 'mla', 'mla-replay', 'kda', 'kda-replay', 'load-moe', 'load-kda', 'load-mla', 'load-ordinary', 'growth', 'conv', 'conv-replay'))
parser.add_argument('attempt')
args = parser.parse_args()
if not re.fullmatch({'native': 'native-smoke', 'cache': 'cache-preflight', 'mla': 'mla-preflight', 'mla-replay': 'mla-replay', 'kda': 'kda-preflight', 'kda-replay': 'kda-replay', 'growth': 'load-growth', 'conv': 'conv-preflight', 'conv-replay': 'conv-replay', **{f'load-{case}': f'load-{case}' for case in ('moe', 'kda', 'mla', 'ordinary')}}[args.kind] + r'-[0-9]{3}', args.attempt):
    raise ValueError('invalid fresh attempt name')
if subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True):
    raise ValueError('repository source is not clean')
output = BASE / args.attempt
output.mkdir()
spec = importlib.util.spec_from_file_location('runner', ROOT / 'scripts/39_run_glm53_probe.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
for name in runner.CODE_FILES:
    target = output / 'code' / name; target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / name, target)
shutil.copyfile(__file__, output / 'freeze.py')
shutil.copyfile(ROOT / 'results/glm53-flash-gates/native-smoke-003/fetch-randomness.py', output / 'fetch-randomness.py')
runtime = BASE / 'native-runtime-002/runtime'
inventory = BASE / 'native-runtime-002/inventory.json'
identities = runner.verify_inventory(runtime, runner.strict_json(inventory))
prepared = BASE / 'build-source-005'
metadata = BASE / 'processor-metadata-002'
tests = [prepared / 'vllm-exl3/tests' / name for name in ('test_exl3_linear.py', 'test_native_moe_contract.py')]
metadata_files = sorted(metadata.iterdir())
if not all(p.is_file() and not p.is_symlink() for p in metadata_files):
    raise ValueError('metadata contains unexpected file types')
external_files = [prepared / 'manifest.json', *tests, *metadata_files]
state = output / 'state'; state.mkdir()
environment = {'HOME': str(state), 'PATH': f'{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin',
               'LANG': 'C.UTF-8', 'CUDA_HOME': '/usr/local/cuda-13.0', 'CUDA_VISIBLE_DEVICES': '0',
               'TORCH_COMPILE_DISABLE': '1', 'GLM53_EXL3_BF16_SHARD_FIX': '1', 'PYTHONDONTWRITEBYTECODE': '1',
               'NVIDIA_TF32_OVERRIDE': '0', 'MAX_JOBS': '2', 'NVCC_THREADS': '1',
               'TRITON_CACHE_DIR': str(state / 'triton'), 'TORCH_EXTENSIONS_DIR': str(state / 'torch-extensions'),
               'CUDA_CACHE_PATH': str(state / 'cuda-cache'), 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
               'TOKENIZERS_PARALLELISM': 'false'}
if args.kind in (*runner.LOAD_KINDS, 'growth'):
    environment.update(FLASHINFER_DISABLE_JIT='1', CUDA_CACHE_DISABLE='1')
sealed_kernels = None
if args.kind == 'mla-replay':
    from glm53_mla_replay import prepare_bundle
    preparation = BASE / 'mla-preflight-001'
    retained = ROOT / 'results/glm53-flash-gates/mla-preflight-001'
    for name in ('manifest.json', 'summary.json', 'generated-cache-inventory.json'):
        if runner.sha256_file(preparation / name) != runner.sha256_file(retained / name):
            raise ValueError('MLA preparation differs from committed evidence')
        external_files.extend((preparation / name, retained / name))
    prior = runner.strict_json(preparation / 'summary.json')
    if prior['verdict'] != 'NO_RESULT' or prior['host']['verdict'] != 'PASS' or prior['inner']['verdict'] != 'PASS':
        raise ValueError('MLA preparation did not complete analytic and host checks')
    if runner.strict_json(preparation / 'manifest.json')['runtime']['sha256'] != runner.sha256_file(inventory):
        raise ValueError('MLA preparation used another runtime')
    sealed_kernels = prepare_bundle(preparation / 'state', output / 'kernels',
                                   runner.strict_json(preparation / 'generated-cache-inventory.json'))
    environment.update(TRITON_CACHE_DIR=str(output / 'kernels/triton'), FLASHINFER_DISABLE_JIT='1', CUDA_CACHE_DISABLE='1')
if args.kind == 'kda-replay':
    from glm53_kda_replay import prepare_bundle
    preparation = BASE / 'kda-preflight-002'
    retained = ROOT / 'results/glm53-flash-gates/kda-preflight-002'
    for name in ('manifest.json', 'summary.json', 'generated-cache-inventory.json'):
        if runner.sha256_file(preparation / name) != runner.sha256_file(retained / name):
            raise ValueError('KDA preparation differs from committed evidence')
        external_files.extend((preparation / name, retained / name))
    prior = runner.strict_json(preparation / 'summary.json')
    if prior['verdict'] != 'NO_RESULT' or prior['host']['verdict'] != 'PASS' or prior['inner']['verdict'] != 'PASS':
        raise ValueError('KDA preparation did not complete analytic and host checks')
    if runner.strict_json(preparation / 'manifest.json')['runtime']['sha256'] != runner.sha256_file(inventory):
        raise ValueError('KDA preparation used another runtime')
    sealed_kernels = prepare_bundle(preparation / 'state', output / 'kernels',
                                   runner.strict_json(preparation / 'generated-cache-inventory.json'))
    environment.update(TRITON_CACHE_DIR=str(output / 'kernels/triton'), TRITON_CACHE_AUTOTUNING='1', CUDA_CACHE_DISABLE='1')
if args.kind == 'conv-replay':
    from glm53_conv_replay import prepare_bundle
    preparation = BASE / 'conv-preflight-001'
    retained = ROOT / 'results/glm53-flash-gates/conv-preflight-001'
    for name in ('manifest.json', 'summary.json', 'generated-cache-inventory.json'):
        if runner.sha256_file(preparation / name) != runner.sha256_file(retained / name):
            raise ValueError('convolution preparation differs from committed evidence')
        external_files.extend((preparation / name, retained / name))
    prior = runner.strict_json(preparation / 'summary.json')
    if prior['verdict'] != 'NO_RESULT' or prior['host']['verdict'] != 'PASS' or prior['inner']['verdict'] != 'PASS':
        raise ValueError('convolution preparation did not complete analytic and host checks')
    if runner.strict_json(preparation / 'manifest.json')['runtime']['sha256'] != runner.sha256_file(inventory):
        raise ValueError('convolution preparation used another runtime')
    sealed_kernels = prepare_bundle(preparation / 'state', output / 'kernels',
                                   runner.strict_json(preparation / 'generated-cache-inventory.json'))
    environment.update(TRITON_CACHE_DIR=str(output / 'kernels/triton'), FLASHINFER_DISABLE_JIT='1', CUDA_CACHE_DISABLE='1')
jit_tools = []
if args.kind in ('kda', 'conv'):
    jit_tools = [Path('/usr/bin/gcc').resolve(), Path('/usr/bin/g++').resolve(),
                 Path('/usr/local/cuda-13.0/bin/nvcc')]
if args.kind == 'mla':
    jit_directory = output / 'tools'; jit_directory.mkdir()
    ninja_source = BASE / 'runtime/bin/ninja'
    if ninja_source.is_symlink() or not ninja_source.is_file():
        raise ValueError('prepared Ninja must be a regular native executable')
    record_path = runtime / 'lib/python3.12/site-packages/ninja-1.13.2.dist-info/RECORD'
    records = [row for row in csv.reader(record_path.read_text().splitlines()) if row and row[0] == '../../../bin/ninja']
    if len(records) != 1 or len(records[0]) != 3 or not records[0][1].startswith('sha256='):
        raise ValueError('inventoried Ninja wheel record is incomplete')
    expected_ninja = base64.urlsafe_b64decode(records[0][1].removeprefix('sha256=') + '=').hex()
    if runner.sha256_file(ninja_source) != expected_ninja or ninja_source.stat().st_size != int(records[0][2]):
        raise ValueError('Ninja executable differs from inventoried wheel record')
    ninja = jit_directory / 'ninja'; shutil.copyfile(ninja_source, ninja); ninja.chmod(0o555)
    if runner.sha256_file(ninja) != runner.sha256_file(ninja_source):
        raise ValueError('prepared Ninja copy digest mismatch')
    environment['PATH'] = str(jit_directory) + ':' + environment['PATH']
    version = subprocess.run([str(ninja), '--version'], env=environment, capture_output=True, text=True, timeout=10)
    if version.returncode != 0 or version.stdout != '1.13.2.git.kitware.jobserver-pipe-1\n' or version.stderr:
        raise ValueError('prepared Ninja version preflight failed')
    (output / 'jit-tool-preflight.json').write_text(json.dumps({'command': [str(ninja), '--version'],
        'returncode': version.returncode, 'stdout': version.stdout, 'stderr': version.stderr,
        'source': str(ninja_source), 'sha256': runner.sha256_file(ninja)}, indent=2) + '\n')
    jit_tools = [ninja, Path('/usr/local/cuda-13.0/bin/nvcc'), Path('/usr/bin/c++').resolve()]
node = Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node')
wrapper = ROOT / 'results/glm52-gates/harness/glm_cgroup_run.sh'
tools = [runtime / 'bin/python3', Path('/usr/bin/git'), Path('/usr/bin/bash'), node, wrapper,
         ROOT / 'results/glm52-gates/harness/glm_safe_run.sh', ROOT / 'scripts/03_memory_guard.py',
         Path('/usr/lib/aarch64-linux-gnu/libc.so.6'), *jit_tools]
sha = runner.sha256_file
native_extensions = {}
for name in ('exllamav3_ext', 'vllm_exl3_c'):
    paths = list((runtime / 'lib/python3.12/site-packages').glob(name + '.*.so'))
    if len(paths) != 1: raise ValueError('native extension inventory mismatch')
    native_extensions[name] = {'path': str(paths[0]), 'sha256': sha(paths[0])}
manifest = {'schema_version': 1, 'qualification': ('preparatory_' + args.kind.upper() + '_JIT_falsifier_only') if args.kind in ('mla', 'kda', 'conv') else 'model_free_' + args.kind + '_probe_only', 'kind': args.kind,
            'tag': 'glm53-' + args.attempt,
            'source_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'runtime': {'root': str(runtime), 'manifest': str(inventory), 'sha256': sha(inventory), 'files': len(identities)},
            'code': {name: {'sha256': sha(output / 'code' / name)} for name in runner.CODE_FILES},
            'tools': {str(path): {'sha256': sha(path)} for path in tools},
            'orchestration': {name: {'sha256': sha(output / name)} for name in ('freeze.py', 'fetch-randomness.py')},
            'external_files': {str(path): {'sha256': sha(path)} for path in external_files},
            'native_extensions': native_extensions,
            'cache_layer_types': runner.strict_json(metadata / 'config.json')['text_config']['layer_types'],
            'native_test_hashes': {path.name: sha(path) for path in tests},
            'cache_metadata_hashes': {path.name: {'sha256': sha(path)} for path in metadata_files},
            'environment': environment, 'node': str(node), 'wrapper': str(wrapper),
            'safety': {'minimum_start_gib': 110, 'kill_floor_gib': 40, 'timeout_seconds': 600},
            'probe_arguments_without_seed': [('--prepared' if args.kind == 'native' else '--metadata'),
                str(prepared if args.kind == 'native' else metadata), '--output', str(output / 'checks')],
            'seed_rule': 'uint64 from first16 hex characters of BLS-verified post-freeze drand randomness',
            'model_weights': 'none', 'frozen_at_unix': time.time()}
if args.kind in ('conv', 'conv-replay'):
    manifest['probe_arguments_without_seed'].append('--pinned-convolution')
if args.kind == 'growth':
    manifest.update(qualification='model_free_two_MoE_layers_incremental_storage_only')
    manifest['probe_arguments_without_seed'].append('--pinned-growth')
if args.kind in runner.LOAD_KINDS:
    manifest.update(qualification='model_free_component_load_storage_only')
    manifest['probe_arguments_without_seed'].extend(['--case', args.kind.removeprefix('load-'), '--pinned-stream'])
if sealed_kernels:
    manifest.update(qualification=('model_free_frozen_MLA_constant_cache_replay_only' if args.kind == 'mla-replay'
                                   else 'model_free_frozen_KDA_selected_configurations_only' if args.kind == 'kda-replay'
                                   else 'model_free_frozen_convolution_replay_only'), sealed_kernels=sealed_kernels)
    manifest['probe_arguments_without_seed'].extend(['--sealed-kernels', sealed_kernels['root'],
                                                    '--sealed-manifest-sha256', sealed_kernels['sha256']])
(output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'directory': str(output), 'runtime_files_verified': len(identities), 'frozen_at_unix': manifest['frozen_at_unix']}))
