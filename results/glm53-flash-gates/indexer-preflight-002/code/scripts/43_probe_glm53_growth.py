#!/usr/bin/env python3
"""Measure two live expert modules with shared scratch; never a full model load."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('glm53_component_api', Path(__file__).with_name('42_probe_glm53_load.py'))
api = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(api)
require = api.require
QUALIFICATION = 'model_free_two_MoE_layers_incremental_storage_only'
PARAMETERS = 1822563072
TABLES = 20736
SHARED = 301989888 + 12288


def minimum_bytes(layers):
    require(type(layers) is int and layers in (1, 2), 'invalid live layer count')
    return layers * (PARAMETERS + TABLES) + SHARED


def require_shared_identity(first, second):
    for key in ('shared_scratch', 'tensor_cache'):
        require(first[key] == second[key], 'shared cache backing changed')


def require_disjoint(first, second):
    intervals = sorted((r['storage_pointer'], r['storage_pointer'] + r['storage_bytes'])
                       for r in [*first, *second] if r['storage_bytes'])
    require(all(end <= start for (_, end), (start, _) in zip(intervals, intervals[1:])), 'independent storage overlaps')


def retained_receipt(layer, capture):
    import vllm_exl3.exl3 as exl3
    from exllamav3.util.tensor import g_tensor_cache
    parameters = dict(layer.named_parameters())
    require(all(getattr(p, '_exl3_owner', None) is layer for p in parameters.values()), 'parameter owner changed')
    handles = [h for row in layer._exl3_inners for h in row.values()]
    return {**api.retained_buffers(layer, 'moe', capture), 'module_object': id(layer),
            'owned_parameters': sorted(parameters), 'python_handles': [id(h) for h in handles],
            'native_handles': [id(h.bc) for h in handles],
            'scratch_keys': [list(key) for key in exl3._FUSED_TEMP_CACHE], 'tensor_cache_keys': sorted(g_tensor_cache.cache)}


def run_native(metadata, output, seed, record):
    from glm53_runtime_jit import activate_triton
    from glm53_kda_replay import reject_retuning
    from glm53_pinned_stream import stream_selected_weights
    class NoKernelCache:
        def __init__(self, *args, **kwargs): raise ValueError('load growth forbids Triton specialization')
    activate_triton(NoKernelCache, enabled=True); reject_retuning(enabled=True)
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
    config = Glm5NextConfig(**api.strict_json(metadata / 'config.json')).text_config
    require(config.hidden_size == 4096 and config.n_routed_experts == 288 and config.moe_intermediate_size == 2048 and
            config.num_hidden_layers == 45, 'model geometry changed')
    import torch
    require(torch.cuda.get_device_capability() == (12, 1) and torch.cuda.get_device_properties(0).multi_processor_count == 48,
            '48-SM GB10 SM121 required')
    record({'event': 'configured', 'selection': 'persistent_pinned_growth', 'layers': 2, 'multiprocessors': 48,
            'triton': 'all_specializations_rejected', 'retuning': 'rejected'})
    fixture = api.write_fixture(output / 'fixture', 'moe', seed)
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    capture = api.Capture(); specs = api.tensor_specs('moe'); mapping = {s['name']: s for s in specs}
    owners = []
    for index in range(2):
        def emit(row): record({'layer_index': index, **row})
        emit({'event': 'memory', 'phase': 'before_constructor', **api.memory()})
        torch.cuda.reset_peak_memory_stats()
        layer = api.construct('moe'); owners.append(layer); torch.cuda.synchronize()
        emit({'event': 'constructed', 'parameters': {name: api.describe(t) for name, t in layer.named_parameters()}, 'memory': api.memory()})
        def consume(name, temporary):
            spec = mapping[name]; api.consume_weight(layer, 'moe', spec, temporary)
            view = api.loaded_view(layer, spec, 'moe'); digest = capture.digest(view)
            require(digest == fixture['selection'][name]['sha256'], 'loaded bytes changed')
            emit({'event': 'loaded', 'name': name, 'sha256': digest, 'storage': api.describe(view)})
        torch.cuda.reset_peak_memory_stats()
        stream_selected_weights(output / 'fixture', fixture['inventory'], fixture['selection'], consume,
                                lambda row: emit({'event': 'transfer', **row}), enabled=True)
        torch.cuda.synchronize(); emit({'event': 'memory', 'phase': 'after_transfers', **api.memory()})
        torch.cuda.reset_peak_memory_stats(); layer.quant_method.process_weights_after_loading(layer); torch.cuda.synchronize()
        emit({'event': 'memory', 'phase': 'after_finalization', **api.memory()})
        for spec in specs:
            view = api.loaded_view(layer, spec, 'moe', final=True); digest = capture.digest(view)
            require(digest == fixture['selection'][spec['name']]['sha256'], 'final bytes changed')
            emit({'event': 'final_bytes', 'name': spec['name'], 'sha256': digest, 'storage': api.describe(view)})
        del view
        emit({'event': 'retained', **retained_receipt(layer, capture), 'memory': api.memory()})
    # Re-read the first live module after loading/finalizing the second.
    for spec in specs:
        view = api.loaded_view(owners[0], spec, 'moe', final=True); digest = capture.digest(view)
        require(digest == fixture['selection'][spec['name']]['sha256'], 'first layer changed during second load')
        record({'event': 'first_layer_recheck', 'name': spec['name'], 'sha256': digest, 'storage': api.describe(view)})
    del view
    require(len(owners) == 2 and owners[0] is not owners[1], 'both module owners must remain live')
    record({'event': 'both_live', 'layers': len(owners), 'first_retained': retained_receipt(owners[0], capture), 'memory': api.memory()})


def validate_retained(row, parameters, final):
    keys = {'pointer_tables', 'shared_scratch', 'tensor_cache', 'handles', 'concurrency', 'module_object',
            'owned_parameters', 'python_handles', 'native_handles', 'scratch_keys', 'tensor_cache_keys'}
    require(set(row) == keys, 'retained receipt schema mismatch')
    require(type(row['module_object']) is int and row['module_object'] > 0 and
            row['owned_parameters'] == sorted(parameters), 'parameter owner coverage mismatch')
    for key in ('python_handles', 'native_handles'):
        ids = row[key]
        require(len(ids) == 864 and len(set(ids)) == 864 and all(type(v) is int and v > 0 for v in ids), 'native handle identity coverage mismatch')
    ids = [row['module_object'], *row['python_handles'], *row['native_handles']]
    require(len(set(ids)) == len(ids), 'module/Python/native handle objects alias')
    require(sorted(row['handles'], key=lambda r: json.dumps(r, sort_keys=True)) == api.expected_handles('moe') and
            all(type(r['bits']) is int and type(r['mcg']) is bool and type(r['mul1']) is bool and
                type(r['input']) is int and type(r['output']) is int for r in row['handles']), 'handle geometry mismatch')
    require(row['scratch_keys'] == [['cuda:0', 4096, 2048, 6]] and row['tensor_cache_keys'] ==
            ['cuda:0/(1, 2048)/torch.float16/', 'cuda:0/(1, 4096)/torch.float16/'], 'global cache key coverage changed')
    api.validate_retained_geometry(row, 'moe')
    tables = row['pointer_tables']
    require(set(tables) == {f'{p}_{s}' for p in ('gate', 'up', 'down') for s in ('trellis', 'suh', 'svh')}, 'pointer-table coverage mismatch')
    for key, table in tables.items():
        projection, part = key.split('_', 1); suffix = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}[projection]
        require(set(table) == {'storage', 'values', 'alias'} and
                table['alias'] == projection + '_' + ('t' if part == 'trellis' else part) + '_ptrs', 'table alias mismatch')
        storage = table['storage']; api.validate_storage(storage)
        require(storage['dtype'] == 'torch.int64' and storage['shape'] == [288] and storage['stride'] == [1] and
                storage['storage_bytes'] == 2304 and storage['storage_offset'] == 0 and table['values'] == [
                    final[f'model.language_model.layers.3.mlp.experts.{e}.{suffix}.{part}']['storage']['data_pointer'] for e in range(288)] and
                all(type(v) is int for v in table['values']), 'table pointees differ from own module')
    shared = [*row['shared_scratch'], *row['tensor_cache']]
    independent = [*parameters.values(), *(t['storage'] for t in tables.values())]
    for storage in shared: api.validate_storage(storage)
    require_disjoint(independent, shared)
    require(sum(r['storage_bytes'] for r in independent) == PARAMETERS + TABLES and
            sum(r['storage_bytes'] for r in shared) == SHARED, 'retained byte formula mismatch')
    return independent, shared


def score_capture(root, rows, seed):
    specs = api.tensor_specs('moe'); names = [s['name'] for s in specs]; count = len(specs)
    digests, inventory = api.canonical_fixture('moe', seed)
    expected = {s['name']: digests[s['name']] for s in specs}
    fixture = api.strict_json(root / 'fixture.json')
    require(fixture == {'schema_version': 1, 'qualification': 'generated_synthetic_input_only', 'case': 'moe', 'seed': seed,
            'generator_sha256': api.sha256_file(ROOT / 'scripts/lib/glm53_load_fixture.py'), 'inventory': inventory,
            'selection': {s['name']: {'file': 'weights.safetensors', 'dtype': api.DTYPES[s['dtype']], 'shape': s['shape'],
                                    'sha256': digests[s['name']]} for s in specs},
            'excluded': {s['name']: {'dtype': api.DTYPES[s['dtype']], 'shape': s['shape'], 'sha256': digests[s['name']]} for s in api.EXCLUDED}},
            'canonical growth fixture binding mismatch')
    api.verify_inventory(root / 'fixture', inventory)
    require({p.name for p in root.iterdir()} == {'manifest.json', 'summary.json', 'raw.jsonl', 'traceback.log', 'fixture.json', 'fixture'}, 'growth artifact coverage mismatch')
    block_events = ['memory', 'constructed'] + ['loaded', 'transfer'] * count + ['memory', 'memory'] + ['final_bytes'] * count + ['retained']
    require([r.get('event') for r in rows] == ['configured'] + block_events * 2 + ['first_layer_recheck'] * count + ['both_live'], 'growth event coverage mismatch')
    require(rows[0] == {'time_unix': rows[0]['time_unix'], 'event': 'configured', 'selection': 'persistent_pinned_growth',
            'layers': 2, 'multiprocessors': 48, 'triton': 'all_specializations_rejected', 'retuning': 'rejected'} and
            type(rows[0]['layers']) is int and type(rows[0]['multiprocessors']) is int, 'growth startup mismatch')
    final_by_layer = []; retained_by_layer = []; private_by_layer = []; phase_by_layer = []
    baseline = None
    for index in range(2):
        block = rows[1 + index * len(block_events):1 + (index + 1) * len(block_events)]
        require(all(type(r.get('layer_index')) is int and r['layer_index'] == index for r in block), 'layer order mismatch')
        memory_rows = [r for r in block if r['event'] == 'memory']
        require([r['phase'] for r in memory_rows] == ['before_constructor', 'after_transfers', 'after_finalization'], 'phase coverage mismatch')
        memories = [{k: v for k, v in r.items() if k not in ('event', 'time_unix', 'layer_index', 'phase')} for r in memory_rows]
        constructed = block[1]
        require(set(constructed) == {'event', 'time_unix', 'layer_index', 'parameters', 'memory'}, 'constructor record schema mismatch')
        for memory in [*memories, constructed['memory'], block[-1]['memory']]: api.validate_memory(memory)
        if baseline is None: baseline = memories[0]['cuda_allocated']
        parameters = constructed['parameters']; layout = api.parameter_layout('moe')
        require(set(parameters) == set(layout), 'parameter coverage mismatch')
        for name, storage in parameters.items():
            api.validate_storage(storage)
            require({k: storage[k] for k in ('dtype', 'shape')} == layout[name] and
                    storage['stride'] == api.contiguous_stride(storage['shape']) and storage['storage_offset'] == 0 and
                    storage['storage_bytes'] == math.prod(storage['shape']) * api.SIZES[storage['dtype']], 'parameter layout mismatch')
        api.validate_byte_coverage(block, expected)
        by_event = {event: [r for r in block if r['event'] == event] for event in ('loaded', 'transfer', 'final_bytes')}
        require(all([r['name'] for r in group] == names for group in by_event.values()), 'tensor order mismatch')
        reuses = 0; staging = None
        for spec, loaded, transfer, final in zip(specs, by_event['loaded'], by_event['transfer'], by_event['final_bytes']):
            for observed in (loaded, final):
                require(set(observed) == {'event', 'time_unix', 'layer_index', 'name', 'sha256', 'storage'}, 'byte row schema mismatch')
                api.validate_storage(observed['storage'])
            require(loaded['storage'] == api.expected_loaded_storage('moe', spec, parameters) and final['storage'] == loaded['storage'], 'expert backing changed')
            size = api.byte_count(spec); chunks = (size + api.CAPACITY - 1) // api.CAPACITY; reuses += chunks * 2
            require(set(transfer) == {'time_unix', 'event', 'layer_index', 'name', 'bytes', 'source_sha256', 'device_sha256',
                    'staging_pointer', 'staging_bytes', 'pinned', 'upload_chunks', 'completed_reuses', 'temporary_bytes'} and
                    transfer['pinned'] is True and all(type(transfer[k]) is int for k in ('bytes', 'staging_pointer', 'staging_bytes',
                        'upload_chunks', 'completed_reuses', 'temporary_bytes')) and transfer['bytes'] == transfer['temporary_bytes'] == size and
                    transfer['staging_bytes'] == api.CAPACITY and transfer['staging_pointer'] > 0 and
                    transfer['upload_chunks'] == chunks and transfer['completed_reuses'] == reuses, 'transfer evidence mismatch')
            if staging is None: staging = transfer['staging_pointer']
            require(transfer['staging_pointer'] == staging, 'staging buffer changed within stream')
        final = {r['name']: r for r in by_event['final_bytes']}; final_by_layer.append(final)
        retained = {k: v for k, v in block[-1].items() if k not in ('time_unix', 'event', 'layer_index', 'memory')}
        private, shared = validate_retained(retained, parameters, final)
        if index:
            require_shared_identity(retained_by_layer[0], retained); require_disjoint(private_by_layer[0], private)
            prior_ids = {retained_by_layer[0]['module_object'], *retained_by_layer[0]['python_handles'], *retained_by_layer[0]['native_handles']}
            current_ids = {retained['module_object'], *retained['python_handles'], *retained['native_handles']}
            require(not (prior_ids & current_ids), 'module handles alias')
        prior = minimum_bytes(1) if index else 0
        require(memories[0]['cuda_allocated'] >= baseline + prior, 'previous module disappeared before next construction')
        for observation, current, peak in ((constructed['memory'], prior + PARAMETERS, prior + PARAMETERS),
                (memories[1], prior + PARAMETERS, prior + PARAMETERS + max(api.byte_count(s) for s in specs)),
                (memories[2], minimum_bytes(index + 1), minimum_bytes(index + 1)),
                (block[-1]['memory'], minimum_bytes(index + 1), minimum_bytes(index + 1))):
            require(observation['cuda_allocated'] >= baseline + current and observation['cuda_peak_allocated'] >= baseline + peak,
                    'growth counters contradict live storage or transfer overlap')
        require(memories[1]['pinned_allocator']['active_bytes.peak'] >= 2 * api.CAPACITY, 'two pinned buffers missing')
        retained_by_layer.append(retained); private_by_layer.append(private)
        phase_by_layer.append({'before_constructor': memories[0], 'constructed': constructed['memory'],
                               'after_transfers': memories[1], 'after_finalization': memories[2], 'retained': block[-1]['memory']})
    rechecks = rows[1 + 2 * len(block_events):-1]
    require([r['name'] for r in rechecks] == names, 'first-layer recheck coverage mismatch')
    for row in rechecks:
        require(set(row) == {'time_unix', 'event', 'name', 'sha256', 'storage'} and
                row['sha256'] == expected[row['name']] and row['storage'] == final_by_layer[0][row['name']]['storage'], 'first layer changed or moved')
    end = rows[-1]
    require(set(end) == {'time_unix', 'event', 'layers', 'first_retained', 'memory'} and type(end['layers']) is int and
            end['layers'] == 2 and end['first_retained'] == retained_by_layer[0], 'first module ownership/pointer tables changed')
    api.validate_memory(end['memory'])
    require(end['memory']['cuda_allocated'] >= baseline + minimum_bytes(2), 'both live allocation coverage missing')
    first, second = [row['retained'] for row in phase_by_layer]
    return {'layers_live': 2, 'tensors_checked_per_layer': count, 'bytes_checked_per_stage_per_layer': sum(api.byte_count(s) for s in specs),
            'known_retained_bytes': minimum_bytes(2), 'known_incremental_bytes': PARAMETERS + TABLES,
            'retained_observation_delta': {'process_kib': {key: second['process_kib'][key] - first['process_kib'][key] for key in first['process_kib']},
                **{key: second[key] - first[key] for key in ('cuda_allocated', 'cuda_reserved')}},
            'phases': phase_by_layer, 'model_loaded': False, 'full_load_fit': 'not measured'}


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--pinned-growth', action='store_true')
    args = parser.parse_args(); require(args.pinned_growth, 'explicit --pinned-growth selection required')
    require(0 <= args.seed < 2**64, 'invalid seed'); args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'qualification': QUALIFICATION, 'seed': args.seed, 'scorer_sha256': api.sha256_file(Path(__file__)),
        'binary_sha256': api.sha256_file(Path(sys.executable).resolve()),
        'decision': {'sha256': api.sha256_file(ROOT / 'configs/decision-specs/glm53-load-growth.json')},
        'metadata': {p.name: {'sha256': api.sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()}, 'start_unix': time.time()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n'); failure = None
    with (args.output / 'raw.jsonl').open('w') as raw, (args.output / 'traceback.log').open('w') as errors:
        def record(row):
            raw.write(json.dumps({'time_unix': time.time(), **row}, allow_nan=False) + '\n'); raw.flush()
        try: run_native(args.metadata, args.output, args.seed, record)
        except Exception as error:
            failure = repr(error); traceback.print_exc(file=errors); record({'event': 'failure', 'failure': failure})
    summary = {'verdict': 'FAIL' if failure else 'PASS', 'failure': failure, 'qualification': QUALIFICATION,
        'model_loaded': False, 'actual_input_tokens_processed': 0, 'context_capability': 'not measured',
        'performance': 'not measured', 'raw_sha256': api.sha256_file(args.output / 'raw.jsonl')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary))
    raise SystemExit(1 if failure else 0)


if __name__ == '__main__': main()
