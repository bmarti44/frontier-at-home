#!/usr/bin/env python3
"""Full-geometry synthetic component loading, never a full model load."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json, verify_inventory
from glm53_load_fixture import tensor_specs, fixture_bytes, byte_count
from glm53_mla_replay import file_inventory

QUALIFICATION = 'model_free_component_load_storage_only'
DTYPES = {'I16': 'torch.int16', 'I32': 'torch.int32', 'F16': 'torch.float16', 'BF16': 'torch.bfloat16'}
SIZES = {'torch.int16': 2, 'torch.int32': 4, 'torch.float16': 2, 'torch.bfloat16': 2}
CAPACITY = 8 * 1024 * 1024
EXCLUDED = [{'name': 'model.language_model.layers.45.input_layernorm.weight', 'dtype': 'BF16', 'shape': [4096]},
            {'name': 'model.language_model.layers.0.self_attn.q_proj.weight', 'dtype': 'BF16', 'shape': [8192, 4096]}]


def require(value, message):
    if not value: raise ValueError(message)


def parameter_layout(case):
    if case == 'moe':
        result = {}
        for prefix, tree, suh, svh, marker in (
                ('w13_', [288, 2, 256, 128, 32], [288, 2, 4096], [288, 2, 2048], [288, 2, 1]),
                ('w2_', [288, 128, 256, 32], [288, 2048], [288, 4096], [288, 1])):
            for part, dtype, shape in (('trellis', 'torch.int16', tree), ('suh', 'torch.float16', suh),
                    ('svh', 'torch.float16', svh), ('mcg', 'torch.int32', marker), ('mul1', 'torch.int32', marker)):
                result[prefix + part] = {'dtype': dtype, 'shape': shape}
        return result
    if case in ('kda', 'mla'):
        shards, columns, output, bf16 = (6, 1556, 24896, 320) if case == 'kda' else (2, 128, 2048, 0)
        return {name: {'dtype': dtype, 'shape': shape} for name, dtype, shape in (
            ('trellis', 'torch.int16', [256, columns, 64]), ('suh', 'torch.float16', [shards, 4096]),
            ('svh', 'torch.float16', [output]), ('mcg', 'torch.int32', [shards, 1]),
            ('mul1', 'torch.int32', [shards, 1]), ('weight', 'torch.bfloat16', [bf16, 4096]))}
    if case == 'ordinary': return {'weight': {'dtype': 'torch.bfloat16', 'shape': [154880, 4096]}}
    raise ValueError('unknown component case')


def layout_bytes(layout):
    return sum(math.prod(row['shape']) * SIZES[row['dtype']] for row in layout.values())


def describe(tensor):
    storage = tensor.untyped_storage()
    return {'device': str(tensor.device), 'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
            'stride': list(tensor.stride()), 'storage_pointer': storage.data_ptr(),
            'storage_bytes': storage.nbytes(), 'data_pointer': tensor.data_ptr(), 'storage_offset': tensor.storage_offset()}


def validate_byte_coverage(rows, expected):
    for event in ('loaded', 'transfer', 'final_bytes'):
        selected = [r for r in rows if r.get('event') == event]
        require(len(selected) == len(expected) and {r['name'] for r in selected} == set(expected), 'byte coverage mismatch')
        for row in selected:
            keys = ('source_sha256', 'device_sha256') if event == 'transfer' else ('sha256',)
            require(all(row.get(key) == expected[row['name']] for key in keys), 'component byte digest mismatch')


def fixture_digest(seed, spec):
    digest = hashlib.sha256()
    for offset in range(0, byte_count(spec), CAPACITY):
        digest.update(fixture_bytes(seed, spec, offset, min(CAPACITY, byte_count(spec) - offset)))
    return digest.hexdigest()


def canonical_fixture(case, seed):
    """Independently reconstruct the complete input file, including excluded bytes."""
    specs = sorted([*tensor_specs(case), *EXCLUDED], key=lambda row: row['name'])
    header = {'__metadata__': {'qualification': 'synthetic_load_fixture_only'}}; offset = 0
    for spec in specs:
        end = offset + byte_count(spec)
        header[spec['name']] = {'dtype': spec['dtype'], 'shape': spec['shape'], 'data_offsets': [offset, end]}
        offset = end
    encoded = json.dumps(header, separators=(',', ':')).encode(); encoded += b' ' * (-len(encoded) % 8)
    digest = hashlib.sha256(struct.pack('<Q', len(encoded)) + encoded); tensor_digests = {}
    for spec in specs:
        tensor_hash = hashlib.sha256()
        for start in range(0, byte_count(spec), CAPACITY):
            data = fixture_bytes(seed, spec, start, min(CAPACITY, byte_count(spec) - start))
            digest.update(data); tensor_hash.update(data)
        tensor_digests[spec['name']] = tensor_hash.hexdigest()
    return tensor_digests, {'schema_version': 1, 'files': [{'path': 'weights.safetensors',
        'size_bytes': 8 + len(encoded) + offset, 'sha256': digest.hexdigest()}]}


def write_fixture(root, case, seed):
    """Generated inputs are retained; all headers/payloads precede CUDA loading."""
    specs = tensor_specs(case)
    # Real excluded shapes; the selected iterator must never call get_tensor on these.
    excluded = EXCLUDED
    all_specs = sorted([*specs, *excluded], key=lambda row: row['name'])
    header = {'__metadata__': {'qualification': 'synthetic_load_fixture_only'}}; offset = 0
    for spec in all_specs:
        end = offset + byte_count(spec)
        header[spec['name']] = {'dtype': spec['dtype'], 'shape': spec['shape'], 'data_offsets': [offset, end]}; offset = end
    encoded = json.dumps(header, separators=(',', ':')).encode(); encoded += b' ' * (-len(encoded) % 8)
    root.mkdir(); path = root / 'weights.safetensors'; digests = {}
    with path.open('xb') as stream:
        stream.write(struct.pack('<Q', len(encoded))); stream.write(encoded)
        for spec in all_specs:
            digest = hashlib.sha256()
            for offset in range(0, byte_count(spec), CAPACITY):
                data = fixture_bytes(seed, spec, offset, min(CAPACITY, byte_count(spec) - offset))
                digest.update(data); stream.write(data)
            digests[spec['name']] = digest.hexdigest()
        stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); root.chmod(0o555)
    return {'schema_version': 1, 'qualification': 'generated_synthetic_input_only', 'case': case, 'seed': seed,
            'generator_sha256': sha256_file(ROOT / 'scripts/lib/glm53_load_fixture.py'), 'inventory': file_inventory(root),
            'selection': {s['name']: {'file': path.name, 'dtype': DTYPES[s['dtype']], 'shape': s['shape'],
                                    'sha256': digests[s['name']]} for s in specs},
            'excluded': {s['name']: {'dtype': DTYPES[s['dtype']], 'shape': s['shape'], 'sha256': digests[s['name']]} for s in excluded}}


def memory():
    import torch
    raw = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB': raw[parts[0].rstrip(':')] = int(parts[1])
    require(all(key in raw for key in ('Rss', 'Pss', 'Pss_Anon', 'Pss_File')), 'missing process memory fields')
    host = torch.cuda.memory.host_memory_stats()
    # Allocation counters only: empty timing buckets can contain non-metric sentinels.
    pinned = {key: value for key, value in host.items() if key.startswith(('allocated_bytes.', 'active_bytes.', 'allocations.', 'active_requests.'))}
    return {'process_kib': {key: raw[key] for key in ('Rss', 'Pss', 'Pss_Anon', 'Pss_File')},
            'cuda_allocated': torch.cuda.memory_allocated(), 'cuda_reserved': torch.cuda.memory_reserved(),
            'cuda_peak_allocated': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved': torch.cuda.max_memory_reserved(),
            'pinned_allocator': pinned}


def construct(case):
    import torch
    import vllm_exl3.exl3 as exl3
    if case == 'moe':
        from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
        mapping = RoutedExperts.build_expert_params_mapping('gate_proj', 'down_proj', 'up_proj', 288, routed_experts_prefix='')
        layer = torch.nn.Module(); layer.prefix = layer.layer_name = 'model.language_model.layers.3.mlp.experts'
        layer.tp_rank = 0; layer.moe_tp_size = 1
        layer._map_global_expert_id_to_local_expert_id = lambda value: value if 0 <= value < 288 else -1
        layer.get_expert_mapping = lambda **kwargs: mapping
        layer.quant_method = exl3.Exl3MoEMethod(None, exl3.Exl3Config(bits=2), bits=2)
        with torch.device('cuda'): layer.quant_method.create_weights(layer, 288, 4096, 2048, torch.bfloat16)
        require(not hasattr(layer, 'w13_weight') and not hasattr(layer, 'w2_weight'), 'dense expert allocation forbidden')
    elif case in ('kda', 'mla'):
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear
        prefix = ('model.language_model.layers.0.self_attn.in_proj_qkvbfg_a' if case == 'kda'
                  else 'model.language_model.layers.3.self_attn.fused_qkv_a_proj')
        outputs = [8192, 8192, 8192, 64, 128, 128] if case == 'kda' else [1536, 512]
        cfg = exl3.Exl3Config(bits=2, non_routed_exl3={'layers': {prefix: {'bits': 4,
                            'bf16_shards': [3, 4, 5] if case == 'kda' else []}}, 'codebook': 'mul1'})
        with torch.device('cuda'):
            layer = MergedColumnParallelLinear(4096, outputs, bias=False, params_dtype=torch.bfloat16,
                                               quant_config=cfg, prefix=prefix, disable_tp=True)
        require(type(layer.quant_method) is exl3.Exl3LinearMethod, 'actual EXL3 linear method not selected')
    else:
        require(case == 'ordinary', 'unknown load case')
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.empty((154880, 4096), device='cuda', dtype=torch.bfloat16), requires_grad=False)
    observed = {name: {'dtype': str(t.dtype), 'shape': list(t.shape)} for name, t in layer.named_parameters()}
    require(observed == parameter_layout(case), 'actual constructor parameter layout mismatch')
    return layer


def loaded_view(layer, spec, case, final=False):
    part, shard, expert = spec['parameter'], spec['shard_id'], spec['expert_id']
    if case == 'moe':
        if final:
            projection = {'w1': 'gate', 'w3': 'up', 'w2': 'down'}[shard]
            suffix = part.split('_', 1)[1]
            value = getattr(layer._exl3_inners[expert][projection], suffix + '_tensor' if suffix in ('mcg', 'mul1') else suffix)
        else:
            parameter = getattr(layer, part)
            value = parameter[expert] if shard == 'w2' else parameter[expert, 0 if shard == 'w1' else 1]
    elif case == 'ordinary': value = layer.weight
    elif final:
        if part == 'weight':
            start = sum([64, 128, 128][:shard - 3]); value = layer._exl3_bf16_weight[start:start + spec['shape'][0]]
        else:
            value = getattr(layer._exl3_linears[shard], part + '_tensor' if part in ('mcg', 'mul1') else part)
    else:
        outputs = [8192, 8192, 8192, 64, 128, 128] if case == 'kda' else [1536, 512]
        if part == 'weight':
            start = sum(outputs[3:shard]); value = layer.weight[start:start + spec['shape'][0]]
        elif part == 'trellis':
            start = sum(outputs[:shard]) // 16; value = layer.trellis[:, start:start + outputs[shard] // 16, :]
        elif part == 'svh':
            start = sum(outputs[:shard]); value = layer.svh[start:start + outputs[shard]]
        else: value = getattr(layer, part)[shard]
    require(value is not None and str(value.dtype) == DTYPES[spec['dtype']] and value.numel() * value.element_size() == byte_count(spec), 'loaded view geometry mismatch')
    return value.reshape(spec['shape'])


class Capture:
    """Persistent pinned readback; temporary contiguous copies are diagnostic costs."""
    def __init__(self):
        import torch
        self.buffer = torch.empty(CAPACITY, dtype=torch.uint8, pin_memory=True)
        require(self.buffer.is_pinned(), 'readback buffer not pinned')
        self.event = torch.cuda.Event()

    def chunks(self, tensor):
        contiguous = tensor.contiguous().reshape(-1).view(__import__('torch').uint8)
        for offset in range(0, contiguous.numel(), CAPACITY):
            count = min(CAPACITY, contiguous.numel() - offset)
            self.buffer[:count].copy_(contiguous[offset:offset + count], non_blocking=True)
            self.event.record(); self.event.synchronize()
            yield memoryview(self.buffer[:count].numpy())

    def digest(self, tensor):
        digest = hashlib.sha256()
        for chunk in self.chunks(tensor): digest.update(chunk)
        return digest.hexdigest()

    def pointer_values(self, tensor):
        raw = b''.join(bytes(chunk) for chunk in self.chunks(tensor))
        return list(struct.unpack('<' + 'Q' * (len(raw) // 8), raw))


def reject_retained_temporary(layer, temporary):
    """Inspect module-owned tensor/container attributes without retaining their storage."""
    import torch
    pointer = temporary.untyped_storage().data_ptr(); seen = set()
    def visit(value):
        if id(value) in seen: return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            require(value.untyped_storage().data_ptr() != pointer, 'consumer retained transfer temporary')
        elif isinstance(value, torch.nn.Module): visit(vars(value))
        elif isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value: visit(child)
    visit(layer)


def consume_weight(layer, case, spec, temporary):
    if case == 'moe':
        local = spec['name'].removeprefix(layer.layer_name + '.')
        require(list(layer.load_weights([(local, temporary)])) == [spec['parameter']], 'actual expert name routing failed')
    elif case == 'ordinary':
        from vllm.model_executor.model_loader.weight_utils import default_weight_loader
        default_weight_loader(layer.weight, temporary)
    else:
        parameter = getattr(layer, spec['parameter'])
        parameter.weight_loader(parameter, temporary, spec['shard_id'])
    view = loaded_view(layer, spec, case)
    require(view.untyped_storage().data_ptr() != temporary.untyped_storage().data_ptr(), 'consumer retained transfer temporary')
    reject_retained_temporary(layer, temporary)


def expected_handles(case):
    geometry = ([(4096, 2048), (4096, 2048), (2048, 4096)] * 288 if case == 'moe' else
                [(4096, 8192)] * 3 if case == 'kda' else [(4096, 1536), (4096, 512)] if case == 'mla' else [])
    return sorted([{'bits': 2 if case == 'moe' else 4, 'mcg': case == 'moe', 'mul1': case != 'moe',
                    'input': i, 'output': o} for i, o in geometry], key=lambda r: json.dumps(r, sort_keys=True))


def retained_buffers(layer, case, capture):
    import torch
    import vllm_exl3.exl3 as exl3
    from exllamav3.util.tensor import g_tensor_cache
    result = {'pointer_tables': {}, 'shared_scratch': [], 'tensor_cache': [], 'handles': []}
    from exllamav3.modules.quant.exl3 import LinearEXL3
    import exllamav3_ext
    handles = ([h for row in layer._exl3_inners for h in row.values()] if case == 'moe' else
               [h for h in layer._exl3_linears if h is not None] if case in ('kda', 'mla') else [])
    for handle in handles:
        require(type(handle) is LinearEXL3 and type(handle.bc) is exllamav3_ext.BC_LinearEXL3, 'native handle class mismatch')
        result['handles'].append({'bits': handle.K, 'mcg': handle.mcg, 'mul1': handle.mul1,
                                  'input': handle.in_features, 'output': handle.out_features})
    require(sorted(result['handles'], key=lambda r: json.dumps(r, sort_keys=True)) == expected_handles(case), 'native handle geometry mismatch')
    def collect(value, output):
        if isinstance(value, torch.Tensor): output.append(describe(value))
        elif isinstance(value, dict):
            for child in value.values(): collect(child, output)
        elif isinstance(value, (list, tuple)):
            for child in value: collect(child, output)
    collect(exl3._FUSED_TEMP_CACHE, result['shared_scratch'])
    collect(vars(g_tensor_cache), result['tensor_cache'])
    if case == 'moe':
        require(len(layer._exl3_inners) == 288 and layer._exl3_codebook_flags == (True, False) * 3 and layer._exl3_k == 2, 'incomplete native expert handles')
        require(isinstance(layer._exl3_ptrs, dict) and len(layer._exl3_ptrs) == 18, 'pointer-table finalization failed')
        for projection in ('gate', 'up', 'down'):
            for part in ('trellis', 'suh', 'svh'):
                name = projection + '_' + part
                alias = projection + '_' + ('t' if part == 'trellis' else part) + '_ptrs'
                tensor = layer._exl3_ptrs[name]
                require(layer._exl3_ptrs[alias] is tensor, 'pointer-table alias mismatch')
                values = capture.pointer_values(tensor)
                require(values == [getattr(row[projection], part).data_ptr() for row in layer._exl3_inners], 'pointer-table pointee mismatch')
                result['pointer_tables'][name] = {'storage': describe(tensor), 'values': values, 'alias': alias}
        require(sum(row['storage']['storage_bytes'] for row in result['pointer_tables'].values()) == 20736, 'pointer-table byte count mismatch')
        result['concurrency'] = layer._exl3_fused_concurrency
        unique = {(row['device'], row['storage_pointer']): row['storage_bytes'] for row in result['shared_scratch']}
        require(sum(unique.values()) == 50331648 * result['concurrency'], 'shared scratch coverage mismatch')
    return result


def run_native(metadata, output, case, seed, record):
    from glm53_runtime_jit import activate_triton
    from glm53_kda_replay import reject_retuning
    class NoKernelCache:
        def __init__(self, *args, **kwargs): raise ValueError('component load forbids Triton specialization')
    activate_triton(NoKernelCache, enabled=True); reject_retuning(enabled=True)
    from vllm.transformers_utils.configs.glm5_next import Glm5NextConfig
    text = Glm5NextConfig(**strict_json(metadata / 'config.json')).text_config
    require(text.hidden_size == 4096 and text.n_routed_experts == 288 and text.moe_intermediate_size == 2048 and
            text.vocab_size == 154880 and text.num_hidden_layers == 45, 'model sizing metadata changed')
    import torch
    require(torch.cuda.get_device_capability() == (12, 1) and
            torch.cuda.get_device_properties(0).multi_processor_count == 48, '48-SM GB10 SM121 required')
    record({'event': 'configured', 'case': case, 'selection': 'persistent_pinned_stream', 'triton': 'all_specializations_rejected',
            'retuning': 'rejected', 'minimal_MoE_context': case == 'moe', 'pinned_capacity': CAPACITY, 'multiprocessors': 48})
    fixture = write_fixture(output / 'fixture', case, seed)
    (output / 'fixture.json').write_text(json.dumps(fixture, indent=2) + '\n')
    capture = Capture()
    record({'event': 'memory', 'phase': 'before_constructor', **memory()})
    torch.cuda.reset_peak_memory_stats()
    layer = construct(case); torch.cuda.synchronize()
    parameters = {name: describe(tensor) for name, tensor in layer.named_parameters()}
    record({'event': 'constructed', 'parameters': parameters, 'memory': memory()})
    specs = tensor_specs(case); mapping = {s['name']: s for s in specs}
    def consume(name, temporary):
        spec = mapping[name]
        consume_weight(layer, case, spec, temporary)
        view = loaded_view(layer, spec, case)
        digest = capture.digest(view)
        require(digest == fixture['selection'][name]['sha256'], 'loaded parameter bytes changed')
        record({'event': 'loaded', 'name': name, 'sha256': digest, 'storage': describe(view)})
    from glm53_pinned_stream import stream_selected_weights
    torch.cuda.reset_peak_memory_stats()
    stream_selected_weights(output / 'fixture', fixture['inventory'], fixture['selection'], consume,
                            lambda row: record({'event': 'transfer', **row}), enabled=True)
    torch.cuda.synchronize()
    record({'event': 'memory', 'phase': 'after_transfers_and_verification', **memory()})
    torch.cuda.reset_peak_memory_stats()
    if case != 'ordinary': layer.quant_method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    record({'event': 'memory', 'phase': 'after_finalization', **memory()})
    if case in ('kda', 'mla'):
        require(all(not hasattr(layer, name) for name in parameter_layout(case)), 'fused staging attributes survived')
        handles = layer._exl3_linears
        require(len(handles) == (6 if case == 'kda' else 2) and
                all(value is None for value in handles[3:]) and all(value is not None for value in handles[:3 if case == 'kda' else 2]), 'dense final handle coverage mismatch')
    for spec in specs:
        view = loaded_view(layer, spec, case, final=True)
        digest = capture.digest(view)
        require(digest == fixture['selection'][spec['name']]['sha256'], 'finalized handle bytes changed')
        record({'event': 'final_bytes', 'name': spec['name'], 'sha256': digest, 'storage': describe(view)})
    del view
    record({'event': 'retained', **retained_buffers(layer, case, capture), 'memory': memory()})


def contiguous_stride(shape):
    stride, value = [], 1
    for dimension in reversed(shape): stride.insert(0, value); value *= max(dimension, 1)
    return stride


def validate_storage(row):
    keys = {'device', 'dtype', 'shape', 'stride', 'storage_pointer', 'storage_bytes', 'data_pointer', 'storage_offset'}
    require(isinstance(row, dict) and set(row) == keys and row['device'] == 'cuda:0', 'invalid GPU storage schema')
    sizes = {**SIZES, 'torch.int64': 8}
    require(isinstance(row['dtype'], str) and row['dtype'] in sizes and isinstance(row['shape'], list) and isinstance(row['stride'], list) and
            len(row['shape']) == len(row['stride']) and all(type(v) is int and v >= 0 for v in row['shape'] + row['stride']), 'invalid storage dimensions')
    require(all(type(row[k]) is int and row[k] >= 0 for k in ('storage_pointer', 'storage_bytes', 'data_pointer', 'storage_offset')), 'invalid storage address')
    if math.prod(row['shape']):
        size = sizes[row['dtype']]
        require(row['storage_pointer'] > 0 and row['data_pointer'] == row['storage_pointer'] + size * row['storage_offset'] and
                (row['storage_offset'] + sum((n - 1) * s for n, s in zip(row['shape'], row['stride'])) + 1) * size <= row['storage_bytes'], 'storage view escapes backing')
    else: require(row['data_pointer'] == 0 and row['storage_bytes'] == 0, 'invalid empty storage')


def expected_loaded_storage(case, spec, parameters):
    result = dict(parameters[spec['parameter']]); shape = spec['shape']; shard = spec['shard_id']; part = spec['parameter']
    offset = 0
    if case == 'moe':
        offset = spec['expert_id'] * result['stride'][0]
        if shard != 'w2': offset += (0 if shard == 'w1' else 1) * result['stride'][1]
        stride = contiguous_stride(shape)
    elif case == 'ordinary': stride = result['stride']
    else:
        outputs = [8192, 8192, 8192, 64, 128, 128] if case == 'kda' else [1536, 512]
        if part == 'trellis': offset = sum(outputs[:shard]) // 16 * 64; stride = result['stride']
        elif part == 'svh': offset = sum(outputs[:shard]); stride = [1]
        elif part == 'suh': offset = shard * 4096; stride = [1]
        elif part == 'weight': offset = sum(outputs[3:shard]) * 4096; stride = [4096, 1]
        else: offset = shard; stride = []
    return {**result, 'shape': shape, 'stride': stride, 'storage_offset': offset,
            'data_pointer': result['storage_pointer'] + offset * SIZES[result['dtype']]}


def validate_memory(value):
    require(isinstance(value, dict) and set(value) == {'process_kib', 'cuda_allocated', 'cuda_reserved',
            'cuda_peak_allocated', 'cuda_peak_reserved', 'pinned_allocator'}, 'memory schema mismatch')
    require(set(value['process_kib']) == {'Rss', 'Pss', 'Pss_Anon', 'Pss_File'} and
            all(type(v) is int and v >= 0 for v in value['process_kib'].values()) and
            value['process_kib']['Rss'] >= value['process_kib']['Pss'] >= value['process_kib']['Pss_Anon'] > 0 and
            value['process_kib']['Pss'] >= value['process_kib']['Pss_File'], 'invalid RSS/PSS')
    require(all(type(value[k]) is int and value[k] >= 0 for k in ('cuda_allocated', 'cuda_reserved', 'cuda_peak_allocated', 'cuda_peak_reserved')) and
            value['cuda_allocated'] <= value['cuda_reserved'] <= value['cuda_peak_reserved'] and
            value['cuda_allocated'] <= value['cuda_peak_allocated'] <= value['cuda_peak_reserved'], 'invalid CUDA allocation counters')
    pinned = value['pinned_allocator']
    require(set(pinned) == {f'{group}.{kind}' for group in ('allocated_bytes', 'active_bytes', 'allocations', 'active_requests')
                           for kind in ('current', 'peak', 'allocated', 'freed')} and
            all(type(v) is int and v >= 0 for v in pinned.values()) and
            pinned['allocated_bytes.current'] >= pinned['active_bytes.current'] >= CAPACITY, 'invalid pinned allocation counters')


def validate_retained_geometry(retained, case):
    if case == 'moe': require(type(retained['concurrency']) is int and retained['concurrency'] == 6, 'invalid scratch concurrency')
    for key in ('shared_scratch', 'tensor_cache'):
        expected_shapes = ([[6, 2048, 4096]] * 2 + [[6, 2048, 2048]] * 2 if case == 'moe' else []) if key == 'shared_scratch' else (
            [[1, 2048], [1, 4096]] if case == 'moe' else [[1, 4096]] if case in ('kda', 'mla') else [])
        require(sorted(r['shape'] for r in retained[key]) == sorted(expected_shapes), 'retained cache/scratch geometry mismatch')
        for row in retained[key]:
            require(row['dtype'] == 'torch.float16' and row['stride'] == contiguous_stride(row['shape']) and
                    row['storage_offset'] == 0 and row['storage_bytes'] == math.prod(row['shape']) * 2,
                    'retained cache/scratch layout mismatch')



def score_capture(root, rows, case, seed):
    specs = tensor_specs(case); names = [s['name'] for s in specs]
    digests, canonical_inventory = canonical_fixture(case, seed)
    expected = {s['name']: digests[s['name']] for s in specs}
    fixture = strict_json(root / 'fixture.json')
    require(set(fixture) == {'schema_version', 'qualification', 'case', 'seed', 'generator_sha256', 'inventory', 'selection', 'excluded'} and
            type(fixture['schema_version']) is int and fixture['schema_version'] == 1 and
            fixture['qualification'] == 'generated_synthetic_input_only' and fixture['case'] == case and
            type(fixture['seed']) is int and fixture['seed'] == seed and
            fixture['generator_sha256'] == sha256_file(ROOT / 'scripts/lib/glm53_load_fixture.py'), 'fixture binding mismatch')
    require(fixture['selection'] == {s['name']: {'file': 'weights.safetensors', 'dtype': DTYPES[s['dtype']],
            'shape': s['shape'], 'sha256': expected[s['name']]} for s in specs}, 'fixture selection differs from independent generator')
    require(fixture['excluded'] == {s['name']: {'dtype': DTYPES[s['dtype']], 'shape': s['shape'],
            'sha256': digests[s['name']]} for s in EXCLUDED}, 'excluded fixture binding mismatch')
    require(fixture['inventory'] == canonical_inventory, 'retained fixture is not the canonical complete file')
    verify_inventory(root / 'fixture', fixture['inventory'])
    require({p.name for p in root.iterdir()} == {'manifest.json', 'summary.json', 'raw.jsonl', 'traceback.log', 'fixture.json', 'fixture'}, 'load artifact coverage mismatch')
    require([r.get('event') for r in rows] == ['configured', 'memory', 'constructed'] +
            ['loaded', 'transfer'] * len(specs) + ['memory', 'memory'] + ['final_bytes'] * len(specs) + ['retained'], 'load event coverage mismatch')
    require(rows[0] == {'time_unix': rows[0]['time_unix'], 'event': 'configured', 'case': case,
            'selection': 'persistent_pinned_stream', 'triton': 'all_specializations_rejected', 'retuning': 'rejected',
            'minimal_MoE_context': case == 'moe', 'pinned_capacity': CAPACITY, 'multiprocessors': 48} and
            type(rows[0]['minimal_MoE_context']) is bool and type(rows[0]['pinned_capacity']) is int and type(rows[0]['multiprocessors']) is int, 'load startup selection mismatch')
    phases = [r for r in rows if r['event'] == 'memory']
    require([r['phase'] for r in phases] == ['before_constructor', 'after_transfers_and_verification', 'after_finalization'], 'load memory phase mismatch')
    for row in phases:
        validate_memory({k: v for k, v in row.items() if k not in ('time_unix', 'event', 'phase')})
    constructed = rows[2]
    require(set(constructed) == {'time_unix', 'event', 'parameters', 'memory'}, 'constructor schema mismatch')
    parameters = constructed['parameters']; layout = parameter_layout(case)
    require(set(parameters) == set(layout), 'constructor parameter coverage mismatch')
    for name, row in parameters.items():
        validate_storage(row)
        require({k: row[k] for k in ('dtype', 'shape')} == layout[name] and row['stride'] == contiguous_stride(row['shape']) and
                row['storage_offset'] == 0 and row['storage_bytes'] == math.prod(row['shape']) * SIZES[row['dtype']], 'constructor storage layout mismatch')
    pointers = [r['storage_pointer'] for r in parameters.values() if r['storage_bytes']]
    require(len(set(pointers)) == len(pointers), 'constructor unexpectedly aliases parameters')
    validate_memory(constructed['memory'])
    validate_byte_coverage(rows, expected)
    loaded = {r['name']: r for r in rows if r['event'] == 'loaded'}
    final = {r['name']: r for r in rows if r['event'] == 'final_bytes'}
    transfers = [r for r in rows if r['event'] == 'transfer']
    require([r['name'] for r in transfers] == names and
            [r['name'] for r in rows if r['event'] == 'loaded'] == names and
            [r['name'] for r in rows if r['event'] == 'final_bytes'] == names, 'load tensor order mismatch')
    staging = None; reuses = 0; new_trees = set(); clones = set()
    for spec, transfer in zip(specs, transfers):
        name = spec['name']; size = byte_count(spec)
        require(set(transfer) == {'time_unix', 'event', 'name', 'bytes', 'source_sha256', 'device_sha256',
                'staging_pointer', 'staging_bytes', 'pinned', 'upload_chunks', 'completed_reuses', 'temporary_bytes'} and
                transfer['pinned'] is True and all(type(transfer[k]) is int for k in ('bytes', 'staging_pointer',
                    'staging_bytes', 'upload_chunks', 'completed_reuses', 'temporary_bytes')) and
                transfer['bytes'] == size and transfer['temporary_bytes'] == size and transfer['staging_bytes'] == CAPACITY,
                'invalid staged transfer record')
        chunks = (size + CAPACITY - 1) // CAPACITY; reuses += chunks * 2
        require(transfer['upload_chunks'] == chunks and transfer['completed_reuses'] == reuses and transfer['staging_pointer'] > 0,
                'staging completion or chunk coverage mismatch')
        if staging is None: staging = transfer['staging_pointer']
        require(staging == transfer['staging_pointer'], 'pinned staging was not persistent')
        for row in (loaded[name], final[name]):
            require(set(row) == {'time_unix', 'event', 'name', 'sha256', 'storage'}, 'loaded/final byte schema mismatch')
            validate_storage(row['storage'])
        before, after = loaded[name]['storage'], final[name]['storage']
        require(before == expected_loaded_storage(case, spec, parameters), 'loaded parameter routing or offset mismatch')
        if case in ('moe', 'ordinary') or spec['parameter'] not in ('trellis', 'weight'):
            require(after == before, 'final handle lost packed backing alias')
        else:
            require(after['storage_pointer'] != before['storage_pointer'] and after['dtype'] == before['dtype'] and
                    after['shape'] == before['shape'], 'finalized copy missing or geometry changed')
            if spec['parameter'] == 'trellis':
                require(after['storage_bytes'] == size and after['storage_offset'] == 0 and after['stride'] == contiguous_stride(spec['shape']), 'invalid finalized trellis copy')
                new_trees.add(after['storage_pointer'])
            else:
                require(after['storage_bytes'] == 2621440 and after['storage_offset'] == before['storage_offset'] and
                        after['stride'] == before['stride'], 'invalid BF16 clone')
                clones.add(after['storage_pointer'])
    require(not (new_trees & clones) and not ((new_trees | clones) & set(pointers)), 'final copy aliases other storage')
    require(len(new_trees) == (3 if case == 'kda' else 2 if case == 'mla' else 0) and
            len(clones) == (1 if case == 'kda' else 0), 'final copy storage coverage mismatch')
    retained = rows[-1]
    require(set(retained) == {'time_unix', 'event', 'pointer_tables', 'shared_scratch', 'tensor_cache', 'memory', 'handles',
                             *(['concurrency'] if case == 'moe' else [])}, 'retained buffer schema mismatch')
    require(sorted(retained['handles'], key=lambda r: json.dumps(r, sort_keys=True)) == expected_handles(case) and
            all(type(r['bits']) is int and type(r['mcg']) is bool and type(r['mul1']) is bool and
                type(r['input']) is int and type(r['output']) is int for r in retained['handles']), 'native handle record mismatch')
    validate_memory(retained['memory'])
    for key in ('shared_scratch', 'tensor_cache'):
        require(isinstance(retained[key], list), 'invalid retained storage list')
        for row in retained[key]: validate_storage(row)
    tables = retained['pointer_tables']
    require(set(tables) == ({f'{p}_{s}' for p in ('gate', 'up', 'down') for s in ('trellis', 'suh', 'svh')} if case == 'moe' else set()), 'pointer table coverage mismatch')
    if case == 'moe':
        for key, row in tables.items():
            projection, part = key.split('_', 1); suffix = {'gate': 'gate_proj', 'up': 'up_proj', 'down': 'down_proj'}[projection]
            require(set(row) == {'storage', 'values', 'alias'} and row['alias'] == projection + '_' + ('t' if part == 'trellis' else part) + '_ptrs', 'pointer alias schema mismatch')
            validate_storage(row['storage'])
            require(row['storage']['dtype'] == 'torch.int64' and row['storage']['shape'] == [288] and
                    row['storage']['stride'] == [1] and row['storage']['storage_offset'] == 0 and
                    row['storage']['storage_bytes'] == 2304 and row['values'] == [
                        final[f'model.language_model.layers.3.mlp.experts.{e}.{suffix}.{part}']['storage']['data_pointer'] for e in range(288)] and
                    all(type(v) is int for v in row['values']), 'pointer table values mismatch')
        require(len({r['storage']['storage_pointer'] for r in tables.values()}) == 9, 'pointer tables alias')
        require(type(retained['concurrency']) is int and retained['concurrency'] == 6, 'invalid scratch concurrency')
        unique = {(r['device'], r['storage_pointer']): r['storage_bytes'] for r in retained['shared_scratch']}
        require(sum(unique.values()) == 50331648 * retained['concurrency'], 'scratch size mismatch')
    known = [r['storage'] for r in final.values()]
    if case == 'moe': known.extend(parameters.values())
    validate_retained_geometry(retained, case)
    known.extend(retained['shared_scratch']); known.extend(retained['tensor_cache'])
    known.extend(r['storage'] for r in tables.values())
    # Repeated views must agree on backing size; independent retained allocations cannot overlap.
    unique = {}
    for row in known:
        if row['storage_bytes']:
            pointer = row['storage_pointer']
            require(pointer not in unique or unique[pointer] == row['storage_bytes'], 'contradictory backing sizes')
            unique[pointer] = row['storage_bytes']
    independent = [r['storage_pointer'] for key in ('shared_scratch', 'tensor_cache') for r in retained[key]] + [r['storage']['storage_pointer'] for r in tables.values()]
    parameter_pointers = {r['storage_pointer'] for r in known[:len(final) + (len(parameters) if case == 'moe' else 0)]}
    require(len(set(independent)) == len(independent) and not (set(independent) & parameter_pointers), 'retained allocations alias')
    allocations = sorted(unique.items())
    require(all(p + size <= q for (p, size), (q, _) in zip(allocations, allocations[1:])), 'retained allocations overlap')
    baseline = phases[0]['cuda_allocated']; constructor = layout_bytes(layout)
    transfer_peak = constructor + max(byte_count(spec) * (2 if case in ('kda', 'mla') and spec['parameter'] == 'trellis' else 1) for spec in specs)
    current_final = sum(unique.values())
    finalization_peak = max(current_final, constructor + sum(r['storage_bytes'] for r in retained['tensor_cache']) +
        sum(next(r['storage']['storage_bytes'] for r in final.values() if r['storage']['storage_pointer'] == pointer)
            for pointer in new_trees | clones))
    for observation, current_min, peak_min in ((constructed['memory'], constructor, constructor),
            (phases[1], constructor, transfer_peak), (phases[2], current_final, finalization_peak),
            (retained['memory'], current_final, finalization_peak)):
        require(observation['cuda_allocated'] >= baseline + current_min and
                observation['cuda_peak_allocated'] >= baseline + peak_min, 'CUDA counters contradict live storage or required overlap')
    require(phases[1]['pinned_allocator']['active_bytes.peak'] >= 2 * CAPACITY, 'missing simultaneous pinned buffers')
    return {'case': case, 'tensors_checked': len(specs), 'bytes_checked_per_stage': sum(byte_count(s) for s in specs),
            'constructor_parameter_bytes': layout_bytes(layout), 'retained_unique_bytes': current_final,
            'minimum_transfer_peak_bytes': transfer_peak, 'minimum_finalization_peak_bytes': finalization_peak, 'new_trellis_storages': len(new_trees), 'bf16_clone_storages': len(clones),
            'model_loaded': False, 'full_load_fit': 'not measured'}


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=('moe', 'kda', 'mla', 'ordinary'), required=True)
    parser.add_argument('--pinned-stream', action='store_true')
    parser.add_argument('--seed', type=int, required=True); args = parser.parse_args()
    require(args.pinned_stream, 'explicit --pinned-stream selection required')
    require(type(args.seed) is int and 0 <= args.seed < 2**64, 'invalid fixture seed')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'qualification': QUALIFICATION, 'seed': args.seed, 'case': args.case,
        'scorer_sha256': sha256_file(Path(__file__)), 'binary_sha256': sha256_file(Path(sys.executable).resolve()),
        'decision': {'sha256': sha256_file(ROOT / 'configs/decision-specs/glm53-load-preflight.json')},
        'metadata': {p.name: {'sha256': sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()}, 'start_unix': time.time()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    failure = None
    with (args.output / 'raw.jsonl').open('w') as raw, (args.output / 'traceback.log').open('w') as errors:
        def record(row):
            raw.write(json.dumps({'time_unix': time.time(), **row}, allow_nan=False) + '\n'); raw.flush()
        try: run_native(args.metadata, args.output, args.case, args.seed, record)
        except Exception as error:
            failure = repr(error); traceback.print_exc(file=errors); record({'event': 'failure', 'failure': failure})
    summary = {'verdict': 'FAIL' if failure else 'PASS', 'failure': failure, 'qualification': QUALIFICATION,
        'case': args.case, 'model_loaded': False, 'actual_input_tokens_processed': 0,
        'context_capability': 'not measured', 'performance': 'not measured', 'raw_sha256': sha256_file(args.output / 'raw.jsonl')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary))
    raise SystemExit(1 if failure else 0)


if __name__ == '__main__': main()
