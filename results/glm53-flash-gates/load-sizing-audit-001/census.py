#!/usr/bin/env python3
"""Read verified headers and pinned source; no Torch, network or tensor allocation.

This counts tensor payloads and selected constructor storage formulas. It does
not load weights, initialize an engine, measure residency, or authorize a load.
Print deterministic JSON; the caller may retain stdout as census.json.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CACHE = Path('/home/bmarti44/.cache/glm53-flash')
DTYPE_BYTES = {'BF16': 2, 'F16': 2, 'F32': 4, 'I16': 2, 'I32': 4}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def size(meta):
    shape = meta['shape']
    require(all(type(d) is int and d >= 0 for d in shape), 'invalid tensor shape')
    result = math.prod(shape) * DTYPE_BYTES[meta['dtype']]
    if 'data_offsets' in meta:
        start, end = meta['data_offsets']
        require(type(start) is int and type(end) is int and 0 <= start <= end and end - start == result,
                'header shape/offset disagreement')
    return result


def layer(name):
    match = re.match(r'model\.language_model\.layers\.([0-9]+)\.', name)
    return int(match[1]) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-available-gib', type=int, default=115)
    args = parser.parse_args()
    require(args.start_available_gib > 40, 'invalid starting availability')
    pins = read_json(HERE / 'source-pins.json')
    for item in pins['files']:
        require(sha(item['path']) == item['sha256'], 'pinned input changed: ' + item['path'])
    archive = REPO / 'results/glm53-flash-gates/model-layout-001'
    metadata = CACHE / 'model-layout-001'
    inventory = read_json(archive / 'inventory.json')['files']
    # All archived metadata files, including both packs' headers, are verified.
    for row in inventory:
        path = metadata / row['path']
        require(path.is_file() and not path.is_symlink() and path.stat().st_size == row['size_bytes']
                and sha(path) == row['sha256'], 'archived metadata changed: ' + row['path'])
    tensors = {}
    tensor_files = {}
    index = read_json(metadata / 'k2/model.safetensors.index.json')['weight_map']
    headers = sorted((metadata / 'k2').glob('*.header.json'))
    require(len(headers) == 120, 'K2 shard coverage changed')
    shard_payloads = []
    for path in headers:
        total = 0
        filename = path.name.removesuffix('.header.json')
        for name, value in read_json(path).items():
            if name == '__metadata__':
                continue
            require(name not in tensors and index.get(name) == filename, 'tensor index mismatch')
            size(value)
            tensors[name] = value
            tensor_files[name] = filename
            total += size(value)
        shard_payloads.append({'file': filename, 'payload_bytes': total})
    require(set(tensors) == set(index), 'tensor coverage changed')

    config = read_json(CACHE / 'processor-metadata-002/config.json')['text_config']
    require(config['num_hidden_layers'] == 45 and config['num_nextn_predict_layers'] == 1,
            'main/MTP layer configuration changed')
    model_source = CACHE / 'build-source-005/vllm/vllm/models/glm5next/nvidia/model.py'
    function = next(n for n in ast.parse(model_source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'get_spec_layer_idx_from_weight_name')
    namespace = {'Glm5NextConfig': object}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(model_source), 'exec'), namespace)
    cfg = SimpleNamespace(**{k: config[k] for k in ('num_hidden_layers', 'num_nextn_predict_layers')})
    skip = namespace[function.name]
    require(skip(cfg, 'layers.44.mlp.gate_proj.weight') is None and
            skip(cfg, 'layers.45.mlp.gate_proj.weight') == 45 and
            skip(cfg, 'model.layers.45.mlp.gate_proj.weight') == 45, 'pinned main-loader MTP skip changed')

    plan = read_json(archive / 'overlay-plan.json')['plan']
    require(len(plan) == 315 and len({e['name'] for e in plan}) == 315, 'overlay coverage changed')
    overlay_source = CACHE / 'build-source-005/vllm-exl3/tools/dense_overlay.py'
    source_tree = ast.parse(overlay_source.read_text())
    plan_function = next(n for n in source_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'plan_outputs')
    namespace = {}
    exec(compile(ast.Module(body=[plan_function], type_ignores=[]), str(overlay_source), 'exec'), namespace)
    outputs = namespace['plan_outputs'](plan)
    base_bytes = sum(size(v) for v in tensors.values())
    removed = {e['name'] for e in plan}
    require(all(layer(name) is not None and layer(name) < 45 for name in removed), 'overlay modifies MTP or other tensors')
    removed_bytes = sum(size(tensors[name]) for name in removed)
    mtp_bytes = sum(size(v) for name, v in tensors.items() if layer(name) == 45)
    selected = {name: value for name, value in tensors.items() if name not in removed and layer(name) != 45}
    for name, dtype, shape, _, _ in outputs:
        require(name not in selected, 'overlay duplicate tensor')
        selected[name] = {'dtype': dtype, 'shape': shape}
    overlay_bytes = sum(size({'dtype': dtype, 'shape': shape}) for _, dtype, shape, _, _ in outputs)
    resident_payload = sum(size(v) for v in selected.values())
    require(resident_payload == base_bytes - removed_bytes + overlay_bytes - mtp_bytes, 'payload arithmetic mismatch')
    previous = read_json(archive / 'summary.json')
    require(removed_bytes == previous['replaced_bf16_bytes'] and overlay_bytes == previous['overlay_data_bytes'],
            'overlay accounting differs from prior archive')

    roles = Counter(); layers = Counter(); dtypes = Counter(); expert_parts = Counter()
    for name, value in selected.items():
        n = size(value); dtypes[value['dtype']] += n
        if layer(name) is not None:
            layers[layer(name)] += n
            role = 'routed_experts' if '.mlp.experts.' in name else 'non_routed_language_layers'
        elif name.startswith('model.visual.'):
            role = 'vision'
        elif name == 'lm_head.weight':
            role = 'lm_head'
        elif name.endswith('embed_tokens.weight'):
            role = 'embedding'
        else:
            role = 'other_language_weights'
        roles[role] += n
        if role == 'routed_experts': expert_parts[name.rsplit('.', 1)[1]] += n

    # Replay the EXL3 linear constructor's integer storage formulas. This is a
    # named-parameter census, not an allocation or C++ object residency claim.
    fork = next(n for n in source_tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'FORK' for t in n.targets))
    fork = ast.literal_eval(fork.value)
    modules = defaultdict(dict)
    for entry in plan:
        index = layer(entry['name']); suffix = entry['base'].split(f'layers.{index}.', 1)[1]
        module, shard = fork[suffix]
        modules[(index, module)][0 if shard is None else shard] = entry
    constructor_rows = []
    for (index, module), shards in sorted(modules.items()):
        exemplar = next(iter(shards.values())); bits = exemplar['k']; in_features = exemplar['in']
        out_sizes = {i: e['out'] for i, e in shards.items()}
        require(all(e['in'] == in_features and e['k'] == bits for e in shards.values()), 'inconsistent fused geometry')
        bf16 = {}
        if module == 'self_attn.in_proj_qkvbfg_a':
            for i, suffix in ((3, 'b_proj'), (4, 'f_a_proj'), (5, 'g_a_proj')):
                item = tensors[f'model.language_model.layers.{index}.self_attn.{suffix}.weight']
                require(item['shape'][1] == in_features, 'BF16 mixed shard input changed')
                out_sizes[i] = item['shape'][0]; bf16[i] = item['shape'][0]
        require(set(out_sizes) == set(range(len(out_sizes))), 'missing fused shard')
        padded_in = (in_features + 127) // 128 * 128
        padded_out = {i: n if i in bf16 else (n + 127) // 128 * 128 for i, n in out_sizes.items()}
        trellis = padded_in // 16 * (sum(padded_out.values()) // 16) * bits * 16 * 2
        signs = len(out_sizes) * padded_in * 2 + sum(padded_out.values()) * 2
        markers = len(out_sizes) * 8
        bf16_bytes = sum(bf16.values()) * padded_in * 2
        copied_trellis = (sum(padded_in // 16 * (padded_out[i] // 16) * bits * 16 * 2 for i in shards)
                          if len(out_sizes) > 1 and padded_in // 16 > 1 else 0)
        constructor_rows.append({'layer': index, 'module': module,
            'constructor_bytes': trellis + signs + markers + bf16_bytes,
            'bf16_staging_bytes': bf16_bytes,
            'temporary_finalization_copy_bytes': copied_trellis + bf16_bytes})

    cache = read_json(REPO / 'configs/decision-specs/glm53-cache-preflight.json')['expected']['unique_backing_bytes']
    capacity40 = (args.start_available_gib - 40) * 2**30
    report = {'schema_version': 1, 'qualification': 'metadata_only_load_sizing',
        'model_loaded': False, 'weights_downloaded': False, 'cuda_used': False,
        'census_sha256': sha(__file__), 'source_pins_sha256': sha(HERE / 'source-pins.json'),
        'base_tensor_count': len(tensors), 'base_tensor_bytes': base_bytes,
        'excluded_MTP_layer': 45, 'excluded_MTP_bytes': mtp_bytes,
        'overlay_replaced_bytes': removed_bytes, 'overlay_added_bytes': overlay_bytes,
        'selected_tensor_count': len(selected), 'selected_payload_bytes': resident_payload,
        'selected_payload_GiB': resident_payload / 2**30,
        'selected_bytes_by_role': dict(sorted(roles.items())), 'selected_bytes_by_dtype': dict(sorted(dtypes.items())),
        'selected_routed_bytes_by_suffix': dict(sorted(expert_parts.items())),
        'selected_layer_payload_bytes': {str(k): v for k, v in sorted(layers.items())},
        'largest_selected_tensors': [{'name': name, 'bytes': size(value), 'shape': value['shape'], 'dtype': value['dtype']}
            for name, value in sorted(selected.items(), key=lambda item: size(item[1]), reverse=True)[:8]],
        'largest_base_shard_payloads': sorted(shard_payloads, key=lambda row: row['payload_bytes'], reverse=True)[:5],
        'overlay_linear_constructor_census': {'assumptions': 'TP1; BF16 shard geometry flag=1; BF16 parameter dtype',
            'module_count': len(constructor_rows), 'constructor_bytes': sum(r['constructor_bytes'] for r in constructor_rows),
            'BF16_staging_bytes': sum(r['bf16_staging_bytes'] for r in constructor_rows),
            'largest_temporary_finalization_copies': sorted(constructor_rows,
                key=lambda row: row['temporary_finalization_copy_bytes'], reverse=True)[:5]},
        'forty_GiB_floor': {'verdict': 'NO_GO' if resident_payload > capacity40 else 'NO_RESULT',
            'scope': 'current full-resident representation, cache disabled; payload sizing only',
            'starting_MemAvailable_GiB_assumption': args.start_available_gib,
            'incremental_budget_bytes': capacity40, 'payload_excess_bytes': resident_payload - capacity40},
        'eighteen_GiB_floor': {'verdict': 'NO_RESULT', 'physical_cache_bytes': cache,
            'payload_plus_cache_bytes': resident_payload + cache,
            'unmeasured_overhead_budget_bytes': (args.start_available_gib - 18) * 2**30 - resident_payload - cache,
            'excluded_costs': ['engine baseline', 'loading/finalization peaks', 'pinned staging', 'allocator slack',
                               'retained native handles and scratch', 'live activations/vision workspaces', 'outside-cgroup pressure']}}
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
