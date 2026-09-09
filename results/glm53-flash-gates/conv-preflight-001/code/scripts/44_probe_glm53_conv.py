#!/usr/bin/env python3
"""Default-off, model-free convolution oracle; JIT preparation is not qualification."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json

QUALIFICATION = 'model_free_convolution_analytic_falsifier_only'
CHANNELS = 24576
CASES = {'decode-1': [1], 'decode-2': [1, 1], 'decode-3': [1, 1, 1], 'decode-4': [1, 1, 1, 1],
         'prefill-1': [2048], 'prefill-4': [512] * 4, 'prefill-ragged': [1, 7, 511, 1529],
         'prefill-ragged-fresh': [1, 7, 511, 1529]}


def require(value, message):
    if not value: raise ValueError(message)


def case_order(seed):
    require(type(seed) is int and 0 <= seed < 2**64, 'invalid convolution seed')
    order = list(CASES); random.Random(seed).shuffle(order); return order


def slots(seed):
    case_order(seed)
    ids = [1, 2, 3, 4]; random.Random(seed ^ 0x434F4E56).shuffle(ids); return ids


def bf16_bytes(array):
    import numpy as np
    bits = np.asarray(array, dtype='<f4').view('<u4')
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16).astype('<u2').tobytes()


def fixture(seed, case, channels=CHANNELS):
    import numpy as np
    case_order(seed); require(case in CASES and type(channels) is int and 0 < channels <= CHANNELS, 'invalid convolution geometry')
    lengths = CASES[case]; ids = slots(seed)[:len(lengths)]
    initial_flags = [True if case.startswith('decode') else ((i % 2 == 0) != case.endswith('-fresh')) for i in range(len(lengths))]
    c = np.arange(channels)[None, :]
    def values(slot, t): return ((4 * slot + c % 16 - 8 + (np.asarray(t)[:, None] + seed % 7) % 7) / 32).astype('<f4')
    source = np.full((sum(lengths), channels + 320), 3.25, dtype='<f4')
    offset = 0
    for slot, length in zip(ids, lengths):
        source[offset:offset + length, :channels] = values(slot, np.arange(length)); offset += length
    initial = np.full((5, 3, channels), 3.25, dtype='<f4')
    for slot in range(1, 5): initial[slot] = values(slot, [-3, -2, -1])
    weight = (((np.arange(channels)[:, None] + np.arange(4)[None, :] + seed % 5) % 5 - 2) / 8).astype('<f4')
    return source, weight, initial, ids, initial_flags


def expected(seed, case, channels=CHANNELS):
    """Independent array convolution: raw-history concatenation, then four shifts."""
    import numpy as np
    source, weight, final, ids, flags = fixture(seed, case, channels)
    output = np.empty((len(source), channels), dtype='<f4'); offset = 0
    for slot, length, flag in zip(ids, CASES[case], flags):
        history = final[slot] if flag else np.zeros((3, channels), dtype='<f4')
        joined = np.concatenate((history, source[offset:offset + length, :channels]))
        z = np.zeros((length, channels), dtype='<f4')
        for tap in range(4): z += joined[tap:tap + length] * weight[:, tap]
        output[offset:offset + length] = (z.astype('float64') / (1 + np.exp(-z.astype('float64')))).astype('<f4')
        final[slot] = joined[-3:]; offset += length
    return output, final


def prior_api():
    spec = importlib.util.spec_from_file_location('qualified_kda_reader', ROOT / 'scripts/41_probe_glm53_kda.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def score_tensors(paths, hashes, seed, case, channels=CHANNELS):
    import numpy as np
    require(len(paths) == len(hashes) == 3, 'convolution artifact coverage')
    count = sum(CASES[case]); sizes = [count * channels * 2, 5 * 3 * channels * 2, count * (channels + 320) * 2]
    read = prior_api().read_tensor
    output, state, post = [read(p, h, size) for p, h, size in zip(paths, hashes, sizes)]
    actual = (np.frombuffer(output, dtype='<u2').astype('<u4') << 16).view('<f4').reshape(count, channels)
    reference, final = expected(seed, case, channels)
    errors = np.abs(actual - reference)
    mismatches = int(np.count_nonzero(~np.isfinite(actual) | (errors > 0.001 + 0.01 * np.abs(reference))))
    require(mismatches == 0, 'convolution output violates analytic reference')
    require(state == bf16_bytes(final), 'convolution state bytes differ')
    source = fixture(seed, case, channels)[0]
    expected_input = bytearray(bf16_bytes(source))
    if case.startswith('decode'):
        for i in range(count):
            expected_input[i * (channels + 320) * 2:i * (channels + 320) * 2 + channels * 2] = output[i * channels * 2:(i + 1) * channels * 2]
    require(post == expected_input, 'convolution input/guard bytes differ')
    return {'elements': actual.size, 'maximum_absolute_error': float(errors.max()), 'mismatched_elements': mismatches,
            'state_bytes_checked': sizes[1], 'input_bytes_checked': sizes[2], 'nonfinite_elements': 0}


def geometry():
    return {'channels': CHANNELS, 'projected_width': CHANNELS + 320, 'state_shape': [5, 3, CHANNELS],
            'state_view_strides': [3 * CHANNELS, 1, CHANNELS], 'input_dtype': 'torch.bfloat16',
            'weight_dtype': 'torch.float32', 'pinned_staging_scope': 'probe_buffers_and_retained_metadata'}


def score_capture(root, rows, seed):
    order = case_order(seed)
    times = [r.get('time_unix') for r in rows]
    require(all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in times) and
            all(b > a for a, b in zip(times, times[1:])), 'convolution timestamp mismatch')
    require(len(rows) == 1 + 2 * len(order), 'convolution row coverage')
    require(rows[0] == {'time_unix': rows[0].get('time_unix'), 'event': 'configured', 'case_order': order,
                        'slots': slots(seed), 'pinned_staging': True, **geometry()} and rows[0]['pinned_staging'] is True,
            'convolution startup geometry or pinning mismatch')
    require(all(type(v) is int for v in rows[0]['slots'] + rows[0]['state_shape'] + rows[0]['state_view_strides']) and
            type(rows[0]['channels']) is int and type(rows[0]['projected_width']) is int, 'convolution geometry types')
    artifacts = {f'{kind}-{case}.bf16.gz' for case in order for kind in ('output', 'state', 'input')}
    require({p.name for p in root.iterdir()} == artifacts | {'manifest.json', 'summary.json', 'raw.jsonl', 'traceback.log'},
            'convolution file coverage mismatch')
    checks = []
    for index, case in enumerate(order):
        start, row = rows[1 + 2 * index:3 + 2 * index]; count = sum(CASES[case]); decode = case.startswith('decode')
        require(start == {'time_unix': start.get('time_unix'), 'event': 'start', 'case': case}, 'convolution case order mismatch')
        keys = {'time_unix', 'event', 'case', 'artifacts', 'cuda_elapsed_ms', 'cuda_peak_allocated', 'cuda_memory_reserved',
                'input_stride', 'state_stride', 'output_alias', 'metadata'}
        require(set(row) == keys and row['event'] == 'output' and row['case'] == case, 'convolution output schema mismatch')
        require(all(type(v) is int for v in row['input_stride'] + row['state_stride']), 'convolution stride types')
        require(row['input_stride'] == [CHANNELS + 320, 1] and row['state_stride'] == [3 * CHANNELS, 1, CHANNELS] and
                row['output_alias'] is decode, 'convolution runtime strides or alias mismatch')
        programs = sum((length + 7) // 8 for length in CASES[case])
        expected_metadata = None if decode else {'programs': programs, 'pinned': True, 'gpu_shapes': [[2048], [2048]],
                                                'host_dtypes': ['torch.int64', 'torch.int32']}
        require(json.dumps(row['metadata'], sort_keys=True) == json.dumps(expected_metadata, sort_keys=True) and (decode or row['metadata']['pinned'] is True), 'convolution metadata mismatch')
        minimum = count * (CHANNELS + 320) * 2 + 5 * 3 * CHANNELS * 2 + CHANNELS * 4 * 4
        if not decode: minimum += count * CHANNELS * 2 + 2 * 2048 * 4
        require(type(row['cuda_elapsed_ms']) in (int, float) and math.isfinite(row['cuda_elapsed_ms']) and row['cuda_elapsed_ms'] > 0 and
                type(row['cuda_peak_allocated']) is int and type(row['cuda_memory_reserved']) is int and
                minimum <= row['cuda_peak_allocated'] <= row['cuda_memory_reserved'], 'convolution memory or timing invalid')
        names = [f'{kind}-{case}.bf16.gz' for kind in ('output', 'state', 'input')]
        items = row['artifacts']
        require(isinstance(items, list) and len(items) == 3 and all(isinstance(item, dict) and set(item) == {'file', 'sha256'} and
                item['file'] == name for item, name in zip(items, names)), 'convolution artifact schema mismatch')
        checks.append({'case': case, **score_tensors([root / name for name in names], [item['sha256'] for item in items], seed, case, channels=CHANNELS)})
    return checks


def run_native(metadata, output, seed, record):
    import numpy as np
    import torch
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata
    prior_api().normalized_geometry(metadata)
    require(torch.cuda.get_device_capability() == (12, 1), 'SM121 required')
    # CPU arrays are copied into persistent page-locked storage before any H2D transfer.
    source_host = torch.empty((2048, CHANNELS + 320), dtype=torch.bfloat16, pin_memory=True)
    state_host = torch.empty((5, 3, CHANNELS), dtype=torch.bfloat16, pin_memory=True)
    weight_host = torch.empty((CHANNELS, 4), dtype=torch.float32, pin_memory=True)
    ids_host = torch.empty(4, dtype=torch.int32, pin_memory=True)
    starts_host = torch.empty(5, dtype=torch.int32, pin_memory=True)
    flags_host = torch.empty(4, dtype=torch.bool, pin_memory=True)
    output_host = torch.empty((2048, CHANNELS), dtype=torch.bfloat16, pin_memory=True)
    post_host = torch.empty_like(source_host, pin_memory=True)
    final_host = torch.empty_like(state_host, pin_memory=True)
    buffers = (source_host, state_host, weight_host, ids_host, starts_host, flags_host, output_host, post_host, final_host)
    require(all(t.is_pinned() for t in buffers), 'probe staging not pinned')
    record({'event': 'configured', 'case_order': case_order(seed), 'slots': slots(seed), 'pinned_staging': True, **geometry()})
    for case in case_order(seed):
        source, weight, initial, ids, flags = fixture(seed, case); count = len(source); n = len(ids)
        source_host[:count].copy_(torch.from_numpy(source)); state_host.copy_(torch.from_numpy(initial)); weight_host.copy_(torch.from_numpy(weight))
        for i, slot in enumerate(ids): ids_host[i] = slot; flags_host[i] = flags[i]
        starts_host[0] = 0
        for i, length in enumerate(CASES[case]): starts_host[i + 1] = int(starts_host[i]) + length
        backing = source_host[:count].to('cuda', non_blocking=True); state_backing = state_host.to('cuda', non_blocking=True)
        weights = weight_host.to('cuda', non_blocking=True); indices = ids_host[:n].to('cuda', non_blocking=True)
        starts = starts_host[:n + 1].to('cuda', non_blocking=True); initial_flags = flags_host[:n].to('cuda', non_blocking=True)
        x = backing[:, :CHANNELS]; state = state_backing.transpose(-1, -2)
        meta = None; meta_receipt = None; decode = case.startswith('decode')
        if not decode:
            nums, batch_ptr, offsets = compute_causal_conv1d_metadata(starts_host[:n + 1], device=torch.device('cuda'))
            require(nums[8]['mlist'].is_pinned() and nums[8]['offsetlist'].is_pinned(), 'upstream metadata not pinned')
            meta = SimpleNamespace(nums_dict=nums, batch_ptr=batch_ptr, token_chunk_offset_ptr=offsets)
            meta_receipt = {'programs': nums[8]['tot'], 'pinned': True, 'gpu_shapes': [list(batch_ptr.shape), list(offsets.shape)],
                            'host_dtypes': [str(nums[8]['mlist'].dtype), str(nums[8]['offsetlist'].dtype)]}
        torch.cuda.synchronize(); record({'event': 'start', 'case': case})
        torch.cuda.reset_peak_memory_stats(); begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True); begin.record()
        if decode:
            result = causal_conv1d_update(x, state, weights, None, activation='silu', conv_state_indices=indices)
        else:
            result = causal_conv1d_fn(x.transpose(0, 1), weights, None, conv_states=state, query_start_loc=starts,
                                     cache_indices=indices, has_initial_state=initial_flags, activation='silu', metadata=meta).transpose(0, 1)
        end.record(); torch.cuda.synchronize()
        output_host[:count].copy_(result, non_blocking=True); final_host.copy_(state_backing, non_blocking=True)
        post_host[:count].copy_(backing, non_blocking=True); torch.cuda.synchronize()
        paths = [output / f'{kind}-{case}.bf16.gz' for kind in ('output', 'state', 'input')]
        for path, tensor in zip(paths, (output_host[:count], final_host, post_host[:count])):
            with gzip.open(path, 'wb') as stream: stream.write(memoryview(tensor.view(torch.uint16).numpy()).cast('B'))
        hashes = [sha256_file(path) for path in paths]
        record({'event': 'output', 'case': case, 'artifacts': [{'file': p.name, 'sha256': h} for p, h in zip(paths, hashes)],
                'cuda_elapsed_ms': begin.elapsed_time(end), 'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
                'cuda_memory_reserved': torch.cuda.memory_reserved(), 'input_stride': list(x.stride()), 'state_stride': list(state.stride()),
                'output_alias': result.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr(), 'metadata': meta_receipt})
        score_tensors(paths, hashes, seed, case)
        del backing, state_backing, weights, indices, starts, initial_flags, x, state, result, meta
        if not decode: del nums, batch_ptr, offsets


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True); parser.add_argument('--pinned-convolution', action='store_true')
    args = parser.parse_args(); case_order(args.seed); require(args.pinned_convolution, 'explicit --pinned-convolution required')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'qualification': QUALIFICATION, 'seed': args.seed, 'scorer_sha256': sha256_file(Path(__file__)),
                'binary_sha256': sha256_file(Path(sys.executable).resolve()), 'startup_selection': 'pinned_convolution',
                'decision': {'sha256': sha256_file(ROOT / 'configs/decision-specs/glm53-conv-preflight.json')},
                'metadata': {p.name: {'sha256': sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()}, 'start_unix': time.time()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n'); failure = None
    with (args.output / 'raw.jsonl').open('w') as raw, (args.output / 'traceback.log').open('w') as errors:
        def record(row): raw.write(json.dumps({'time_unix': time.time(), **row}, allow_nan=False) + '\n'); raw.flush()
        try: run_native(args.metadata, args.output, args.seed, record)
        except Exception as error:
            failure = repr(error); traceback.print_exc(file=errors); record({'event': 'failure', 'failure': failure})
    summary = {'verdict': 'FAIL' if failure else 'PASS', 'failure': failure, 'qualification': QUALIFICATION,
               'model_loaded': False, 'actual_input_tokens_processed': 0, 'context_capability': 'not measured',
               'performance': 'not measured', 'raw_sha256': sha256_file(args.output / 'raw.jsonl')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary))
    raise SystemExit(1 if failure else 0)


if __name__ == '__main__': main()
