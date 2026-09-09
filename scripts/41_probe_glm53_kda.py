#!/usr/bin/env python3
"""Model-free KDA prefill/decode analytic falsifier. JIT preparation only."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json

QUALIFICATION = 'model_free_KDA_analytic_falsifier_only'
ROWS = (1, 2, 3, 4, 2048)
PATTERN = (0.5, -0.25, 1.0, -2.0)


def require(value, message):
    if not value: raise ValueError(message)


def case_order(seed):
    require(type(seed) is int and 0 <= seed < 2**64, 'invalid KDA seed')
    order = list(ROWS); random.Random(seed).shuffle(order); return order


def request_order(seed):
    case_order(seed)
    order = list(range(4)); random.Random(seed ^ 0x4B4441).shuffle(order); return order


def initial_coefficient(seed):
    case_order(seed)
    return 0.125 * (seed % 4 + 1)


def expected(seed, count):
    """Closed-form one-basis recurrence; never uses captured GPU values."""
    import numpy as np
    require(type(count) is int and count in ROWS, 'invalid KDA row count')
    order = request_order(seed)
    pattern = np.tile(np.array(PATTERN, dtype=np.float32), 32)
    head_scale = 1 + (np.arange(64, dtype=np.float32) % 8) / 8
    values = np.arange(1, 5, dtype=np.float32)[:, None, None] * head_scale[None, :, None] * pattern[None, None, :]
    state = np.zeros((5, 64, 128, 128), dtype=np.float32); state[0] = 3.25
    n = 1 / math.sqrt(1 + 1e-6); a = math.exp(-2.5) * (1 - 0.5 * n * n); b = 0.5 * n
    def coefficient(t):
        return a**t * initial_coefficient(seed) + b * (1 - a**t) / (1 - a)
    for request in range(4):
        for head in range(64): state[request + 1, head, :, head] = values[request, head] * initial_coefficient(seed)
    output = np.empty((count, 64, 128), dtype=np.float32)
    for index in range(count):
        request = order[index // 512] if count == 2048 else order[index]
        t = index % 512 + 1 if count == 2048 else 1
        output[index] = values[request] * (coefficient(t) * n / math.sqrt(128))
    for request in (order if count == 2048 else order[:count]):
        for head in range(64): state[request + 1, head, :, head] = values[request, head] * coefficient(512 if count == 2048 else 1)
    return output, state


def read_tensor(path, digest, size):
    value = path.lstat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    require(stat.S_ISREG(value.st_mode) and value.st_size <= size + 1048576, 'invalid KDA artifact size or type')
    require(sha256_file(path) == digest, 'KDA artifact digest mismatch')
    with gzip.open(path, 'rb') as stream:
        blob = stream.read(size + 1)
        require(len(blob) == size and stream.read(1) == b'', 'KDA artifact payload size mismatch')
    require(identity(path.lstat()) == identity(value) and sha256_file(path) == digest, 'KDA artifact changed')
    return blob


def score_tensors(output_path, state_path, digests, count, seed):
    import numpy as np
    reference = expected(seed, count)
    output = (np.frombuffer(read_tensor(output_path, digests[0], count * 64 * 128 * 2), dtype='<u2').astype('<u4') << 16).view('<f4').reshape(count, 64, 128)
    state = np.frombuffer(read_tensor(state_path, digests[1], 5 * 64 * 128 * 128 * 4), dtype='<f4').reshape(5, 64, 128, 128)
    mismatches = nonfinite = elements = 0; maximum = 0.0
    for actual, target in zip((output, state), reference):
        finite = np.isfinite(actual); errors = np.abs(actual - target)
        nonfinite += int(np.count_nonzero(~finite)); elements += actual.size
        mismatches += int(np.count_nonzero(~finite | (errors > 0.001 + 0.01 * np.abs(target))))
        if finite.any(): maximum = max(maximum, float(errors[finite].max()))
    require(nonfinite == 0 and mismatches == 0, 'KDA tensors violate analytic reference')
    return {'elements': elements, 'maximum_absolute_error': maximum, 'mismatched_elements': mismatches, 'nonfinite_elements': nonfinite}


def run_native(metadata, output, seed, record):
    import torch
    import vllm.models.glm5next.nvidia.kda as kda
    require(torch.cuda.get_device_capability() == (12, 1), 'SM121 required')
    text = strict_json(metadata / 'config.json')['text_config']
    require(text['linear_num_heads'] == 64 and text['linear_head_dim'] == 128 and
            text['linear_lower_bound'] == -5.0, 'KDA model geometry changed')
    requests = request_order(seed)
    pattern_host = torch.empty(4, dtype=torch.bfloat16, pin_memory=True)
    order_host = torch.empty(4, dtype=torch.int32, pin_memory=True)
    seqlens_host = torch.empty(5, dtype=torch.int32, pin_memory=True)
    for i in range(4): pattern_host[i] = PATTERN[i]; order_host[i] = requests[i]
    pattern = pattern_host.to('cuda', non_blocking=True).repeat(32)
    order = order_host.to('cuda', non_blocking=True)
    output_host = torch.empty((2048, 64, 128), dtype=torch.bfloat16, pin_memory=True)
    state_host = torch.empty((5, 64, 128, 128), dtype=torch.float32, pin_memory=True)
    basis = torch.eye(128, device='cuda', dtype=torch.bfloat16)[:64].contiguous()
    head_scale = 1 + (torch.arange(64, device='cuda', dtype=torch.float32) % 8) / 8
    values = torch.arange(1, 5, device='cuda', dtype=torch.float32)[:, None, None] * head_scale[None, :, None] * pattern[None, None, :]
    a_log = torch.zeros((1, 1, 64, 1), device='cuda', dtype=torch.float32)
    bias = torch.zeros(64 * 128, device='cuda', dtype=torch.float32)
    record({'event': 'configured', 'case_order': case_order(seed), 'request_order': requests,
            'state_shape': [5, 64, 128, 128], 'pinned_staging': all(t.is_pinned() for t in
                (pattern_host, order_host, seqlens_host, output_host, state_host))})
    for count in case_order(seed):
        state = torch.zeros((5, 64, 128, 128), device='cuda', dtype=torch.float32); state[0].fill_(3.25)
        state[1:] = values[..., None] * basis[None, :, None, :].float() * initial_coefficient(seed)
        request_ids = order.repeat_interleave(512) if count == 2048 else order[:count]
        q = basis[None, None].expand(1, count, -1, -1).contiguous(); k = q.clone()
        v = values[request_ids.long()].to(torch.bfloat16).unsqueeze(0).contiguous()
        g = torch.zeros_like(q); beta = torch.zeros((1, count, 64), device='cuda', dtype=torch.bfloat16)
        sequences = 4 if count == 2048 else count
        for i in range(sequences + 1): seqlens_host[i] = i * (512 if count == 2048 else 1)
        cu_seqlens = seqlens_host[:sequences + 1].to('cuda', non_blocking=True)
        ids = (order[:sequences] + 1).contiguous()
        torch.cuda.synchronize(); record({'event': 'start', 'query_rows': count})
        torch.cuda.reset_peak_memory_stats(); start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        if count == 2048:
            initial = kda.gather_initial_states(state, ids, torch.ones(4, device='cuda', dtype=torch.bool))
            result, final = kda.chunk_kda_with_fused_gate(q=q, k=k, v=v, raw_g=g,
                beta=kda._cast_sigmoid(beta.squeeze(0)).unsqueeze(0), A_log=a_log, g_bias=bias,
                initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens, safe_gate=True, lower_bound=-5.0)
            kda.scatter_states(state, final, ids)
        else:
            result, _ = kda.fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens, ssm_state_indices=ids,
                sigmoid_beta=True, a_log=a_log, g_bias=bias, compute_gate=True, lower_bound=-5.0)
        end.record(); torch.cuda.synchronize()
        output_host[:count].copy_(result.squeeze(0), non_blocking=True); state_host.copy_(state, non_blocking=True); torch.cuda.synchronize()
        paths = [output / f'output-{count}.bf16.gz', output / f'state-{count}.fp32.gz']
        for path, tensor in zip(paths, (output_host[:count].view(torch.uint16), state_host)):
            with gzip.open(path, 'wb') as stream: stream.write(memoryview(tensor.numpy()).cast('B'))
        digests = [sha256_file(path) for path in paths]
        record({'event': 'output', 'query_rows': count, 'artifacts': [
            {'file': path.name, 'sha256': digest} for path, digest in zip(paths, digests)],
            'cuda_elapsed_ms': start.elapsed_time(end), 'cuda_peak_allocated': torch.cuda.max_memory_allocated(),
            'cuda_memory_reserved': torch.cuda.memory_reserved()})
        score_tensors(*paths, digests, count, seed)
        del state, q, k, v, g, beta, result


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True); args = parser.parse_args(); case_order(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'qualification': QUALIFICATION, 'seed': args.seed, 'scorer_sha256': sha256_file(Path(__file__)),
        'binary_sha256': sha256_file(Path(sys.executable).resolve()),
        'decision': {'sha256': sha256_file(ROOT / 'configs/decision-specs/glm53-kda-preflight.json')},
        'metadata': {p.name: {'sha256': sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()}, 'start_unix': time.time()}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    failure = None
    with (args.output / 'raw.jsonl').open('w') as raw, (args.output / 'traceback.log').open('w') as errors:
        def record(row):
            raw.write(json.dumps({'time_unix': time.time(), **row}, allow_nan=False) + '\n'); raw.flush()
        try: run_native(args.metadata, args.output, args.seed, record)
        except Exception as error:
            failure = repr(error); traceback.print_exc(file=errors); record({'event': 'failure', 'failure': failure})
    summary = {'verdict': 'FAIL' if failure else 'PASS', 'failure': failure, 'qualification': QUALIFICATION,
        'model_loaded': False, 'actual_input_tokens_processed': 0, 'context_capability': 'not measured',
        'performance': 'not measured', 'raw_sha256': sha256_file(args.output / 'raw.jsonl')}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n'); print(json.dumps(summary))
    raise SystemExit(1 if failure else 0)


if __name__ == '__main__': main()
