#!/usr/bin/env python3
"""Bounded SM121 MLA constant-cache falsifier, with independently scored outputs.

Evidence only. Identical values per request have an analytic attention result.
This does not qualify general attention, fidelity, processed context or speed.
Run only from a frozen copy through the reviewed identity/cgroup controller.
"""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json

QUALIFICATION = 'model_free_MLA_constant_cache_falsifier_only'
ROWS = (1, 2, 3, 4, 2048)
PATTERN = (0.5, -0.25, 1.0, -2.0)


def require(value, message):
    if not value:
        raise ValueError(message)


def case_order(seed):
    require(type(seed) is int and 0 <= seed < 2**64, 'invalid public seed')
    order = list(ROWS); random.Random(seed).shuffle(order)
    return order


def request_order(seed):
    order = list(range(4)); random.Random(seed ^ 0x4D4C41).shuffle(order)
    return order


def fixture_seed(seed, rows):
    return int.from_bytes(hashlib.sha256(json.dumps([seed, rows], separators=(',', ':')).encode()).digest()[:8], 'big') % 2**63


def score_tensor(path, digest, rows, requests):
    """Read every BF16 output element, with bounded decompression and no pickle."""
    import numpy as np
    require(type(rows) is int and rows in ROWS, 'invalid output shape')
    require(isinstance(requests, list) and len(requests) == 4 and
            all(type(r) is int for r in requests) and set(requests) == set(range(4)), 'invalid request order')
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= rows * 65536 + 1048576, 'invalid output file or size')
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    require(sha256_file(path) == digest, 'output digest mismatch')
    chunk_bytes = 64 * 512 * 2
    pattern = np.tile(np.array(PATTERN, dtype=np.float32), 64 * 512 // 4)
    maximum = 0.0; mismatches = 0; nonfinite = 0
    with gzip.open(path, 'rb') as stream:
        for index in range(rows):
            blob = stream.read(chunk_bytes)
            require(len(blob) == chunk_bytes, 'output size mismatch')
            values = (np.frombuffer(blob, dtype='<u2').astype('<u4') << 16).view('<f4')
            expected = pattern * (requests[index % 4] + 1)
            finite = np.isfinite(values)
            nonfinite += int(np.count_nonzero(~finite))
            error = np.abs(values - expected)
            if finite.any(): maximum = max(maximum, float(error[finite].max()))
            mismatches += int(np.count_nonzero(~finite | (error > 0.02 + 0.02 * np.abs(expected))))
        require(stream.read(1) == b'', 'output size exceeds frozen shape')
    require(identity(path.lstat()) == identity(before) and sha256_file(path) == digest, 'output file changed during scoring')
    require(nonfinite == 0 and mismatches == 0, 'MLA output violates analytic reference')
    return {'elements': rows * 64 * 512, 'maximum_absolute_error': maximum,
            'mismatched_elements': mismatches, 'nonfinite_elements': nonfinite}


def run_native(metadata, output, seed, record):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
    require(torch.cuda.get_device_capability() == (12, 1), 'SM121 required')
    cfg = EngineArgs(model=str(metadata), skip_tokenizer_init=True, dtype='bfloat16', quantization='exl3',
                     max_model_len=262144, max_num_seqs=4, max_num_batched_tokens=2048,
                     enable_chunked_prefill=True, enable_prefix_caching=False, kv_cache_dtype='fp8',
                     compilation_config={'mode': 0}).create_engine_config()
    text = cfg.model_config.hf_text_config
    require(text.num_attention_heads == 64 and text.kv_lora_rank == 512 and
            text.qk_nope_head_dim == 256 and text.qk_rope_head_dim == 0 and
            text.index_topk == 2048 and text.index_kpool == 4, 'model MLA geometry changed')
    requests = request_order(seed)
    # Persistent pinned staging covers every introduced host-to-device copy.
    pattern_host = torch.empty(4, dtype=torch.bfloat16, pin_memory=True)
    order_host = torch.empty(4, dtype=torch.int32, pin_memory=True)
    for i in range(4): pattern_host[i] = PATTERN[i]; order_host[i] = requests[i]
    pattern = pattern_host.to('cuda', non_blocking=True)
    order_device = order_host.to('cuda', non_blocking=True)
    capture_host = torch.empty((2048, 64, 512), dtype=torch.bfloat16, pin_memory=True)
    positions = torch.arange(2048, device='cuda', dtype=torch.int64) * 262143 // 2047
    tables = torch.arange(16384, device='cuda', dtype=torch.int32).reshape(4, 4096)
    topk = positions.to(torch.int32).expand(2048, -1).contiguous()
    cache = torch.zeros((16384, 64, 656), device='cuda', dtype=torch.uint8)
    values = (torch.arange(1, 5, device='cuda', dtype=torch.bfloat16)[:, None, None] *
              pattern.repeat(128)[None, None, :]).expand(4, 2048, 512).contiguous().reshape(-1, 512)
    slots = (positions[None, :] + torch.arange(4, device='cuda', dtype=torch.int64)[:, None] * 262144).flatten()
    with set_current_vllm_config(cfg):
        impl = FlashInferMLASparseSM120Impl(num_heads=64, head_size=512, scale=256**-0.5,
            num_kv_heads=1, alibi_slopes=None, sliding_window=None, kv_cache_dtype='fp8_ds_mla',
            logits_soft_cap=None, attn_type=AttentionType.DECODER, kv_sharing_target_layer_name=None,
            kv_lora_rank=512, qk_nope_head_dim=256, qk_rope_head_dim=0, topk_indices_buffer=topk)
        require(impl.kv_scale_format == 'arbitrary_fp32' and impl.rope_pad == 64, 'SM121 GLM packing changed')
        impl.do_kv_cache_update(values, torch.empty((8192, 1, 0), device='cuda', dtype=torch.bfloat16),
                                cache, slots, 'fp8_ds_mla', torch.ones(1, device='cuda'))
        torch.cuda.synchronize()
        record({'event': 'configured', 'backend': type(impl).__name__, 'cache_bytes': cache.untyped_storage().nbytes(),
                'request_order': requests, 'case_order': case_order(seed), 'addressed_last_position': int(positions[-1].item()),
                'pinned_staging': bool(pattern_host.is_pinned() and order_host.is_pinned() and capture_host.is_pinned())})
        for count in case_order(seed):
            chosen_seed = fixture_seed(seed, count); torch.manual_seed(chosen_seed)
            query = torch.randn((count, 64, 512), device='cuda', dtype=torch.bfloat16)
            capture_host[:count].copy_(query, non_blocking=True); torch.cuda.synchronize()
            query_digest = hashlib.sha256(memoryview(capture_host[:count].view(torch.uint16).numpy()).cast('B')).hexdigest()
            request_ids = order_device[torch.arange(count, device='cuda') % 4]
            metadata_view = SimpleNamespace(req_id_per_token=request_ids, block_table=tables, block_size=64)
            record({'event': 'start', 'query_rows': count, 'fixture_seed': chosen_seed, 'query_sha256': query_digest})
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(); result, lse = impl.forward_mqa(query, cache, metadata_view, None); end.record()
            torch.cuda.synchronize()
            require(result.shape == (count, 64, 512) and result.dtype == torch.bfloat16 and lse is None, 'MLA output shape changed')
            capture_host[:count].copy_(result, non_blocking=True); torch.cuda.synchronize()
            artifact = output / f'output-{count}.bf16.gz'
            with artifact.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                compressed.write(memoryview(capture_host[:count].view(torch.uint16).numpy()).cast('B'))
            row = {'event': 'output', 'query_rows': count, 'file': artifact.name, 'sha256': sha256_file(artifact),
                   'uncompressed_bytes': count * 64 * 512 * 2, 'cuda_elapsed_ms': start.elapsed_time(end),
                   'cuda_peak_allocated': torch.cuda.max_memory_allocated(), 'cuda_memory_reserved': torch.cuda.memory_reserved()}
            record(row)
            score_tensor(artifact, row['sha256'], count, requests)
            del result, query


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True); args = parser.parse_args(); case_order(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    decision = ROOT / 'configs/decision-specs/glm53-mla-preflight.json'
    manifest = {'qualification': QUALIFICATION, 'seed': args.seed, 'scorer_sha256': sha256_file(Path(__file__)),
        'binary_sha256': sha256_file(Path(sys.executable).resolve()), 'decision': {'sha256': sha256_file(decision)},
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
