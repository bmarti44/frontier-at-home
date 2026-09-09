"""Bounded native top-k diagnostic. No model is loaded and indices are not used as addresses."""
import hashlib
import json
from pathlib import Path
import sys
import time

if sys.flags.optimize:
    raise RuntimeError('optimized Python is not accepted')
OUT = Path(sys.argv[1])
CASE = sys.argv[2]
if CASE not in ('finite', 'nan'):
    raise ValueError('unknown diagnostic case')
OUT.mkdir(exist_ok=False)
with (OUT / 'raw.jsonl').open('x') as raw:
    def emit(value):
        raw.write(json.dumps({'time_unix': time.time(), **value}, allow_nan=False) + '\n')
        raw.flush()
    emit({'event': 'start', 'case': CASE, 'scope': 'native operator only; not crash attribution'})
    import torch
    import vllm._custom_ops  # Registers the exact installed native operator.
    torch.set_num_threads(1)
    rows, k = 128, 512
    computed = [52992, 52928, 52896, 52864]
    ends_per_request = [(n + 32) // 4 for n in computed]
    offsets = [0]
    for n in ends_per_request:
        offsets.append(offsets[-1] + n)
    # Persistent pinned staging for every host-to-device input.
    host = torch.arange(offsets[-1], dtype=torch.float32).repeat(rows, 1).pin_memory()
    starts = torch.tensor([offsets[i] for i in range(4) for _ in range(32)], dtype=torch.int32).pin_memory()
    ends = torch.tensor([offsets[i] + (computed[i] + step + 1) // 4
                         for i in range(4) for step in range(32)], dtype=torch.int32).pin_memory()
    if CASE == 'nan':
        host.fill_(float('nan'))
    inputs = {'logits': host, 'row_starts': starts, 'row_ends': ends}
    torch.save(inputs, OUT / 'inputs.pt')
    device_inputs = {name: value.to('cuda', non_blocking=True) for name, value in inputs.items()}
    torch.cuda.synchronize()
    storage = torch.full((rows * k + 64,), -777777, dtype=torch.int32, device='cuda')
    output = storage[32:-32].view(rows, k)
    emit({'event': 'before_operator', 'rows': rows, 'columns': offsets[-1], 'top_k': k,
          'case': CASE, 'allocated_bytes': torch.cuda.memory_allocated()})
    logits = device_inputs['logits']
    torch.ops._C.top_k_per_row_prefill(logits, device_inputs['row_starts'],
        device_inputs['row_ends'], output, rows, logits.stride(0), logits.stride(1), k)
    torch.cuda.synchronize()
    observed = storage.cpu()
    torch.save(observed, OUT / 'output.pt')
    indices = observed[32:-32].view(rows, k)
    lengths = (ends - starts).tolist()
    bad_ranges = sum(not bool(((row >= 0) & (row < n)).all()) for row, n in zip(indices, lengths))
    duplicate_rows = sum(len(set(row.tolist())) != k for row in indices)
    canaries = bool((observed[:32] == -777777).all() and (observed[-32:] == -777777).all())
    finite_matches = (all(sorted(row.tolist()) == list(range(n - k, n))
                          for row, n in zip(indices, lengths)) if CASE == 'finite' else None)
    emit({'event': 'completed', 'rows_with_out_of_range_indices': bad_ranges,
          'rows_with_duplicate_indices': duplicate_rows, 'canaries_intact': canaries,
          'finite_topk_matches_independent_expected_set': finite_matches})
    summary = {'case': CASE, 'scope': 'synthetic operator diagnostic, no model weights',
               'verdict': 'PASS' if canaries and not bad_ranges and not duplicate_rows and finite_matches is not False else 'FAIL',
               'crash_attribution': 'NO_RESULT', 'rows_with_out_of_range_indices': bad_ranges,
               'rows_with_duplicate_indices': duplicate_rows, 'canaries_intact': canaries,
               'finite_topk_matches_expected': finite_matches,
               'limits': 'No native model logits were captured. This tests selection only and never gathers attention using returned indices.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    files = []
    for p in [Path(__file__), OUT / 'inputs.pt', OUT / 'output.pt', OUT / 'summary.json']:
        with p.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append({'path': str(p), 'sha256': digest, 'size_bytes': p.stat().st_size})
    (OUT / 'manifest.json').write_text(json.dumps({'files': files}, indent=2) + '\n')
