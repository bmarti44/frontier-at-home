"""Evidence-only selected weight loading with persistent pinned byte staging.

Not installed in serving. The synchronous consumer must copy its input and may
not retain the temporary or any view. Actual module finalization/peak checks are
separate; this adapter does not pin the plugin's internal pointer-table copies.
"""
import hashlib
import math
from pathlib import Path
import re
import types

from glm53_contract import verify_inventory

DTYPES = {'torch.int16': 2, 'torch.int32': 4, 'torch.float16': 2,
          'torch.bfloat16': 2, 'torch.float32': 4}
MAX_TENSOR_BYTES = 1268776960
MAX_PINNED_BYTES = 8 * 1024 * 1024


def require(value, message):
    if not value: raise ValueError(message)


def selected_weights(root, inventory, selection):
    """Verify inputs now; return a locally filtered stock lazy iterator.

    No shared loader globals are changed. Whole-file inventory verification is
    explicit I/O; selection prevents get_tensor on excluded names, not filesystem
    readahead or the independent prerequisite file hashing.
    """
    root = Path(root).absolute()
    identities = verify_inventory(root, inventory)
    require(isinstance(selection, dict) and selection, 'empty weight selection')
    owners = {}
    for name, row in selection.items():
        require(isinstance(name, str) and name and isinstance(row, dict) and
                set(row) == {'file', 'dtype', 'shape', 'sha256'}, 'invalid weight selection schema')
        require(isinstance(row['file'], str) and row['file'] in identities and
                row['file'].endswith('.safetensors'), 'unbound selected shard')
        require(isinstance(row['dtype'], str) and row['dtype'] in DTYPES and isinstance(row['shape'], list) and
                all(type(n) is int and n > 0 for n in row['shape']) and
                0 < math.prod(row['shape']) * DTYPES[row['dtype']] <= MAX_TENSOR_BYTES,
                'invalid selected dtype/shape/size')
        require(isinstance(row['sha256'], str) and re.fullmatch(r'[0-9a-f]{64}', row['sha256']), 'invalid selected tensor digest')
        owners[name] = row['file']
    selection = {name: {**row, 'shape': list(row['shape'])} for name, row in selection.items()}
    require(set(owners.values()) == set(identities), 'selection/shard coverage mismatch')
    from vllm.model_executor.model_loader import weight_utils
    original = weight_utils.safetensors_weights_iterator
    skip_original = weight_utils.should_skip_weight

    def iterate():
        seen = set()
        for filename in sorted(identities):
            def skip(name, expert_ids):
                if name in owners and owners[name] != filename:
                    raise ValueError('selected tensor found in wrong shard')
                return name not in owners or skip_original(name, expert_ids)
            namespace = {**original.__globals__, 'should_skip_weight': skip}
            local = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
            local.__kwdefaults__ = dict(original.__kwdefaults__ or {})
            for name, tensor in local([str(root / filename)], use_tqdm_on_load=False,
                                      safetensors_load_strategy='lazy', local_expert_ids=None):
                require(name in owners and name not in seen, 'duplicate or unselected weight')
                row = selection[name]
                require(tensor.device.type == 'cpu' and tensor.is_contiguous() and
                        str(tensor.dtype) == row['dtype'] and list(tensor.shape) == row['shape'],
                        'selected source dtype/shape/layout mismatch')
                seen.add(name)
                yield name, tensor
                del tensor
        require(seen == set(selection), 'missing selected weights')
        require(verify_inventory(root, inventory) == identities, 'selected source changed while loading')
    return iterate()


def stream_selected_weights(root, inventory, selection, consume, record, *, enabled=False,
                            pinned_capacity=MAX_PINNED_BYTES):
    """Startup-selected callback loading; disabled selection does no work.

    `consume(name, cuda_tensor)` completes synchronously and must not retain its
    argument or views. Destination copy overlap is real memory overhead. The
    caller runs each fixture in fresh containment and records actual storages.
    """
    require(type(enabled) is bool, 'pinned stream selection must be boolean')
    if not enabled: return
    require(callable(consume) and callable(record), 'synchronous consumer and recorder required')
    require(type(pinned_capacity) is int and 0 < pinned_capacity <= MAX_PINNED_BYTES, 'invalid pinned capacity')
    inputs = selected_weights(root, inventory, selection)
    import torch
    staging = torch.empty(pinned_capacity, dtype=torch.uint8, pin_memory=True)
    require(staging.is_pinned(), 'persistent staging is not pinned')
    complete = torch.cuda.Event()
    reuse_count = 0
    for name, source in inputs:
        source_bytes = source.reshape(-1).view(torch.uint8)
        destination = torch.empty(tuple(source.shape), dtype=source.dtype, device='cuda')
        device_bytes = destination.reshape(-1).view(torch.uint8)
        source_hash = hashlib.sha256(); device_hash = hashlib.sha256()
        count = source_bytes.numel()
        chunks = 0
        try:
            for offset in range(0, count, pinned_capacity):
                size = min(pinned_capacity, count - offset)
                chunk = source_bytes[offset:offset + size]
                source_hash.update(memoryview(chunk.numpy()))
                staging[:size].copy_(chunk)
                device_bytes[offset:offset + size].copy_(staging[:size], non_blocking=True)
                complete.record(); complete.synchronize()
                chunks += 1; reuse_count += 1
            require(source_hash.hexdigest() == selection[name]['sha256'], 'source tensor digest mismatch')
            for offset in range(0, count, pinned_capacity):
                size = min(pinned_capacity, count - offset)
                staging[:size].copy_(device_bytes[offset:offset + size], non_blocking=True)
                complete.record(); complete.synchronize()
                device_hash.update(memoryview(staging[:size].numpy()))
                reuse_count += 1
            require(source_hash.digest() == device_hash.digest(), 'staged transfer byte mismatch')
            consume(name, destination)
            torch.cuda.synchronize()
            record({'name': name, 'bytes': count, 'source_sha256': source_hash.hexdigest(),
                    'device_sha256': device_hash.hexdigest(), 'staging_pointer': staging.data_ptr(),
                    'staging_bytes': pinned_capacity, 'pinned': True, 'upload_chunks': chunks,
                    'completed_reuses': reuse_count, 'temporary_bytes': destination.untyped_storage().nbytes()})
        finally:
            torch.cuda.synchronize()
            del device_bytes, destination, source_bytes, source
