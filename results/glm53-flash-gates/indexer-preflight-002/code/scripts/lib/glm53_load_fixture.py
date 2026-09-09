"""Deterministic full-geometry synthetic loading fixtures; never model weights."""
import hashlib
import json
import math
import struct

DTYPE_BYTES = {'I16': 2, 'I32': 4, 'F16': 2, 'BF16': 2}
BLOCK_BYTES = 4096
DOMAIN = b'GLM53-LOAD-FIXTURE-v1\x00'


def require(value, message):
    if not value: raise ValueError(message)


def byte_count(spec):
    require(isinstance(spec, dict) and isinstance(spec.get('name'), str) and spec['name'] and
            isinstance(spec.get('dtype'), str) and spec['dtype'] in DTYPE_BYTES and
            isinstance(spec.get('shape'), list) and all(type(n) is int and n > 0 for n in spec['shape']), 'invalid fixture tensor')
    return math.prod(spec['shape']) * DTYPE_BYTES[spec['dtype']]


def tensor_specs(case):
    """Exact selected checkpoint geometry plus explicit parameter-copy routing."""
    specs = []
    def add(name, dtype, shape, parameter, shard_id=None, expert_id=None):
        specs.append({'name': name, 'dtype': dtype, 'shape': shape, 'parameter': parameter,
                      'shard_id': shard_id, 'expert_id': expert_id})
    if case == 'moe':
        prefix = 'model.language_model.layers.3.mlp.experts'
        for expert in range(288):
            for projection, shard, inputs, outputs in (('gate_proj', 'w1', 4096, 2048),
                    ('up_proj', 'w3', 4096, 2048), ('down_proj', 'w2', 2048, 4096)):
                for part, dtype, shape in (('trellis', 'I16', [inputs // 16, outputs // 16, 32]),
                        ('suh', 'F16', [inputs]), ('svh', 'F16', [outputs]), ('mcg', 'I32', [1])):
                    add(f'{prefix}.{expert}.{projection}.{part}', dtype, shape,
                        ('w2_' if shard == 'w2' else 'w13_') + part, shard, expert)
    elif case in ('kda', 'mla'):
        prefix = 'model.language_model.layers.' + ('0' if case == 'kda' else '3') + '.self_attn'
        projections = (('q_proj', 8192), ('k_proj', 8192), ('v_proj', 8192)) if case == 'kda' else (
            ('q_a_proj', 1536), ('kv_a_proj_with_mqa', 512))
        for shard, (projection, outputs) in enumerate(projections):
            for part, dtype, shape in (('trellis', 'I16', [256, outputs // 16, 64]),
                    ('suh', 'F16', [4096]), ('svh', 'F16', [outputs]), ('mul1', 'I32', [])):
                add(f'{prefix}.{projection}.{part}', dtype, shape, part, shard)
        if case == 'kda':
            for shard, projection, outputs in ((3, 'b_proj', 64), (4, 'f_a_proj', 128), (5, 'g_a_proj', 128)):
                add(f'{prefix}.{projection}.weight', 'BF16', [outputs, 4096], 'weight', shard)
    elif case == 'ordinary':
        add('model.language_model.embed_tokens.weight', 'BF16', [154880, 4096], 'weight')
    else:
        raise ValueError('unknown bounded load case')
    return sorted(specs, key=lambda row: row['name'])


def fixture_bytes(seed, spec, offset, count):
    """Random-access canonical blocks; output is independent of copy boundaries."""
    require(type(seed) is int and 0 <= seed < 2**64, 'invalid fixture seed')
    length = byte_count(spec)
    require(type(offset) is int and type(count) is int and 0 <= offset <= length and
            0 <= count <= 8 * 1024 * 1024 and offset + count <= length, 'invalid bounded fixture slice')
    if spec['dtype'] == 'I32':
        marker = {'mcg': -877912083, 'mul1': -2082680531}.get(spec['name'].rsplit('.', 1)[-1])
        require(marker is not None and spec['shape'] in ([], [1]), 'invalid fixture marker')
        return struct.pack('<i', marker)[offset:offset + count]
    if not count: return b''
    import numpy as np
    result = bytearray()
    for block_index in range(offset // BLOCK_BYTES, (offset + count - 1) // BLOCK_BYTES + 1):
        encoded = json.dumps([seed, spec['name'], spec['dtype'], spec['shape'], block_index],
                             ensure_ascii=True, separators=(',', ':')).encode('utf-8')
        raw = hashlib.shake_256(DOMAIN + struct.pack('>Q', len(encoded)) + encoded).digest(BLOCK_BYTES)
        if spec['dtype'] in ('F16', 'BF16'):
            words = np.frombuffer(raw, dtype='<u2').copy()
            mask, exponent = (0x83ff, 0x2400) if spec['dtype'] == 'F16' else (0x807f, 0x3e80)
            words &= mask; words |= exponent
            raw = words.tobytes()
        start = max(offset - block_index * BLOCK_BYTES, 0)
        end = min(offset + count - block_index * BLOCK_BYTES, BLOCK_BYTES)
        result.extend(raw[start:end])
    return bytes(result)
