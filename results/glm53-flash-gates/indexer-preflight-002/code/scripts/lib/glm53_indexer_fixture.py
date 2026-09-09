"""Independent synthetic post-projection indexer inputs and byte/selection oracles."""
import math
import random
import struct

CAP = 262144
POOLS = CAP // 4
CASES = {'decode-1': [1], 'decode-2': [1,1], 'decode-3': [1,1,1], 'decode-4': [1,1,1,1],
         'prefill-1': [2048], 'prefill-4': [512]*4}


def require(value, message):
    if not value: raise ValueError(message)


def case_order(seed):
    require(type(seed) is int and 0 <= seed < 2**64, 'invalid indexer seed')
    order = list(CASES); random.Random(seed).shuffle(order); return order


def request_order(seed):
    case_order(seed)
    order = [0,1,2,3]; random.Random(seed ^ 0x494E4458).shuffle(order); return order


def physical_blocks(seed):
    case_order(seed)
    blocks = list(range(1,145)); random.Random(seed ^ 0x50414745).shuffle(blocks)
    return [blocks[i*36:(i+1)*36] for i in range(4)]


def ranks(seed, request):
    import numpy as np
    case_order(seed); require(type(request) is int and 0 <= request < 4, 'invalid logical request')
    return ((2*request+1)*np.arange(POOLS, dtype='int64') + seed % POOLS + request*8191) % POOLS


def top_history(seed, request, count):
    import numpy as np
    require(type(count) is int and 512 <= count <= POOLS, 'invalid old-history count')
    return np.argsort(ranks(seed, request)[:count], kind='stable')[-512:]


def case_config(seed, case):
    import numpy as np
    case_order(seed); require(case in CASES, 'invalid indexer case')
    lengths = CASES[case]; requests = request_order(seed)[:len(lengths)]
    ends = [CAP-3+i for i in range(len(lengths))] if case.startswith('decode') else [CAP]*len(lengths)
    positions = np.concatenate([np.arange(end-length,end,dtype='<i8') for end,length in zip(ends,lengths)])
    history = [(end-length)//4 for end,length in zip(ends,lengths)]
    return {'requests': requests, 'lengths': list(lengths), 'ends': ends, 'starts': np.cumsum([0,*lengths], dtype='<i4'),
            'positions': positions, 'row_requests': np.repeat(requests, lengths), 'history_counts': np.repeat(history,lengths)}


def bf16_bits(value):
    require(type(value) in (int,float) and math.isfinite(value), 'invalid BF16 scalar')
    bits = struct.unpack('<I',struct.pack('<f',value))[0]
    return ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) & 0xffff


def bf16_value(value):
    return struct.unpack('<f',struct.pack('<I',bf16_bits(value) << 16))[0]


def fp8_value(bits):
    require(type(bits) is int and 0 <= bits <= 0x7e, 'invalid positive E4M3 bits')
    exponent, mantissa = bits >> 3, bits & 7
    return mantissa * 2**-9 if exponent == 0 else (1+mantissa/8) * 2**(exponent-7)


def fp8_positive(value):
    require(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 448, 'invalid positive FP8 value')
    return min(range(0x7f), key=lambda bits: (abs(fp8_value(bits)-value), bits & 1))


def raw_value(request, phase):
    require(type(request) is int and 0 <= request < 4 and type(phase) is int and 0 <= phase < 4, 'invalid raw key coordinate')
    return (request+1)*(1,2,4,8)[phase]/4096


def compressed_key(request):
    import numpy as np
    # Uniform four-way gate and channel-constant keys give a single Hadamard DC term.
    mean = bf16_value(sum(raw_value(request,p) for p in range(4))/4)
    dc = bf16_value(mean * math.sqrt(128))
    scale = 2.**math.ceil(math.log2(max(abs(dc),1e-4)/448))
    vector = np.zeros(128,dtype='uint8'); vector[0] = fp8_positive(min(dc/scale,448.))
    return vector, scale


def tables(seed, case):
    import numpy as np
    blocks = physical_blocks(seed); cfg = case_config(seed,case)
    common = np.array([[block*136+offset for block in blocks[r][:31] for offset in range(136)] for r in cfg['requests']], dtype='<i4')
    tail = np.array([[blocks[r][31]] for r in cfg['requests']],dtype='<i4')
    return common, tail


def cache_and_tail(seed, case, final=False):
    import numpy as np
    require(type(final) is bool, 'invalid final-state selection')
    cfg = case_config(seed,case); blocks = physical_blocks(seed)
    cache = np.full((4930,8448),0x55,dtype='uint8')
    tail = np.full((145,2,4,128),0x4050,dtype='<u2')
    page_keys = np.zeros((64,128),dtype='uint8'); page_keys[:,0] = 0x38
    for request in range(4):
        pages = np.array([block*34+offset for block in blocks[request][:31] for offset in range(34)])[:1024]
        scales = ((ranks(seed,request)+1)/65536).astype('<f4').reshape(1024,64)
        cache[pages,:8192] = page_keys.reshape(8192)
        cache[pages,8192:] = scales.view('uint8').reshape(1024,256)
    for index, request in enumerate(cfg['requests']):
        decode = case.startswith('decode'); last_phase = index if decode else 3
        # Before decode only older phases of this in-progress pool are valid.
        phases = range(last_phase+int(final)) if decode else (range(4) if final else ())
        for phase in phases:
            tail[blocks[request][31],0,phase] = bf16_bits(raw_value(request,phase))
            tail[blocks[request][31],1,phase] = 0
        if final:
            begin = (cfg['ends'][index]-cfg['lengths'][index])//4
            end = cfg['ends'][index]//4
            vector, scale = compressed_key(request)
            for pool in range(begin,end):
                block_index, local_pool = divmod(pool,2176)
                page, offset = blocks[request][block_index]*34+local_pool//64, local_pool%64
                cache[page,offset*128:(offset+1)*128] = vector
                cache[page,8192:].view('<f4')[offset] = scale
    return cache, tail


def valid_logits(seed, request, old_count, valid_count):
    """Only valid keys; masked/uninitialized GPU columns never enter this oracle."""
    import numpy as np
    require(type(old_count) is int and type(valid_count) is int and 512 <= old_count <= valid_count <= POOLS,
            'invalid logit bounds')
    result = ((ranks(seed,request)[:valid_count]+1)/65536).astype('<f4')
    vector, scale = compressed_key(request)
    result[old_count:] = fp8_value(int(vector[0]))*scale
    return result


def score_indices(indices, seed, case):
    import numpy as np
    cfg = case_config(seed,case); rows = len(cfg['positions'])
    require(isinstance(indices,np.ndarray) and indices.dtype == np.dtype('<i4') and indices.shape == (rows,2048), 'indexer output shape/dtype')
    groups = indices[:,:2044].reshape(rows,511,4)
    require(np.all(groups[:,:,0] >= 0) and np.all(groups[:,:,0] < CAP) and np.all(groups[:,:,0] % 4 == 0), 'invalid pool base')
    require(np.array_equal(groups,groups[:,:,:1]+np.arange(4)), 'incomplete expanded pool')
    pools = groups[:,:,0]//4
    require(np.all(np.diff(np.sort(pools,axis=1),axis=1)>0), 'duplicate selected pool')
    allowed_sets = {}
    for request,old_count in set(zip(cfg['row_requests'],cfg['history_counts'])):
        allowed = np.zeros(POOLS,dtype='bool'); allowed[top_history(seed,int(request),int(old_count))] = True
        allowed_sets[(int(request),int(old_count))] = allowed
    for i,(request,position,old_count) in enumerate(zip(cfg['row_requests'],cfg['positions'],cfg['history_counts'])):
        allowed = allowed_sets[(int(request),int(old_count))]
        require(np.all(allowed[pools[i]]), 'pool outside request historical top512')
        require(np.all(groups[i] <= position), 'future token selected')
        tail = np.full(4,-1,dtype='<i4'); count = (int(position)+1)%4
        tail[:count] = np.arange(int(position)//4*4,int(position)//4*4+count)
        require(np.array_equal(indices[i,2044:],tail), 'incorrect tail or padding')
    return {'rows':rows, 'valid_history_groups':rows*511, 'invalid_indices':0}
