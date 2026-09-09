#!/usr/bin/env python3
"""Default-off full-address-space indexer preparation; never a model qualification."""
import argparse
import gzip
import importlib.util
import json
import math
from pathlib import Path
import stat
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
from glm53_contract import sha256_file, strict_json
import glm53_indexer_fixture as fixture

require = fixture.require
QUALIFICATION = 'model_free_post_projection_indexer_falsifier_only'
WORKSPACE = 1385168896
CACHE_BYTES = 4930 * 8448
TAIL_BYTES = 145 * 2 * 4 * 128 * 2
TOPK_BYTES = 2048 * 2048 * 4


def call_specs(case):
    require(case in fixture.CASES, 'invalid indexer case')
    if case.startswith('decode'):
        return [{'kind':'decode', 'start':0, 'stop':sum(fixture.CASES[case]), 'columns':262144}]
    if case == 'prefill-1': return [{'kind':'prefill', 'start':0, 'stop':2048, 'columns':65536}]
    return [{'kind':'prefill', 'start':start, 'stop':start+1024, 'columns':131072} for start in (0,1024)]


def logit_bounds(seed, case, index):
    import numpy as np
    specs = call_specs(case)
    require(type(index) is int and 0 <= index < len(specs), 'invalid indexer call index')
    spec = specs[index]; cfg = fixture.case_config(seed,case)
    positions = cfg['positions'][spec['start']:spec['stop']]
    starts = np.zeros(len(positions), dtype='<i4')
    if case == 'prefill-4': starts[512:] = 65536
    return starts, starts+((positions+1)//4).astype('<i4')


def tensor_names(case):
    require(case in fixture.CASES, 'invalid indexer case')
    return [f'indices-{case}.i32.gz', f'cache-{case}.u8.gz', f'tail-{case}.bf16.gz']


def reader_api():
    spec = importlib.util.spec_from_file_location('indexer_prior_reader', ROOT / 'scripts/41_probe_glm53_kda.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def score_logits(path, digest, seed, case, index):
    """Stream exact valid rows, excluding every masked/uninitialized GPU column."""
    cfg = fixture.case_config(seed,case); spec = call_specs(case)[index]
    starts,ends = logit_bounds(seed,case,index); counts = ends-starts
    size = int(counts.sum())*4
    identity = lambda s: (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    initial = path.lstat()
    require(stat.S_ISREG(initial.st_mode) and initial.st_size <= size+1048576, 'invalid logit artifact')
    require(sha256_file(path) == digest, 'logit digest mismatch')
    references = {}
    with gzip.open(path,'rb') as stream:
        for local,row in enumerate(range(spec['start'],spec['stop'])):
            request,old = int(cfg['row_requests'][row]),int(cfg['history_counts'][row])
            key=(request,old)
            if key not in references: references[key] = fixture.valid_logits(seed,request,old,65536)
            target = references[key][:int(counts[local])].tobytes()
            require(stream.read(len(target)) == target, 'valid logit bytes differ or are missing')
        require(stream.read(1) == b'', 'extra logit payload')
    require(identity(path.lstat()) == identity(initial) and sha256_file(path) == digest, 'logit artifact changed')
    return {'valid_logit_elements':size//4, 'valid_logit_byte_mismatches':0, 'nonfinite_valid_logits':0}


def score_tensors(paths, hashes, seed, case):
    import numpy as np
    require(len(paths) == len(hashes) == 3, 'indexer tensor coverage')
    count=sum(fixture.CASES[case]); sizes=[count*2048*4,CACHE_BYTES,TAIL_BYTES]
    blobs=[reader_api().read_tensor(p,h,n) for p,h,n in zip(paths,hashes,sizes)]
    checked=fixture.score_indices(np.frombuffer(blobs[0],dtype='<i4').reshape(count,2048),seed,case)
    cache,tail=fixture.cache_and_tail(seed,case,final=True)
    require(blobs[1] == cache.tobytes(), 'indexer cache bytes differ')
    require(blobs[2] == tail.tobytes(), 'indexer tail bytes differ')
    return {**checked,'cache_bytes_checked':CACHE_BYTES,'tail_bytes_checked':TAIL_BYTES,
            'cache_byte_mismatches':0,'tail_byte_mismatches':0}


def validate_memory(row, case):
    keys={'cuda_allocated','cuda_reserved','cuda_peak_allocated','device_free','device_total','workspace_bytes'}
    require(isinstance(row,dict) and set(row) == keys and all(type(v) is int and v > 0 for v in row.values()), 'invalid indexer memory fields')
    base=WORKSPACE+CACHE_BYTES+TAIL_BYTES+TOPK_BYTES
    require(row['workspace_bytes'] == WORKSPACE and base <= row['cuda_allocated'] <= row['cuda_peak_allocated'] <= row['cuda_reserved'] <= row['device_total'] and
            row['device_free'] < row['device_total'], 'indexer memory accounting mismatch')
    logits = 1073741824 if case == 'prefill-4' else 536870912 if case == 'prefill-1' else sum(fixture.CASES[case])*262144*4
    require(row['cuda_peak_allocated'] >= base+logits, 'missing native logits/workspace allocation')


def expected_metadata(seed, case):
    cfg=fixture.case_config(seed,case); blocks=fixture.physical_blocks(seed)
    common,_=fixture.tables(seed,case); table=(common[:,::4]//4).tolist()
    slots=[];tails=[]
    for r,p in zip(cfg['row_requests'],cfg['positions']):
        r,p=int(r),int(p); pool=p//4; block,offset=divmod(pool,2176)
        slots.append(blocks[r][block]*2176+offset if p%4 == 3 else -1)
        tails.append(blocks[r][31]*4+p%4)
    decode=case.startswith('decode'); n=len(cfg['requests']); count=len(slots)
    return {'num_decodes':n if decode else 0,'num_decode_tokens':count if decode else 0,
            'num_prefills':0 if decode else n,'num_prefill_tokens':0 if decode else count,
            'index_slots':slots,'tail_slots':tails,'block_table':table,
            'decode_lengths':[[end//4] for end in cfg['ends']] if decode else None,
            'chunks':[] if decode else [{'start':s['start'],'stop':s['stop'],'total_seq_lens':s['columns'],
                       'ks':logit_bounds(seed,case,i)[0].tolist(),'ke':logit_bounds(seed,case,i)[1].tolist()}
                      for i,s in enumerate(call_specs(case))]}


def geometry(seed):
    return {'case_order':fixture.case_order(seed),'request_order':fixture.request_order(seed),
            'physical_blocks':fixture.physical_blocks(seed),'cache_shape':[4930,64,132],
            'tail_shape':[145,2,4,128],'max_model_len':262144,'max_num_seqs':4,
            'max_num_batched_tokens':2048,'gather_rows':10485760,'heads':32,'head_dim':128,
            'pinned_staging':True,'startup_selection':'pinned_indexer'}


def validate_capture(root, rows, seed):
    order=fixture.case_order(seed)
    require(len(rows) == 2+2*len(order), 'indexer row coverage')
    times=[r.get('time_unix') for r in rows]
    require(all(type(t) in (int,float) and math.isfinite(t) and t > 0 for t in times) and
            all(b>a for a,b in zip(times,times[1:])), 'indexer timestamps invalid')
    require(json.dumps(rows[0],sort_keys=True) == json.dumps({'time_unix':times[0],'event':'configured',**geometry(seed)},sort_keys=True), 'indexer startup geometry mismatch')
    profile=rows[1]
    require(set(profile) == {'time_unix','event','workspace_bytes','cuda_peak_allocated','cuda_reserved','device_total'} and profile['event']=='profiled' and
            all(type(profile[k]) is int for k in ('workspace_bytes','cuda_peak_allocated','cuda_reserved','device_total')) and
            profile['workspace_bytes']==WORKSPACE and WORKSPACE+CACHE_BYTES+TAIL_BYTES+TOPK_BYTES+536870912 <= profile['cuda_peak_allocated'] <= profile['cuda_reserved'] <= profile['device_total'], 'indexer profiling allocation mismatch')
    names={'manifest.json','summary.json','raw.jsonl','traceback.log'}; checks=[]
    for i,case in enumerate(order):
        start,row=rows[2+2*i:4+2*i]
        require(start == {'time_unix':start.get('time_unix'),'event':'start','case':case}, 'indexer case order mismatch')
        require(set(row)=={'time_unix','event','case','artifacts','calls','metadata','memory','cuda_elapsed_ms','cache_stride','tail_stride','output_alias'} and
                row['event']=='output' and row['case']==case, 'indexer output schema')
        require(type(row['cuda_elapsed_ms']) in (int,float) and math.isfinite(row['cuda_elapsed_ms']) and row['cuda_elapsed_ms'] > 0, 'indexer timing invalid')
        require(json.dumps(row['cache_stride']) == '[8448, 132, 1]' and json.dumps(row['tail_stride']) == '[1024, 512, 128, 1]' and row['output_alias'] is True, 'indexer tensor layout/alias mismatch')
        require(json.dumps(row['metadata'],sort_keys=True) == json.dumps(expected_metadata(seed,case),sort_keys=True), 'indexer native metadata mismatch')
        validate_memory(row['memory'],case)
        tensor_files=tensor_names(case); items=row['artifacts']
        require(isinstance(items,list) and len(items)==3 and all(isinstance(x,dict) and set(x)=={'file','sha256'} and x['file']==n for x,n in zip(items,tensor_files)), 'indexer tensor artifact schema')
        names.update(tensor_files)
        calls=row['calls']; specs=call_specs(case)
        require(isinstance(calls,list) and len(calls)==len(specs), 'indexer native call coverage')
        for j,(call,spec) in enumerate(zip(calls,specs)):
            name=f'logits-{case}-{j}.f32.gz'; names.add(name)
            require(isinstance(call,dict) and set(call)=={'kind','start','stop','columns','file','sha256','cuda_allocated_at_return'} and
                    all(type(call[k]) is int for k in ('start','stop','columns','cuda_allocated_at_return')) and
                    {k:call[k] for k in spec}==spec and call['file']==name, 'indexer native call receipt mismatch')
            overlap=536870912*(j+1) if case.startswith('prefill') else sum(fixture.CASES[case])*262144*4
            require(WORKSPACE+CACHE_BYTES+TAIL_BYTES+TOPK_BYTES+overlap <= call['cuda_allocated_at_return'] <= row['memory']['cuda_peak_allocated'], 'native logit lifetime overlap missing')
        checks.append(row)
    require({p.name for p in root.iterdir()}==names, 'indexer file coverage mismatch')
    return checks


def score_capture(root, rows, seed):
    captures=validate_capture(root,rows,seed); checks=[]
    for row in captures:
        case=row['case']; items=row['artifacts']
        tensors=score_tensors([root/x['file'] for x in items],[x['sha256'] for x in items],seed,case)
        logits=[score_logits(root/call['file'],call['sha256'],seed,case,i) for i,call in enumerate(row['calls'])]
        checks.append({'case':case,**tensors,'logits':logits})
    return checks


def run_native(metadata, output, seed, record):
    import numpy as np
    import torch
    from dataclasses import replace
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.attention.mla_attention import _canonicalize_sparse_mla_kv_cache_dtype
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import sparse_attn_indexer_kpool
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache, Glm5NextTailCache
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend, KpoolTailBackend
    from vllm.v1.worker.utils import AttentionGroup
    from vllm.v1.worker.workspace import init_workspace_manager, current_workspace_manager
    import vllm.utils.deep_gemm as deep_gemm

    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12,1), 'SM121 required')
    device=torch.device('cuda',0)
    cfg=EngineArgs(model=str(metadata),skip_tokenizer_init=True,dtype='bfloat16',quantization='exl3',
                   max_model_len=262144,max_num_seqs=4,max_num_batched_tokens=2048,
                   enable_chunked_prefill=True,enable_prefix_caching=False,kv_cache_dtype='fp8',
                   compilation_config={'mode':0}).create_engine_config()
    require(cfg.speculative_config is None and not cfg.cache_config.enable_prefix_caching, 'indexer speculation/cache mismatch')
    require(all(getattr(cfg.parallel_config,n)==1 for n in ('tensor_parallel_size','pipeline_parallel_size','decode_context_parallel_size','prefill_context_parallel_size')), 'indexer single-device configuration required')
    text=cfg.model_config.hf_text_config
    require(text.index_n_heads==32 and text.index_head_dim==128 and text.index_topk==2048 and text.index_kpool==4, 'indexer model geometry mismatch')
    prefix='model.language_model.layers.3.self_attn.indexer.k_cache'
    tail_prefix='model.language_model.layers.3.self_attn.indexer.tail_cache'
    # One retained page-locked readback arena, including the largest returned logits.
    readback=torch.empty(536870912,dtype=torch.uint8,pin_memory=True)
    hosts={}
    def stage(name, array, dtype=None):
        source=torch.from_numpy(np.ascontiguousarray(array))
        if name not in hosts: hosts[name]=torch.empty_like(source,pin_memory=True)
        host=hosts[name]; require(host.shape==source.shape and host.dtype==source.dtype and host.is_pinned(), 'input staging geometry/pinning changed')
        host.copy_(source)
        result=host.to(device,non_blocking=True)
        return result.view(dtype) if dtype is not None else result
    def read(tensor):
        size=tensor.numel()*tensor.element_size(); require(size<=readback.numel(), 'readback capacity exceeded')
        host=readback[:size].view(tensor.dtype).reshape(tensor.shape)
        host.copy_(tensor,non_blocking=True); torch.cuda.synchronize()
        return host
    def listed(tensor): return read(tensor).tolist()
    def compressed(path,tensor):
        host=read(tensor)
        with gzip.open(path,'wb') as stream: stream.write(memoryview(host.view(torch.uint8).numpy()).cast('B'))
        return {'file':path.name,'sha256':sha256_file(path)}
    def workspace_bytes():
        arenas=current_workspace_manager()._current_workspaces
        require(len(arenas)==1 and arenas[0] is not None, 'unexpected indexer workspace arenas')
        return arenas[0].numel()*arenas[0].element_size()
    def memory():
        free,total=torch.cuda.mem_get_info()
        return {'cuda_allocated':torch.cuda.memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),
                'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'device_free':free,'device_total':total,'workspace_bytes':workspace_bytes()}

    with set_current_vllm_config(cfg):
        backend=FlashInferMLASparseSM120Backend
        cfg.cache_config.cache_dtype=_canonicalize_sparse_mla_kv_cache_dtype(backend,cfg.cache_config.cache_dtype)
        cfg.cache_config.block_size=backend.get_preferred_block_size(cfg.cache_config.block_size)
        current_platform._align_hybrid_block_size(cfg,backend)
        require(cfg.cache_config.block_size==8704 and cfg.cache_config.mamba_block_size==262144, 'indexer normalized block geometry mismatch')
        cache_layer=Glm5NextIndexerCache(head_dim=132,dtype=torch.uint8,prefix=prefix,cache_config=cfg.cache_config,index_kpool=4)
        tail_layer=Glm5NextTailCache(head_dim=128,dtype=torch.bfloat16,prefix=tail_prefix,cache_config=cfg.cache_config,index_kpool=4)
        index_spec=cache_layer.get_kv_cache_spec(cfg); tail_spec=tail_layer.get_kv_cache_spec(cfg)
        require(index_spec.storage_block_size==256 and index_spec.tokens_per_state==4 and tail_spec.block_size==4, 'indexer cache specs changed')
        group=AttentionGroup(DeepseekV32IndexerBackend,[prefix],index_spec,0)
        tail_group=AttentionGroup(KpoolTailBackend,[tail_prefix],tail_spec,1)
        group.create_metadata_builders(cfg,device,kernel_block_size=64)
        tail_group.create_metadata_builders(cfg,device,kernel_block_size=4)
        builder=group.get_metadata_builder(); tail_builder=tail_group.get_metadata_builder()
        require(builder.kv_cache_spec.num_states==64 and builder.kernel_block_size==64, 'worker indexer builder adaptation mismatch')
        init_workspace_manager(device)
        cache=torch.empty((4930,64,132),device=device,dtype=torch.uint8)
        tail=torch.empty((145,2,4,128),device=device,dtype=torch.bfloat16)
        topk=torch.empty((2048,2048),device=device,dtype=torch.int32)
        # Retained inputs use full scheduler capacity; smaller cases take views.
        hidden=stage('hidden',np.zeros((2048,4096),dtype='<u2'),torch.bfloat16)
        qbits=np.zeros((2048,32,128),dtype='uint8'); qbits[:,:,0]=0x38
        q=stage('query',qbits,current_platform.fp8_dtype())
        weights=stage('weights',np.full((2048,32),1/32,dtype='<f4'))
        gate=stage('gate',np.zeros((2048,128),dtype='<u2'),torch.bfloat16)
        ape=stage('ape',np.zeros((4,128),dtype='<f4'))
        keys=stage('keys',np.zeros((2048,128),dtype='<u2'),torch.bfloat16)
        def invoke(count,positions):
            return sparse_attn_indexer_kpool(hidden[:count],prefix,cache,q[:count],None,keys[:count],weights[:count],
                128,'ue8m0',2048,128,262144,10485760,topk,False,
                gate_score=gate[:count],compress_ape=ape,index_kpool=4,positions=positions,tail_kv_cache=tail,tail_prefix=tail_prefix)
        require(readback.is_pinned() and all(t.is_pinned() for t in hosts.values()), 'indexer staging not pinned')
        record({'event':'configured',**geometry(seed)})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with set_forward_context(None,cfg,num_tokens=2048): invoke(2048,None)
        torch.cuda.synchronize()
        record({'event':'profiled','workspace_bytes':workspace_bytes(),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),'device_total':torch.cuda.mem_get_info()[1]})
        require(workspace_bytes()==WORKSPACE, 'native profiling workspace changed')

        for case in fixture.case_order(seed):
            config=fixture.case_config(seed,case); count=len(config['positions']); n=len(config['requests'])
            initial_cache,initial_tail=fixture.cache_and_tail(seed,case)
            # Copies use persistent host staging; transfer temporaries are released before measurement.
            temp=stage('cache',initial_cache); cache.copy_(temp.reshape(cache.shape)); del temp
            temp=stage('tail',initial_tail,torch.bfloat16); tail.copy_(temp); del temp
            raw=np.zeros((2048,128),dtype='<u2')
            for row,(r,p) in enumerate(zip(config['row_requests'],config['positions'])):
                raw[row]=fixture.bf16_bits(fixture.raw_value(int(r),int(p)%4))
            hosts['keys'].copy_(torch.from_numpy(raw)); keys.copy_(hosts['keys'],non_blocking=True)
            # Capacity is fixed across cases, including CPU copies retained by metadata.
            start_values=np.zeros(5,dtype='<i4'); start_values[:n+1]=config['starts']
            end_values=np.zeros(4,dtype='<i4'); end_values[:n]=config['ends']
            pos_values=np.zeros(2048,dtype='<i8'); pos_values[:count]=config['positions']
            starts=stage('starts',start_values)[:n+1]; ends=stage('ends',end_values)[:n]
            positions=stage('positions',pos_values)[:count]
            table,tail_table=fixture.tables(seed,case)
            table_values=np.zeros((4,4216),dtype='<i4'); table_values[:n]=table
            tail_values=np.zeros((4,1),dtype='<i4'); tail_values[:n]=tail_table
            gpu_table=stage('table',table_values)[:n]; gpu_tail_table=stage('tail_table',tail_values)[:n]
            # Real token slot IDs are supplied even though each specialized builder reconstructs its mapping.
            slot_values=np.full(2048,-1,dtype='<i8')
            blocks=fixture.physical_blocks(seed)
            for row,(r,p) in enumerate(zip(config['row_requests'],config['positions'])):
                slot_values[row]=blocks[int(r)][int(p)//8704]*8704+int(p)%8704
            slots=stage('slots',slot_values)[:count]
            common=CommonAttentionMetadata(query_start_loc=starts,query_start_loc_cpu=hosts['starts'][:n+1],
                seq_lens=ends,num_reqs=n,num_actual_tokens=count,max_query_len=max(config['lengths']),max_seq_len=max(config['ends']),
                block_table_tensor=gpu_table,slot_mapping=slots,positions=positions,seq_lens_cpu_upper_bound=hosts['ends'][:n])
            torch.cuda.synchronize(); record({'event':'start','case':case}); torch.cuda.reset_peak_memory_stats()
            meta=builder.build(0,common)
            tail_meta=tail_builder.build(0,replace(common,block_table_tensor=gpu_tail_table))
            observed={'num_decodes':meta.num_decodes,'num_decode_tokens':meta.num_decode_tokens,
                      'num_prefills':meta.num_prefills,'num_prefill_tokens':meta.num_prefill_tokens,
                      'index_slots':listed(meta.slot_mapping),'tail_slots':listed(tail_meta.slot_mapping),
                      'block_table':[],'decode_lengths':None,'chunks':[]}
            if meta.decode is not None:
                require(meta.decode.requires_padding is False and meta.decode.decode_is_uniform is True and meta.decode.write_max_decode_len==1, 'unexpected decode grouping')
                observed['block_table']=listed(meta.decode.block_table)
                observed['decode_lengths']=listed(meta.decode.seq_lens)
            if meta.prefill is not None:
                for chunk in meta.prefill.chunks:
                    require(not chunk.skip_kv_gather, 'unexpected skipped gather')
                    observed['block_table'].extend(listed(chunk.block_table))
                    observed['chunks'].append({'start':chunk.token_start,'stop':chunk.token_end,'total_seq_lens':chunk.total_seq_lens,
                                               'ks':listed(chunk.cu_seqlen_ks),'ke':listed(chunk.cu_seqlen_ke)})
            require(json.dumps(observed,sort_keys=True)==json.dumps(expected_metadata(seed,case),sort_keys=True), 'actual indexer metadata differs')
            calls=[]; specs=call_specs(case)
            original_prefill=deep_gemm.fp8_fp4_mqa_logits; original_decode=deep_gemm.fp8_fp4_paged_mqa_logits
            def capture_call(kind, original, args, kwargs):
                index=len(calls); require(index<len(specs) and specs[index]['kind']==kind, 'unexpected native logits call')
                spec=specs[index]
                result=original(*args,**kwargs)
                require(result.dtype==torch.float32 and list(result.shape)==[spec['stop']-spec['start'],spec['columns']], 'native logit shape/dtype mismatch')
                allocated=torch.cuda.memory_allocated()
                actual=read(result).numpy(); ks,ke=logit_bounds(seed,case,index)
                path=output/f'logits-{case}-{index}.f32.gz'
                with gzip.open(path,'wb') as stream:
                    for row,(lo,hi) in enumerate(zip(ks,ke)):
                        stream.write(memoryview(actual[row,int(lo):int(hi)]).cast('B'))
                calls.append({**spec,'file':path.name,'sha256':sha256_file(path),'cuda_allocated_at_return':allocated})
                return result
            def prefill_capture(*args,**kwargs): return capture_call('prefill',original_prefill,args,kwargs)
            def decode_capture(*args,**kwargs): return capture_call('decode',original_decode,args,kwargs)
            # Uninstalled evidence process only. Restore exact function identities even on failure.
            deep_gemm.fp8_fp4_mqa_logits=prefill_capture; deep_gemm.fp8_fp4_paged_mqa_logits=decode_capture
            begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            try:
                with set_forward_context({prefix:meta,tail_prefix:tail_meta},cfg,num_tokens=count):
                    begin.record(); result=invoke(count,positions); end.record(); torch.cuda.synchronize()
            finally:
                deep_gemm.fp8_fp4_mqa_logits=original_prefill; deep_gemm.fp8_fp4_paged_mqa_logits=original_decode
            require(len(calls)==len(specs), 'missing native logit call')
            artifacts=[compressed(output/name,tensor) for name,tensor in zip(tensor_names(case),(result[:count],cache,tail))]
            record({'event':'output','case':case,'artifacts':artifacts,'calls':calls,'metadata':observed,'memory':memory(),
                    'cuda_elapsed_ms':begin.elapsed_time(end),'cache_stride':list(cache.stride()),'tail_stride':list(tail.stride()),
                    'output_alias':result.untyped_storage().data_ptr()==topk.untyped_storage().data_ptr()})
            score_tensors([output/a['file'] for a in artifacts],[a['sha256'] for a in artifacts],seed,case)
            for i,call in enumerate(calls): score_logits(output/call['file'],call['sha256'],seed,case,i)
            del meta,tail_meta,common,starts,ends,positions,gpu_table,gpu_tail_table,slots,result


def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode, 'unoptimized no-bytecode Python required')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True); parser.add_argument('--pinned-indexer',action='store_true')
    args=parser.parse_args(); fixture.case_order(args.seed); require(args.pinned_indexer, 'explicit --pinned-indexer required')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest={'qualification':QUALIFICATION,'seed':args.seed,'startup_selection':'pinned_indexer',
              'scorer_sha256':sha256_file(Path(__file__)),'binary_sha256':sha256_file(Path(sys.executable).resolve()),
              'decision':{'sha256':sha256_file(ROOT/'configs/decision-specs/glm53-indexer-preflight.json')},
              'fixture':{'sha256':sha256_file(ROOT/'scripts/lib/glm53_indexer_fixture.py')},
              'metadata':{p.name:{'sha256':sha256_file(p)} for p in args.metadata.iterdir() if p.is_file()},'start_unix':time.time()}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); failure=None
    with (args.output/'raw.jsonl').open('w') as raw,(args.output/'traceback.log').open('w') as errors:
        def record(row): raw.write(json.dumps({'time_unix':time.time(),**row},allow_nan=False)+'\n'); raw.flush()
        try: run_native(args.metadata,args.output,args.seed,record)
        except Exception as error:
            traceback.print_exc(file=errors); failure=repr(error); record({'event':'failure','failure':failure})
    (args.output/'summary.json').write_text('{}\n')
    checks=[]
    if failure is None:
        try: checks=score_capture(args.output,[json.loads(line) for line in (args.output/'raw.jsonl').read_text().splitlines()],args.seed)
        except Exception as error: failure=repr(error)
    summary={'verdict':'FAIL' if failure else 'PASS','qualification':QUALIFICATION,'model_loaded':False,
             'actual_input_tokens_processed':0,'context_capability':'not measured','performance':'not measured',
             'raw_sha256':sha256_file(args.output/'raw.jsonl'),'checks':checks,'failure':failure}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,allow_nan=False)); raise SystemExit(1 if failure else 0)


if __name__=='__main__': main()
