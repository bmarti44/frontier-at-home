"""New evidence copy of45.run_native with512-token capacity and chunk capture.

No serving source is patched. Dependencies are explicitly injected by probe.main.
The native model/cache constructors and indexer function remain stock installed APIs.
"""
import gzip
import hashlib
import json

def run_native(metadata, output, seed, selected_case, budget_mib, record):
    import numpy as np
    import torch
    from dataclasses import replace
    from vllm.config import set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.forward_context import set_forward_context
    from vllm.platforms import current_platform
    from vllm.model_executor.layers.attention.mla_attention import _canonicalize_sparse_mla_kv_cache_dtype
    from vllm.models.glm5next.nvidia.attention import Glm5NextIndexerCache, Glm5NextTailCache
    from vllm.model_executor.layers.sparse_attn_indexer_kpool import sparse_attn_indexer_kpool
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import FlashInferMLASparseSM120Backend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend, KpoolTailBackend
    from vllm.v1.worker.utils import AttentionGroup
    from vllm.v1.worker.workspace import init_workspace_manager, current_workspace_manager
    import vllm.utils.deep_gemm as deep_gemm

    import vllm.envs as envs
    require(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB==budget_mib, 'native startup logits budget')
    require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (12,1), 'SM121 required')
    device=torch.device('cuda',0)
    cfg=EngineArgs(model=str(metadata),skip_tokenizer_init=True,dtype='bfloat16',quantization='exl3',
                   max_model_len=262144,max_num_seqs=4,max_num_batched_tokens=512,
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
                'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved(),'device_free':free,'device_total':total,'workspace_bytes':workspace_bytes()}

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
        topk=torch.empty((512,2048),device=device,dtype=torch.int32)
        # Retained inputs use full scheduler capacity; smaller cases take views.
        hidden=stage('hidden',np.zeros((512,4096),dtype='<u2'),torch.bfloat16)
        qbits=np.zeros((512,32,128),dtype='uint8'); qbits[:,:,0]=0x38
        q=stage('query',qbits,current_platform.fp8_dtype())
        weights=stage('weights',np.full((512,32),1/32,dtype='<f4'))
        gate=stage('gate',np.zeros((512,128),dtype='<u2'),torch.bfloat16)
        ape=stage('ape',np.zeros((4,128),dtype='<f4'))
        keys=stage('keys',np.zeros((512,128),dtype='<u2'),torch.bfloat16)
        def invoke(count,positions):
            return sparse_attn_indexer_kpool(hidden[:count],prefix,cache,q[:count],None,keys[:count],weights[:count],
                128,'ue8m0',2048,128,262144,10485760,topk,False,
                gate_score=gate[:count],compress_ape=ape,index_kpool=4,positions=positions,tail_kv_cache=tail,tail_prefix=tail_prefix)
        require(readback.is_pinned() and all(t.is_pinned() for t in hosts.values()), 'indexer staging not pinned')
        record({'event':'configured',**geometry(seed)})
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        with set_forward_context(None,cfg,num_tokens=512): invoke(512,None)
        torch.cuda.synchronize()
        record({'event':'profiled','workspace_bytes':workspace_bytes(),'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_reserved':torch.cuda.memory_reserved(),'device_total':torch.cuda.mem_get_info()[1]})
        require(workspace_bytes()==WORKSPACE, 'native profiling workspace changed')

        for case in [selected_case]:
            config=fixture.case_config(seed,case); count=len(config['positions']); n=len(config['requests'])
            initial_cache,initial_tail=fixture.cache_and_tail(seed,case)
            # Copies use persistent host staging; transfer temporaries are released before measurement.
            temp=stage('cache',initial_cache); cache.copy_(temp.reshape(cache.shape)); del temp
            temp=stage('tail',initial_tail,torch.bfloat16); tail.copy_(temp); del temp
            raw=np.zeros((512,128),dtype='<u2')
            for row,(r,p) in enumerate(zip(config['row_requests'],config['positions'])):
                raw[row]=fixture.bf16_bits(fixture.raw_value(int(r),int(p)%4))
            hosts['keys'].copy_(torch.from_numpy(raw)); keys.copy_(hosts['keys'].view(torch.bfloat16),non_blocking=True)
            # Capacity is fixed across cases, including CPU copies retained by metadata.
            start_values=np.zeros(5,dtype='<i4'); start_values[:n+1]=config['starts']
            end_values=np.zeros(4,dtype='<i4'); end_values[:n]=config['ends']
            pos_values=np.zeros(512,dtype='<i8'); pos_values[:count]=config['positions']
            starts=stage('starts',start_values)[:n+1]; ends=stage('ends',end_values)[:n]
            positions=stage('positions',pos_values)[:count]
            table,tail_table=fixture.tables(seed,case)
            table_values=np.zeros((4,4216),dtype='<i4'); table_values[:n]=table
            tail_values=np.zeros((4,1),dtype='<i4'); tail_values[:n]=tail_table
            gpu_table=stage('table',table_values)[:n]; gpu_tail_table=stage('tail_table',tail_values)[:n]
            # Real token slot IDs are supplied even though each specialized builder reconstructs its mapping.
            slot_values=np.full(512,-1,dtype='<i8')
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
            require(meta.num_decodes==0 and meta.decode is None and meta.prefill is not None, 'prefill-only worker metadata')
            observed={'num_decodes':meta.num_decodes,'num_decode_tokens':meta.num_decode_tokens,
                      'num_prefills':meta.num_prefills,'num_prefill_tokens':meta.num_prefill_tokens,
                      'index_slots':listed(meta.slot_mapping),'tail_slots':listed(tail_meta.slot_mapping),'chunks':[]}
            for chunk in meta.prefill.chunks:
                observed['chunks'].append({'start':chunk.token_start,'stop':chunk.token_end,'total_seq_lens':chunk.total_seq_lens,
                    'skip_gather':chunk.skip_kv_gather,'block_table':listed(chunk.block_table),
                    'ks':listed(chunk.cu_seqlen_ks),'ke':listed(chunk.cu_seqlen_ke)})
            require(observed==expected_metadata(seed,case),'actual indexer metadata differs')
            # Verify actual device input bytes using the retained pinned readback.
            input_digests={name:hashlib.sha256(memoryview(read(tensor).view(torch.uint8).numpy()).cast('B')).hexdigest()
                for name,tensor in {'cache':cache,'tail':tail,'keys':keys,'query':q,'weights':weights,'gate':gate,'ape':ape,'hidden':hidden}.items()}
            require(input_digests==expected_inputs(seed,case),'device input bytes changed')
            calls=[]; specs=call_specs(case)
            original_prefill=deep_gemm.fp8_fp4_mqa_logits
            # Canonical stream is globally ordered by query row, independent of chunks.
            logit_path=output/'logits.f32.gz';logit_stream=gzip.open(logit_path,'wb')
            def prefill_capture(*args,**kwargs):
                index=len(calls);require(index<len(specs),'extra native logit call');spec=specs[index]
                result=original_prefill(*args,**kwargs)
                require(result.dtype==torch.float32 and list(result.shape)==[spec['stop']-spec['start'],spec['columns']], 'native logit shape/dtype mismatch')
                allocated=torch.cuda.memory_allocated()
                actual=read(result).numpy();ks,ke=bounds(seed,case,index)
                for row,(lo,hi) in enumerate(zip(ks,ke)):
                    logit_stream.write(memoryview(actual[row,int(lo):int(hi)]).cast('B'))
                calls.append({**spec,'cuda_allocated_at_return':allocated})
                return result
            deep_gemm.fp8_fp4_mqa_logits=prefill_capture
            begin=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            try:
                with set_forward_context({prefix:meta,tail_prefix:tail_meta},cfg,num_tokens=count):
                    begin.record(); result=invoke(count,positions); end.record(); torch.cuda.synchronize()
            finally:
                deep_gemm.fp8_fp4_mqa_logits=original_prefill
                logit_stream.close()
            require(len(calls)==len(specs), 'missing native logit call')
            names=['indices.i32.gz','cache.u8.gz','tail.bf16.gz']
            artifacts={name:compressed(output/name,tensor)['sha256'] for name,tensor in zip(names,(result[:count],cache,tail))}
            artifacts['logits.f32.gz']=sha256_file(logit_path)
            record({'event':'output','case':case,'budget_mib':budget_mib,'artifacts':artifacts,'calls':calls,'metadata':observed,'memory':memory(),
                    'cuda_elapsed_ms':begin.elapsed_time(end),'cache_stride':list(cache.stride()),'tail_stride':list(tail.stride()),
                    'output_alias':result.untyped_storage().data_ptr()==topk.untyped_storage().data_ptr(),'input_digests':input_digests})
            del meta,tail_meta,common,starts,ends,positions,gpu_table,gpu_tail_table,slots,result
