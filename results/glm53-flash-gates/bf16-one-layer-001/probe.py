"""Explicit real-weight/synthetic-activation feasibility probe; no serving imports."""
from pathlib import Path
import argparse,gc,gzip,hashlib,importlib.util,inspect,json,math,struct,time,urllib.request

def require(ok,message):
    if not ok: raise ValueError(message)

def strict(data):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    def constant(value):raise ValueError('nonfinite JSON constant')
    return json.loads(data,object_pairs_hook=pairs,parse_constant=constant)

def sha(data):return hashlib.sha256(data).hexdigest()

INPUT_SPEC={'generator':'shake256-bf16-small-v1','seed_encoding':'unsigned-64-big-endian','tensor_byte_order':'little-endian','shape':[1,516,4,4096],'dtype':'torch.bfloat16'}

def canonical_input(seed):
    require(type(seed) is int and 0<=seed<2**64,'canonical input seed')
    # BF16 sign is random, exponent is 120, mantissa is random: small finite HC.
    data=bytearray(hashlib.shake_256(b'glm53-bf16-layer-input-v1\0'+seed.to_bytes(8,'big')).digest(516*4*4096*2))
    data[::2]=data[::2].translate(bytes(i & 127 for i in range(256)))
    data[1::2]=data[1::2].translate(bytes(0x3c | (i & 128) for i in range(256)))
    return data

def stock_selector(native):
    rows=[]
    for name in ['causal_conv1d_fn','causal_conv1d_update','chunk_kimi_delta_attention','recurrent_kimi_delta_attention']:
        fn=inspect.unwrap(getattr(native,name))
        require(fn.__module__==native.__name__ and fn.__globals__ is native.__dict__,'native fallback identity')
        require(Path(fn.__code__.co_filename).resolve()==Path(native.__file__).resolve(),'native fallback file')
        rows.append({'name':name,'source':{'sha256':sha(inspect.getsource(fn).encode())}})
    return {'module':{'path':native.__file__,'sha256':sha(Path(native.__file__).read_bytes())},'functions':rows}

def validate_shard(buffer,record,expected_header):
    require(len(buffer)==record['size'],'shard byte count')
    require(sha(buffer)==record['lfs']['sha256'],'whole shard digest')
    require(len(buffer)>=8,'missing safetensors prefix')
    length=struct.unpack('<Q',memoryview(buffer)[:8])[0]
    require(2<=length<=16*1024**2 and 8+length<=len(buffer),'safetensors header bounds')
    header=strict(memoryview(buffer)[8:8+length].tobytes());header.pop('__metadata__',None)
    require(header==expected_header,'pinned shard header differs')
    end=0
    for name,value in sorted(header.items(),key=lambda kv:kv[1]['data_offsets'][0]):
        require(value['dtype'] in ('BF16','F32'),'unsupported native dtype')
        dims=value['shape'];require(all(type(n) is int and n>0 for n in dims),'tensor shape')
        left,right=value['data_offsets'];require(type(left) is int and type(right) is int and left==end,'tensor offsets')
        require(right-left==math.prod(dims)*(2 if value['dtype']=='BF16' else 4),'tensor byte geometry');end=right
    require(8+length+end==len(buffer),'unclaimed shard bytes')
    return 8+length,header

def download_shard(buffer,url,progress):
    spec=importlib.util.spec_from_file_location('native_range_transport',Path(__file__).with_name('range_download.py'))
    transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    return transport.download(buffer,url,progress)

def run(root):
    import torch
    from transformers import Glm5NextConfig
    import transformers.models.glm5_next.modeling_glm5_next as native
    from transformers.core_model_loading import convert_and_load_state_dict_in_model
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.modeling_utils import LoadStateDictConfig
    manifest=strict((root/'manifest.json').read_bytes());seed=strict((root/'randomness.json').read_bytes())['seed']
    out=root/'checks';out.mkdir(exist_ok=False)
    def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    def event(**row):
        row.setdefault('host',host())
        with (out/'raw.jsonl').open('a') as f:f.write(json.dumps({'time_unix':time.time(),**row},allow_nan=False)+'\n')
    def host():
        m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());v=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
        return {'time_unix':time.time(),'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'used_swap_kib':int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0]),'available_kib':int(m['MemAvailable'].split()[0])}
    def check_host():
        x=host();a=manifest['broad_baseline'];require(all(x[k]==a[k] for k in ('pswpin','pswpout','used_swap_kib')),'broad host swap changed');require(x['available_kib']>=40*1024**2,'host floor');return x
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest');check_host()
        selectors=stock_selector(native)
        require(selectors==strict((root/'metadata/layer-plan.json').read_bytes())['stock_torch_selector'],'frozen fallback selector differs')
        for row in selectors['functions']:setattr(native,row['name'],inspect.unwrap(getattr(native,row['name'])))
        save('stock-torch-selector.json',selectors)
        config=Glm5NextConfig.from_dict(strict((root/'metadata/config.json').read_bytes()))
        with torch.device('meta'):
            model=native.Glm5NextForConditionalGeneration._from_config(config,dtype=torch.bfloat16,attn_implementation='eager',experts_implementation='eager').eval()
        require(all(p.device.type=='meta' for p in model.parameters()),'meta initialization allocated weights')
        prefix='model.language_model.layers.8.';all_names=set(model.state_dict());want={k for k in all_names if k.startswith(prefix)}
        expected_native={k for k in strict((root/'metadata/model.safetensors.index.json').read_bytes())['weight_map'] if k.startswith(prefix)}
        require(manifest['transport']=={'connections':4,'whole_shard_verification_before_use':True},'frozen transport selection')
        loaded={};downloaded=[]
        for record in manifest['shards']:
            check_host();name=record['rfilename'];buffer=bytearray(record['size']);position=0;began=time.time();event(kind='download_start',shard=name,bytes=record['size'])
            url='https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/'+manifest['model_revision']+'/'+name+'?download=true'
            receipt=download_shard(buffer,url,lambda count:event(kind='download_progress',shard=name,bytes=count,host=check_host()))
            save(name+'.transport.json',receipt)
            offset,header=validate_shard(buffer,record,strict((root/'metadata'/ (name+'.header.json')).read_bytes()))
            for key,value in header.items():
                if not key.startswith(prefix):continue
                require(key not in loaded,'duplicate layer tensor');dtype=torch.bfloat16 if value['dtype']=='BF16' else torch.float32
                t=torch.frombuffer(buffer,dtype=dtype,count=math.prod(value['shape']),offset=offset+value['data_offsets'][0]).reshape(value['shape']).clone()
                loaded[key]=t
            downloaded.append({'path':name,'size_bytes':len(buffer),'sha256':sha(buffer)});event(kind='verified_shard',shard=downloaded[-1],seconds=time.time()-began,selected_tensors=len(loaded),host=check_host())
            del t,buffer,header;gc.collect()
        require(set(loaded)==expected_native,'native layer input coverage')
        load_config=LoadStateDictConfig(dtype=torch.bfloat16,dtype_plan=model._get_dtype_plan(torch.bfloat16),device_map={'':'cpu'},weight_mapping=get_model_conversion_mapping(model))
        info,offload=convert_and_load_state_dict_in_model(model,loaded,load_config)
        require(not offload and not info.unexpected_keys and not info.mismatched_keys and not info.error_msgs and not info.conversion_errors,'native conversion failure')
        require(info.missing_keys==all_names-want,'partial loading-info coverage')
        actual=model.state_dict();live={k for k,v in actual.items() if v.device.type!='meta'};require(live==want,'unexpected materialized model tensor')
        strict32=model._keep_in_fp32_modules_strict
        for name in want:
            dtype=torch.float32 if any(x in name.split('.') for x in strict32) else torch.bfloat16
            require(actual[name].dtype==dtype,'native dtype plan mismatch: '+name)
        expected_plan=strict((root/'metadata/layer-plan.json').read_bytes())['tensors']
        require({k.removeprefix(prefix):{'shape':list(actual[k].shape),'dtype':str(actual[k].dtype),'bytes':actual[k].numel()*actual[k].element_size()} for k in want}==expected_plan,'frozen converted tensor schema')
        del actual,loaded;gc.collect();check_host()
        layer=model.model.language_model.layers[8];named=[(n,p,'parameter') for n,p in layer.named_parameters()]+[(n,b,'buffer') for n,b in layer.named_buffers()]
        staging={name:p.detach().pin_memory() for name,p,kind in named}
        require(all(p.is_pinned() for p in staging.values()),'pageable weight staging')
        records=[]
        for name,p,tensor_kind in named:
            source=staging[name];target=source.to('cuda',non_blocking=True);parent,_,leaf=name.rpartition('.');module=layer.get_submodule(parent) if parent else layer;setattr(module,leaf,torch.nn.Parameter(target,requires_grad=False) if tensor_kind=='parameter' else target)
            records.append({'name':name,'shape':list(source.shape),'dtype':str(source.dtype),'bytes':source.numel()*source.element_size(),'pinned':source.is_pinned(),'sha256':sha(source.contiguous().view(torch.uint8).numpy())})
        torch.cuda.synchronize();del named,p,target,source;gc.collect()
        require(all(p.device.type=='cuda' for p in layer.state_dict().values()),'layer GPU placement')
        require(all(p.device.type=='meta' for n,p in model.named_parameters() if not n.startswith(prefix)),'other weights materialized')
        save('staging.json',{'persistent_until_forward_completion':True,'weights':records,'converted_layer_bytes':sum(x['bytes'] for x in records),'native_source_tensors':len(expected_native),'downloaded_shards':downloaded})
        for name,param in layer.state_dict().items():
            expected=sha(staging[name].contiguous().view(torch.uint8).numpy())
            observed=sha(param.detach().cpu().contiguous().view(torch.uint8).numpy())
            require(observed==expected,'H2D weight bytes differ: '+name)
            event(kind='verified_gpu_tensor',name=name,sha256=observed)
        event(kind='weights_on_gpu',host=check_host(),cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
        n=manifest['input_tokens'];require(n==516,'fixed input shape changed')
        require(manifest['input_spec']==INPUT_SPEC,'frozen input specification')
        input_bytes=canonical_input(seed);source_input=torch.frombuffer(input_bytes,dtype=torch.bfloat16).reshape(INPUT_SPEC['shape']).pin_memory();del input_bytes
        require(source_input.is_pinned(),'pageable activation staging');x=source_input.to('cuda',non_blocking=True);torch.cuda.synchronize()
        mask=torch.ones((1,n),dtype=torch.bool,device='cuda');positions=torch.arange(n,device='cuda').unsqueeze(0)
        torch.cuda.reset_peak_memory_stats();began=time.time()
        with torch.inference_mode():y,topk=layer(x,attention_mask=mask,position_ids=positions,position_embeddings=None,input_ids=None,past_key_values=None,prev_topk_indices=None)
        torch.cuda.synchronize();require(y.shape==x.shape and y.dtype==torch.bfloat16 and bool(torch.isfinite(y).all()),'invalid complete layer output');require(topk is None,'unexpected KDA topk')
        event(kind='forward_complete',elapsed_seconds=time.time()-began,host=check_host(),cuda_allocated=torch.cuda.memory_allocated(),cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
        output=y.cpu().contiguous().view(torch.uint8).numpy().tobytes();(out/'output.bf16.gz').write_bytes(gzip.compress(output,mtime=0))
        save('output.json',{'shape':list(y.shape),'dtype':str(y.dtype),'bytes':len(output),'sha256':sha(output),'seed':seed,'input':{'shape':list(x.shape),'dtype':str(x.dtype),'sha256':sha(source_input.view(torch.uint8).numpy().tobytes()),'bytes':source_input.numel()*source_input.element_size(),'pinned':source_input.is_pinned()}})
        save('summary.json',{'verdict':'PASS','scope':'One native BF16 KDA layer with synthetic activations; host verdict required separately','native_model_reference':False,'full_model_loaded':False,'layer':8,'input_tokens_shape':n,'converted_layer_bytes':sum(x['bytes'] for x in records),'finite_output':True,'pinned_weight_and_activation_staging':True,'unrelated_model_weights_remained_meta':True})
    except BaseException as error:
        event(kind='failure',error=repr(error));save('summary.json',{'verdict':'FAIL','scope':'One-layer feasibility only','error':repr(error),'native_model_reference':False});raise

def score(root):
    import array,sys,re
    m=strict((root/'manifest.json').read_bytes());out=root/'checks';seed=strict((root/'randomness.json').read_bytes())
    require(type(seed['seed']) is int and seed['seed']==int(seed['randomness'][:16],16),'seed conversion mismatch')
    rows=[strict(x) for x in (out/'raw.jsonl').read_bytes().splitlines()]
    require(rows and not any(x['kind']=='failure' for x in rows),'missing raw or inner failure')
    def finite(x):return type(x) in (int,float) and math.isfinite(x)
    def uint(x):return type(x) is int and x>=0
    def digest(x):return type(x) is str and re.fullmatch('[0-9a-f]{64}',x) is not None
    times=[x['time_unix'] for x in rows];require(all(finite(t) and t>0 for t in times) and all(b>a for a,b in zip(times,times[1:])),'raw timestamps')
    fields={'download_start':{'shard','bytes'},'download_progress':{'shard','bytes'},'verified_shard':{'shard','seconds','selected_tensors'},'verified_gpu_tensor':{'name','sha256'},'weights_on_gpu':{'cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved'},'forward_complete':{'elapsed_seconds','cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved'}}
    for row in rows:
        require(row['kind'] in fields,'unknown raw event')
        require(set(row)==fields[row['kind']]|{'kind','time_unix','host'},'missing or unexpected phase fields')
        h=row['host'];require(set(h)=={'time_unix','pswpin','pswpout','used_swap_kib','available_kib'},'phase host fields')
        require(finite(h['time_unix']) and 0<=row['time_unix']-h['time_unix']<=2,'phase host timestamp')
        require(all(uint(h[k]) and h[k]==m['broad_baseline'][k] for k in ('pswpin','pswpout','used_swap_kib')),'phase host swap counters')
        require(uint(h['available_kib']) and h['available_kib']>=40*1024**2,'phase host memory floor')
    stage=strict((out/'staging.json').read_bytes());plan=strict((root/'metadata/layer-plan.json').read_bytes())
    require(strict((out/'stock-torch-selector.json').read_bytes())==plan['stock_torch_selector'],'frozen stock selector binding')
    observed={x['name']:x for x in stage['weights']};require(len(observed)==len(stage['weights'])==len(plan['tensors']),'staging duplicates or coverage')
    require(set(observed)==set(plan['tensors']),'staging parameter names')
    for name,spec in plan['tensors'].items():
        row=observed[name];require(set(row)=={'name','shape','dtype','bytes','pinned','sha256'} and all(row[k]==v for k,v in spec.items()) and row['pinned'] is True and digest(row['sha256']),'staging shape/dtype/pinning/digest')
    require(stage['persistent_until_forward_completion'] is True and stage['converted_layer_bytes']==plan['converted_layer_bytes']==sum(x['bytes'] for x in plan['tensors'].values()) and stage['native_source_tensors']==892,'native staging totals')
    expected_shards=[{'path':x['rfilename'],'size_bytes':x['size'],'sha256':x['lfs']['sha256']} for x in m['shards']]
    require(stage['downloaded_shards']==expected_shards and [x['shard'] for x in rows if x['kind']=='verified_shard']==expected_shards,'verified shard coverage')
    require([x['shard'] for x in rows if x['kind']=='download_start']==[x['path'] for x in expected_shards],'download admission coverage')
    gpu=[x for x in rows if x['kind']=='verified_gpu_tensor'];require(len(gpu)==len(observed) and {x['name'] for x in gpu}==set(observed),'GPU parameter coverage')
    for row in gpu:require(row['sha256']==observed[row['name']]['sha256'],'GPU weight digest mismatch')
    # Exact sequential download/verification lifecycle, followed by GPU validation.
    cursor=0;selected=0
    for shard in expected_shards:
        row=rows[cursor];require(row['kind']=='download_start' and row['shard']==shard['path'] and row['bytes']==shard['size_bytes'],'download phase order/size');cursor+=1;last=0
        while cursor<len(rows) and rows[cursor]['kind']=='download_progress':
            row=rows[cursor];require(row['shard']==shard['path'] and uint(row['bytes']) and last<row['bytes']<=shard['size_bytes'],'download progress');last=row['bytes'];cursor+=1
        row=rows[cursor];require(row['kind']=='verified_shard' and row['shard']==shard and finite(row['seconds']) and row['seconds']>0 and uint(row['selected_tensors']) and selected<row['selected_tensors']<=892,'verified shard phase');selected=row['selected_tensors'];cursor+=1
    require(selected==892,'selected native tensor coverage')
    require([x['kind'] for x in rows[cursor:]]==['verified_gpu_tensor']*len(gpu)+['weights_on_gpu','forward_complete'],'GPU/forward phase order')
    weights,forward=rows[-2:];nbytes=516*4*4096*2
    for row,lower in [(weights,plan['converted_layer_bytes']),(forward,plan['converted_layer_bytes']+2*nbytes+516*9)]:
        require(all(uint(row[k]) for k in ('cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved')),'CUDA measurement types')
        a,r,pa,pr=(row[k] for k in ('cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved'))
        require(lower<=a<=r<=pr<=64*1024**3 and a<=pa<=pr,'CUDA allocation counters/limits')
    require(finite(forward['elapsed_seconds']) and 0<forward['elapsed_seconds']<=forward['time_unix']-weights['time_unix']<=600,'forward elapsed time')
    desc=strict((out/'output.json').read_bytes());payload=gzip.decompress((out/'output.bf16.gz').read_bytes())
    require(desc['shape']==[1,516,4,4096] and desc['dtype']=='torch.bfloat16' and len(payload)==desc['bytes']==nbytes and sha(payload)==desc['sha256'],'output geometry or digest')
    require(m['input_spec']==INPUT_SPEC and desc['seed']==seed['seed'] and desc['input']=={'shape':desc['shape'],'dtype':desc['dtype'],'bytes':nbytes,'pinned':True,'sha256':sha(canonical_input(seed['seed']))},'canonical input binding')
    bits=array.array('H');bits.frombytes(payload)
    if sys.byteorder!='little':bits.byteswap()
    require(all((x & 0x7f80)!=0x7f80 for x in bits),'nonfinite output bytes')
    return {'verdict':'PASS','scope':'One-layer real weights and synthetic activation checks only','verified_shards':len(expected_shards),'verified_gpu_tensors':len(gpu),'converted_layer_bytes':plan['converted_layer_bytes'],'finite_output_elements':len(bits),'cuda_peak_allocated':forward['cuda_peak_allocated'],'forward_elapsed_seconds':forward['elapsed_seconds']}

def plan(config_path):
    import torch
    from transformers import Glm5NextConfig
    import transformers.models.glm5_next.modeling_glm5_next as native
    Glm5NextForConditionalGeneration=native.Glm5NextForConditionalGeneration
    config=Glm5NextConfig.from_dict(strict(config_path.read_bytes()))
    with torch.device('meta'):
        model=Glm5NextForConditionalGeneration._from_config(config,dtype=torch.bfloat16,attn_implementation='eager',experts_implementation='eager')
    require(all(p.device.type=='meta' for p in model.parameters()),'planning materialized weights')
    expected={}
    for name,p in model.model.language_model.layers[8].state_dict().items():
        dtype='torch.float32' if any(x in name.split('.') for x in model._keep_in_fp32_modules_strict) else 'torch.bfloat16'
        expected[name]={'shape':list(p.shape),'dtype':dtype,'bytes':p.numel()*(4 if dtype=='torch.float32' else 2)}
    print(json.dumps({'scope':'Native meta layout only; no weights loaded','stock_torch_selector':stock_selector(native),'tensors':expected,'converted_layer_bytes':sum(x['bytes'] for x in expected.values())}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--frozen',type=Path);group.add_argument('--plan',type=Path);args=parser.parse_args()
    if args.plan:plan(args.plan)
    else:run(args.frozen)
