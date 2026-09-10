"""Explicit real-weight/synthetic-activation feasibility probe; no serving imports."""
from pathlib import Path
import argparse,gc,gzip,hashlib,inspect,json,math,struct,time,urllib.request

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
        selectors=[]
        for name in ['causal_conv1d_fn','causal_conv1d_update','chunk_kimi_delta_attention','recurrent_kimi_delta_attention']:
            fn=inspect.unwrap(getattr(native,name));require(fn.__module__==native.__name__ and fn.__globals__ is native.__dict__,'native fallback identity')
            require(Path(fn.__code__.co_filename).resolve()==Path(native.__file__).resolve(),'native fallback file');setattr(native,name,fn)
            selectors.append({'name':name,'source':{'sha256':sha(inspect.getsource(fn).encode())}})
        save('stock-torch-selector.json',{'module':{'path':native.__file__,'sha256':sha(Path(native.__file__).read_bytes())},'functions':selectors})
        config=Glm5NextConfig.from_dict(strict((root/'metadata/config.json').read_bytes()))
        with torch.device('meta'):
            model=native.Glm5NextForConditionalGeneration._from_config(config,dtype=torch.bfloat16,attn_implementation='eager',experts_implementation='eager').eval()
        require(all(p.device.type=='meta' for p in model.parameters()),'meta initialization allocated weights')
        prefix='model.language_model.layers.8.';all_names=set(model.state_dict());want={k for k in all_names if k.startswith(prefix)}
        expected_native={k for k in strict((root/'metadata/model.safetensors.index.json').read_bytes())['weight_map'] if k.startswith(prefix)}
        loaded={};downloaded=[]
        for record in manifest['shards']:
            check_host();name=record['rfilename'];buffer=bytearray(record['size']);position=0;began=time.time();event(kind='download_start',shard=name,bytes=record['size'])
            url='https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/'+manifest['model_revision']+'/'+name+'?download=true'
            request=urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'glm53-bounded-reference-feasibility'})
            with urllib.request.urlopen(request,timeout=30) as response:
                require(response.status==200 and int(response.headers.get('Content-Length','-1'))==len(buffer),'full shard HTTP framing')
                while position<len(buffer):
                    count=response.readinto(memoryview(buffer)[position:min(position+8*1024**2,len(buffer))]);require(count is not None and count>0,'short shard read');position+=count
                    if position%(256*1024**2)<8*1024**2:event(kind='download_progress',shard=name,bytes=position,host=check_host())
                require(response.read(1)==b'','oversized shard response')
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
        event(kind='weights_on_gpu',host=check_host(),cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved())
        n=manifest['input_tokens'];require(n==516,'fixed input shape changed')
        generator=torch.Generator(device='cpu').manual_seed(seed);source_input=(torch.randn((1,n,4,4096),generator=generator,dtype=torch.float32)*0.02).to(torch.bfloat16).pin_memory()
        require(source_input.is_pinned(),'pageable activation staging');x=source_input.to('cuda',non_blocking=True);torch.cuda.synchronize()
        mask=torch.ones((1,n),dtype=torch.bool,device='cuda');positions=torch.arange(n,device='cuda').unsqueeze(0)
        torch.cuda.reset_peak_memory_stats();began=time.time()
        with torch.inference_mode():y,topk=layer(x,attention_mask=mask,position_ids=positions,position_embeddings=None,input_ids=None,past_key_values=None,prev_topk_indices=None)
        torch.cuda.synchronize();require(y.shape==x.shape and y.dtype==torch.bfloat16 and bool(torch.isfinite(y).all()),'invalid complete layer output');require(topk is None,'unexpected KDA topk')
        event(kind='forward_complete',elapsed_seconds=time.time()-began,host=check_host(),cuda_allocated=torch.cuda.memory_allocated(),cuda_peak_allocated=torch.cuda.max_memory_allocated())
        output=y.cpu().contiguous().view(torch.uint8).numpy().tobytes();(out/'output.bf16.gz').write_bytes(gzip.compress(output,mtime=0))
        save('output.json',{'shape':list(y.shape),'dtype':str(y.dtype),'bytes':len(output),'sha256':sha(output),'seed':seed,'input':{'shape':list(x.shape),'dtype':str(x.dtype),'sha256':sha(source_input.view(torch.uint8).numpy().tobytes())}})
        save('summary.json',{'verdict':'PASS','scope':'One native BF16 KDA layer with synthetic activations; host verdict required separately','native_model_reference':False,'full_model_loaded':False,'layer':8,'input_tokens_shape':n,'converted_layer_bytes':sum(x['bytes'] for x in records),'finite_output':True,'pinned_weight_and_activation_staging':True,'unrelated_model_weights_remained_meta':True})
    except BaseException as error:
        event(kind='failure',error=repr(error));save('summary.json',{'verdict':'FAIL','scope':'One-layer feasibility only','error':repr(error),'native_model_reference':False});raise

def score(root):
    import array,sys
    m=strict((root/'manifest.json').read_bytes());out=root/'checks';seed=strict((root/'randomness.json').read_bytes())
    require(seed['seed']==int(seed['randomness'][:16],16),'seed conversion mismatch')
    rows=[strict(x) for x in (out/'raw.jsonl').read_bytes().splitlines()]
    require(rows and not any(x['kind']=='failure' for x in rows),'missing raw or inner failure')
    times=[x['time_unix'] for x in rows];require(all(type(t) in (int,float) and math.isfinite(t) for t in times) and all(b>a for a,b in zip(times,times[1:])),'raw timestamps')
    require(all(x['kind'] in ['download_start','download_progress','verified_shard','verified_gpu_tensor','weights_on_gpu','forward_complete'] for x in rows),'unknown raw event')
    stage=strict((out/'staging.json').read_bytes());plan=strict((root/'metadata/layer-plan.json').read_bytes())
    observed={x['name']:x for x in stage['weights']};require(len(observed)==len(stage['weights'])==len(plan['tensors']),'staging duplicates or coverage')
    require(set(observed)==set(plan['tensors']),'staging parameter names')
    for name,spec in plan['tensors'].items():
        row=observed[name];require(all(row[k]==v for k,v in spec.items()) and row['pinned'] is True,'staging shape/dtype/pinning')
    require(stage['persistent_until_forward_completion'] is True and stage['converted_layer_bytes']==plan['converted_layer_bytes'] and stage['native_source_tensors']==892,'native staging totals')
    expected_shards=[{'path':x['rfilename'],'size_bytes':x['size'],'sha256':x['lfs']['sha256']} for x in m['shards']]
    require(stage['downloaded_shards']==expected_shards and [x['shard'] for x in rows if x['kind']=='verified_shard']==expected_shards,'verified shard coverage')
    require([x['shard'] for x in rows if x['kind']=='download_start']==[x['path'] for x in expected_shards],'download admission coverage')
    gpu=[x for x in rows if x['kind']=='verified_gpu_tensor'];require(len(gpu)==len(observed) and {x['name'] for x in gpu}==set(observed),'GPU parameter coverage')
    for row in gpu:require(row['sha256']==observed[row['name']]['sha256'],'GPU weight digest mismatch')
    for kind in ['weights_on_gpu','forward_complete']:require(sum(x['kind']==kind for x in rows)==1,'missing or duplicate phase')
    desc=strict((out/'output.json').read_bytes());payload=gzip.decompress((out/'output.bf16.gz').read_bytes())
    require(desc['shape']==[1,516,4,4096] and desc['dtype']=='torch.bfloat16' and len(payload)==desc['bytes']==516*4*4096*2 and sha(payload)==desc['sha256'],'output geometry or digest')
    require(desc['seed']==seed['seed'] and desc['input']['shape']==desc['shape'] and desc['input']['dtype']==desc['dtype'],'input binding')
    bits=array.array('H');bits.frombytes(payload)
    if sys.byteorder!='little':bits.byteswap()
    require(all((x & 0x7f80)!=0x7f80 for x in bits),'nonfinite output bytes')
    return {'verdict':'PASS','scope':'One-layer real weights and synthetic activation checks only','verified_shards':len(expected_shards),'verified_gpu_tensors':len(gpu),'converted_layer_bytes':plan['converted_layer_bytes'],'finite_output_elements':len(bits)}

def plan(config_path):
    import torch
    from transformers import Glm5NextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
    config=Glm5NextConfig.from_dict(strict(config_path.read_bytes()))
    with torch.device('meta'):
        model=Glm5NextForConditionalGeneration._from_config(config,dtype=torch.bfloat16,attn_implementation='eager',experts_implementation='eager')
    require(all(p.device.type=='meta' for p in model.parameters()),'planning materialized weights')
    expected={}
    for name,p in model.model.language_model.layers[8].state_dict().items():
        dtype='torch.float32' if any(x in name.split('.') for x in model._keep_in_fp32_modules_strict) else 'torch.bfloat16'
        expected[name]={'shape':list(p.shape),'dtype':dtype,'bytes':p.numel()*(4 if dtype=='torch.float32' else 2)}
    print(json.dumps({'scope':'Native meta layout only; no weights loaded','tensors':expected,'converted_layer_bytes':sum(x['bytes'] for x in expected.values())}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--frozen',type=Path);group.add_argument('--plan',type=Path);args=parser.parse_args()
    if args.plan:plan(args.plan)
    else:run(args.frozen)
