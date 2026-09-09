#!/usr/bin/env python3
"""Start an optional, authenticated local GLM server under existing containment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import secrets
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/bmarti44/.cache/glm53-flash')
RUNTIME=BASE/'native-runtime-002/runtime'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def reuse_prepared_kernels(state):
    """Copy existing warm caches; keep preparation evidence untouched."""
    sources=[]
    for attempt in ('mla-preflight-001','kda-preflight-002','conv-preflight-001','indexer-preflight-003','sampling-preparation-003'):
        source=BASE/attempt/'state'
        for subtree in ('triton','.cache/flashinfer','.cache/vllm/modelinfos','deep-gemm'):
            if (source/subtree).exists():
                shutil.copytree(source/subtree,state/subtree,dirs_exist_ok=True)
        sources.append(source)
    for group in (state/'triton').rglob('__grp__*.json'):
        data=json.loads(group.read_text())
        children={}
        for name,value in data['child_paths'].items():
            old=Path(value)
            origin=next((root for root in sources if old.is_relative_to(root)),None)
            if origin is None:raise ValueError('unexpected prepared kernel child path')
            relocated=state/old.relative_to(origin)
            if not relocated.is_file() or sha(old)!=sha(relocated):raise ValueError('prepared kernel copy mismatch')
            children[name]=str(relocated)
        group.write_text(json.dumps({'child_paths':children})+'\n')


def serve(directory):
    launch=json.loads((directory/'launch.json').read_text())
    os.environ['VLLM_API_KEY']=(directory/'api-key').read_text().strip()
    sys.path.insert(0,str(directory/'lib'))
    sys.argv=['vllm',*launch['arguments']]
    runpy.run_module('vllm.entrypoints.openai.api_server',run_name='__main__')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',action='store_true')
    parser.add_argument('--model',type=Path,default=BASE/'model-weights-001')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8015)
    parser.add_argument('--text-only',action='store_true',help='First text smoke; keeps the full four-slot text geometry')
    parser.add_argument('--skip-mm-profiling',action='store_true',help='Skip automatic media warm-up while retaining media support')
    parser.add_argument('--prefill-batch',type=int,choices=(128,256,512,1024,2048),default=2048,help='Prompt chunk size; does not change context or slot count')
    parser.add_argument('--standard-cuda-allocator',action='store_true',help='Avoid expandable virtual reservations under the existing address-space limit')
    parser.add_argument('--release-warmup-cache',action='store_true',help='Release unused startup allocations before reserving the full KV cache')
    parser.add_argument('--prepared-flashinfer',action='store_true',help='Use existing compiled FlashInfer libraries through its native cache provider')
    args=parser.parse_args()
    if not args.start:parser.error('explicit --start is required; this never changes the serving default')
    if args.port not in range(1024,65536) or args.port in (8010,8013,8014):parser.error('use a separate local development port')
    model=args.model.resolve(); output=args.output.resolve()
    inventory=json.loads((model/'inventory.json').read_text())
    for row in inventory['files']:
        path=model/row['path']
        if path.parent!=model or path.is_symlink() or path.stat().st_size!=row['size_bytes'] or sha(path)!=row['sha256']:
            raise ValueError('model inventory mismatch')
    output.mkdir(parents=True,exist_ok=False); (output/'lib').mkdir(); (output/'state').mkdir()
    shutil.copyfile(ROOT/'scripts/lib/glm53_runtime_policy.py',output/'lib/glm53_runtime_policy.py')
    if args.prepared_flashinfer:
        shutil.copyfile(ROOT/'scripts/lib/glm53_flashinfer_cache.py',output/'lib/flashinfer_jit_cache.py')
    if args.release_warmup_cache:
        shutil.copyfile(ROOT/'scripts/lib/glm53_worker.py',output/'lib/glm53_worker.py')
    shutil.copyfile(ROOT/'scripts/38_guard_glm53_probe.py',output/'guard.py')
    shutil.copyfile(__file__,output/'server.py')
    reuse_prepared_kernels(output/'state')
    descriptor=os.open(output/'api-key',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(descriptor,'w') as key:key.write(secrets.token_urlsafe(32)+'\n')
    profile=json.loads((ROOT/'configs/profiles/glm-5.3-flash/cuda-spark-128g-1m.json').read_text())
    arguments=[value.replace('{model}',str(model)).replace('{port}',str(args.port)) for value in profile['launch']['args'][4:]]
    arguments[arguments.index('--max-num-batched-tokens')+1]=str(args.prefill_batch)
    arguments+=['--load-format','instanttensor','--dtype','bfloat16','--enforce-eager',
                '--enable-chunked-prefill','--kv-cache-memory-bytes','9565304320']
    if args.release_warmup_cache:arguments+=['--worker-cls','glm53_worker.WarmupCleanupWorker']
    if args.text_only:arguments+=['--language-model-only']
    if args.skip_mm_profiling:arguments+=['--skip-mm-profiling']
    state=output/'state'
    environment={**profile['launch']['env'],'HOME':str(state),'LANG':'C.UTF-8',
        'PATH':f'{RUNTIME}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin','CUDA_HOME':'/usr/local/cuda-13.0',
        'CUDA_VISIBLE_DEVICES':'0','VLLM_PLUGINS':'vllm_exl3','GLM53_EXL3_BF16_SHARD_FIX':'1',
        'MAX_JOBS':'2','NVCC_THREADS':'1','TOKENIZERS_PARALLELISM':'false',
        'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','CUDA_CACHE_DISABLE':'1','FLASHINFER_DISABLE_JIT':'1',
        'TRITON_CACHE_AUTOTUNING':'1',
        'TRITON_CACHE_DIR':str(state/'triton'),'DG_JIT_CACHE_DIR':str(state/'deep-gemm'),
        'DG_JIT_USE_NVRTC':'0','DG_JIT_NVCC_COMPILER':'/usr/local/cuda-13.0/bin/nvcc','DG_JIT_CPP_STANDARD':'20',
        'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB':'512',
        'INSTANTTENSOR_BUFFER_SIZE':'1342177280','INSTANTTENSOR_CHUNK_SIZE':'8388608',
        'INSTANTTENSOR_CONCURRENCY':'1','INSTANTTENSOR_IO_DEPTH':'3','INSTANTTENSOR_BACKEND':'AIO'}
    if args.standard_cuda_allocator:environment['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:False'
    launch={'scope':'development bring-up; no model qualification or performance claim',
        'start_unix':time.time(),'arguments':arguments,'environment':environment,
        'library_sources':[{'path':p.name,'sha256':sha(p)} for p in sorted((output/'lib').glob('*.py'))],
        'worker_source':{'sha256':sha(output/'lib/glm53_worker.py')} if args.release_warmup_cache else None,
        'model_inventory':{'sha256':sha(model/'inventory.json')},
        'python':{'sha256':sha(RUNTIME/'bin/python3')},
        'source':{'sha256':sha(output/'server.py')},'profile':profile['profile_id'],
        'safety':{'minimum_start_gib':110,'kill_floor_gib':18,'memory_high_gib':92,'memory_max_gib':94,'timeout_seconds':9000}}
    (output/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
    control={'HOME':'/home/bmarti44','USER':'bmarti44','LOGNAME':'bmarti44','LANG':'C.UTF-8','PATH':'/usr/bin:/bin',
        'XDG_RUNTIME_DIR':'/run/user/1000','DBUS_SESSION_BUS_ADDRESS':'unix:path=/run/user/1000/bus',
        'GLM_SAFE_RUN_AS_CURRENT_USER':'1','GLM_SAFE_MIN_START_GIB':'110','GLM_SAFE_KILL_FLOOR_GIB':'18',
        'GLM_SAFE_MEMORY_HIGH_GIB':'92','GLM_SAFE_TIMEOUT_S':'9000','GLM_SAFE_DONE_DIGESTS':'1'}
    command=['/usr/bin/bash',str(ROOT/'results/glm52-gates/harness/glm_cgroup_run.sh'),
        '--tag','glm53-dev-'+str(os.getpid()),'--','/usr/bin/env','-i',
        *(k+'='+v for k,v in sorted(environment.items())),str(RUNTIME/'bin/python3'),'-I','-B',
        str(output/'guard.py'),'--output',str(output/'identity'),'--',str(output/'server.py'),'--serve',str(output)]
    print(json.dumps({'event':'launch','port':args.port,'output':str(output),'context_per_slot':262144,'slots':4}),flush=True)
    with (output/'wrapper.log').open('w') as log:
        result=subprocess.run(command,env=control,stdout=log,stderr=subprocess.STDOUT)
    (output/'exit.json').write_text(json.dumps({'exit_code':result.returncode,'time_unix':time.time()})+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':
    if sys.argv[1:2]==['--serve']:serve(Path(sys.argv[2]))
    else:main()
