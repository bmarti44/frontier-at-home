"""Freeze a single real-weight feasibility probe; never load native weights here."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-one-layer-001';S=Path(__file__).parent
sys.path.insert(0,str(R/'scripts/lib'))
from glm53_contract import sha256_file,strict_json,verify_inventory
from glm53_probe_capture import inference_lock

def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
with inference_lock():
    if subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True):raise ValueError('clean source required')
    if shutil.disk_usage(B).free<1024**3:raise ValueError('at least1GiB disk required')
    O.mkdir(exist_ok=False);(O/'code').mkdir();(O/'metadata').mkdir();(O/'state').mkdir()
    m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());v=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
    baseline={'time_unix':time.time(),'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'used_swap_kib':int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0]),'available_kib':int(m['MemAvailable'].split()[0])};save(O/'broad-baseline.json',baseline)
    if baseline['available_kib']<110*1024**2:raise ValueError('start memory below110GiB')
    facts={}
    for unit in ['docker.service','docker.socket','containerd.service']:
        text=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','-p','MainPID','-p','InvocationID'],text=True);facts[unit]=dict(x.split('=',1) for x in text.splitlines());assert facts[unit]['ActiveState']=='inactive' and facts[unit].get('MainPID','0')=='0'
    status=json.loads(subprocess.check_output([str(R/'scripts/93_profile_serve.sh'),'--profile','glm-5.3-flash/cuda-spark-128g-1m-experimental','status'],text=True));assert status['state']=='stopped'
    listeners=subprocess.check_output(['ss','-ltnp'],text=True);assert ':8015' not in listeners
    save(O/'host-before.json',{'time_unix':time.time(),'services':facts,'profile':status,'listeners':listeners,'meminfo':Path('/proc/meminfo').read_text()})
    runtime=B/'native-runtime-002/runtime';inventory=B/'native-runtime-002/inventory.json';verified=verify_inventory(runtime,strict_json(inventory))
    for name in ['probe.py','freeze.py','run.py','test_probe.py','PROTOCOL.md']:shutil.copyfile(S/name,O/'code'/name)
    for name in ['scripts/38_guard_glm53_probe.py','scripts/103_verify_drand_receipt_bundle.mjs','scripts/lib/glm53_contract.py','scripts/lib/glm53_host_evidence.py','scripts/lib/glm53_probe_capture.py']:
        p=O/'code'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(R/name,p)
    shutil.copyfile(B/'context-freeze-003/fetch-randomness.py',O/'fetch-randomness.py')
    metadata=B/'bf16-local-feasibility-001';api=strict_json(metadata/'streaming-layout/model-api-blobs.json');revision='a5b45eb41df6402735dedc900be14a42e8d5e538';assert api['sha']==revision
    for name in ['config.json','model.safetensors.index.json']:shutil.copyfile(metadata/name,O/'metadata'/name)
    for name in ['model-api-blobs.json','summary.json']:shutil.copyfile(metadata/'streaming-layout'/name,O/'metadata'/name)
    index=strict_json(O/'metadata/model.safetensors.index.json')['weight_map'];names=sorted({v for k,v in index.items() if k.startswith('model.language_model.layers.8.')});assert names==[f'model-{i:05d}-of-00120.safetensors' for i in [115,116,117]]
    shards=[next(x for x in api['siblings'] if x['rfilename']==name) for name in names]
    for name in names:shutil.copyfile(metadata/'streaming-layout'/(name+'.header.json'),O/'metadata'/(name+'.header.json'))
    planned=subprocess.run([str(runtime/'bin/python3'),'-I','-B',str(O/'code/probe.py'),'--plan',str(O/'metadata/config.json')],env={'HOME':str(O/'state'),'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','CUDA_VISIBLE_DEVICES':'','USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},capture_output=True,text=True,timeout=120)
    (O/'metadata/layer-plan.stderr').write_text(planned.stderr);assert planned.returncode==0,planned.stderr
    layout=json.loads(planned.stdout);assert len(layout['tensors'])==28;save(O/'metadata/layer-plan.json',layout)
    corpus=R/'results/glm53-flash-gates/kimi-serialization-001';assert strict_json(corpus/'summary.json')['input_tokens_max']==516
    for name in ['cases.jsonl.gz','summary.json','manifest.json']:shutil.copyfile(corpus/name,O/'metadata'/('corpus-'+name))
    node=Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node');python=runtime/'bin/python3';wrapper=R/'results/glm52-gates/harness/glm_cgroup_run.sh'
    environment={'HOME':str(O/'state'),'PATH':f'{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin','LANG':'C.UTF-8','CUDA_HOME':'/usr/local/cuda-13.0','CUDA_VISIBLE_DEVICES':'0','CUDA_CACHE_DISABLE':'1','USE_HUB_KERNELS':'0','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DEACTIVATE_ASYNC_LOAD':'1','TOKENIZERS_PARALLELISM':'false','NVIDIA_TF32_OVERRIDE':'0','TORCH_COMPILE_DISABLE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:False'}
    paths=[p for root in [O/'code',O/'metadata'] for p in sorted(root.rglob('*')) if p.is_file()]+[O/'fetch-randomness.py',inventory,python,node,wrapper,R/'results/glm52-gates/harness/glm_safe_run.sh',R/'scripts/03_memory_guard.py',Path('/usr/lib/aarch64-linux-gnu/libc.so.6')]
    manifest={'scope':'One real BF16 KDA layer with synthetic HCactivations; no native full-model reference or serving qualification','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'broad_baseline':baseline,'runtime':{'root':str(runtime),'inventory':str(inventory),'verified_files':len(verified)},'python':str(python),'node':str(node),'wrapper':str(wrapper),'tag':'glm53-bf16-layer-001','model_revision':revision,'shards':shards,'input_tokens':516,'environment':environment,'safety':{'minimum_start_gib':110,'kill_floor_gib':40,'timeout_seconds':600,'memory_high_gib':62,'memory_max_gib':64},'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in paths],'frozen_at_unix':time.time()}
    save(O/'manifest.json',manifest);print(json.dumps({'frozen':str(O),'runtime_files':len(verified),'frozen_at_unix':manifest['frozen_at_unix']}),flush=True)
