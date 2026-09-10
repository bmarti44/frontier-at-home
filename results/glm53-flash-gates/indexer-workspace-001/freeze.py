"""Unloaded, lock-held preparation only; do not execute during another GPU run."""
import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path('/home/bmarti44/spark-deepseek-v4-flash');BASE=Path('/home/bmarti44/.cache/glm53-flash');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO/'scripts/lib'))
from glm53_contract import sha256_file,strict_json,verify_inventory
from glm53_probe_capture import inference_lock

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def baseline():
    m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());v=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
    return {'time_unix':time.time(),'available_kib':int(m['MemAvailable'].split()[0]),'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'used_swap_kib':int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0])}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--metadata',type=Path,required=True);ap.add_argument('--prepared-state',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();metadata=a.metadata.resolve();state=a.prepared_state.resolve()
    assert out.parent==BASE and not out.exists() and metadata.is_dir() and state.is_dir()
    with inference_lock():
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO),'commit/review clean candidate first'
        assert shutil.disk_usage(BASE).free>=2*1024**3,'2GiB disk admission'
        before=baseline();assert before['available_kib']>=110*1024**2,'unloaded110GiB required'
        assert len(Path('/proc/swaps').read_text().splitlines())==1,'current owner swap pause required'
        status=json.loads(subprocess.check_output([str(REPO/'scripts/93_profile_serve.sh'),'--profile','glm-5.3-flash/cuda-spark-128g-1m-experimental','status'],text=True));assert status['state']=='stopped'
        out.mkdir();(out/'code').mkdir();(out/'metadata').mkdir();(out/'arms').mkdir();save(out/'broad-baseline.json',before)
        runtime=BASE/'native-runtime-002/runtime';profile=REPO/'configs/profiles/glm-5.3-flash/cuda-spark-128g-agent-fast.json';build=REPO/'configs/build-manifests/glm53-flash-local.json'
        role=strict_json(profile)['artifact_roles']['runtime_inventory'];binding=strict_json(build)['inventories'][role];inventory=Path(binding['path'].replace('{repo}',str(REPO)));assert sha256_file(inventory)==binding['sha256']
        data=json.loads(gzip.decompress(inventory.read_bytes()));normal=out/'metadata/runtime-inventory.json';save(normal,{'schema_version':1,'files':[{k:r[k] for k in ('path','size_bytes','sha256')} for r in data['files']]});verified=verify_inventory(runtime,strict_json(normal))
        for name in ['probe.py','native.py','run.py','freeze.py','test_probe.py','PROTOCOL.md']:shutil.copyfile(HERE/name,out/'code'/name)
        shutil.copyfile(REPO/'scripts/lib/glm53_indexer_fixture.py',out/'code/prior_fixture.py')
        shutil.copyfile(REPO/'scripts/45_probe_glm53_indexer.py',out/'code/prior_probe45.py')
        for name in ['scripts/38_guard_glm53_probe.py','scripts/103_verify_drand_receipt_bundle.mjs','scripts/lib/glm53_contract.py','scripts/lib/glm53_probe_capture.py','scripts/lib/glm53_host_evidence.py']:
            dest=out/'code'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
        # Metadata-only model directory: weights are expressly not copied/opened.
        for name in ['config.json','generation_config.json','preprocessor_config.json','processor_config.json','video_preprocessor_config.json']:
            if (metadata/name).is_file():shutil.copyfile(metadata/name,out/'metadata'/name)
        assert (out/'metadata/config.json').is_file()
        # Every arm begins from an identical prepared cache snapshot. Existing
        # absolute paths in cache metadata are retained; this stage is preparatory.
        template=out/'cache-template';shutil.copytree(state,template,symlinks=True)
        cache_files=[]
        for p in sorted(template.rglob('*')):
            assert not p.is_symlink(),'cache snapshot must have closed regular files'
            if p.is_file():cache_files.append({'path':str(p.relative_to(template)),'sha256':sha256_file(p),'size_bytes':p.stat().st_size})
        save(out/'cache-template-inventory.json',{'files':cache_files})
        source_files=[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in sorted(state.rglob('*')) if p.is_file()]
        assert len(source_files)==len(cache_files) and all(a['sha256']==b['sha256'] and a['size_bytes']==b['size_bytes'] for a,b in zip(source_files,cache_files)),'source cache changed during copy'
        fetch=BASE/'context-freeze-003/fetch-randomness.py';shutil.copyfile(fetch,out/'fetch-randomness.py')
        node=Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node');python=runtime/'bin/python3';wrapper=REPO/'results/glm52-gates/harness/glm_cgroup_run.sh'
        spec=importlib.util.spec_from_file_location('workspace_freeze_probe',HERE/'probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        files=[p for d in [out/'code',out/'metadata',template] for p in sorted(d.rglob('*')) if p.is_file()]+[out/'cache-template-inventory.json',out/'fetch-randomness.py',inventory,profile,build,python,node,wrapper,REPO/'results/glm52-gates/harness/glm_safe_run.sh',REPO/'scripts/03_memory_guard.py',Path('/usr/lib/aarch64-linux-gnu/libc.so.6'),Path('/usr/local/cuda-13.0/bin/nvcc'),Path('/usr/bin/c++')]
        env={'PATH':f'{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin','LANG':'C.UTF-8','CUDA_HOME':'/usr/local/cuda-13.0','CUDA_VISIBLE_DEVICES':'0','CUDA_CACHE_DISABLE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_PLUGINS':'vllm_exl3','GLM53_EXL3_BF16_SHARD_FIX':'1','TOKENIZERS_PARALLELISM':'false','NVIDIA_TF32_OVERRIDE':'0','TORCH_COMPILE_DISABLE':'1','OMP_NUM_THREADS':'2','MAX_JOBS':'2','NVCC_THREADS':'1','DG_JIT_USE_NVRTC':'0','DG_JIT_CPP_STANDARD':'20','DG_JIT_NVCC_COMPILER':'/usr/local/cuda-13.0/bin/nvcc','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:False'}
        save(out/'manifest.json',{'scope':'Model-free paired indexer workspace preparation only; successful combined verdict NO_RESULT until independently frozen binary replay','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'safety':probe.SAFETY,'cases':probe.CASES,'budgets_mib':[512,64],'broad_baseline':before,'python':str(python),'node':str(node),'wrapper':str(wrapper),'runtime':{'root':str(runtime),'inventory':str(normal),'verified_files':len(verified)},'environment':env,'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in files],'frozen_at_unix':time.time()})
        print(out,flush=True)

if __name__=='__main__':main()
