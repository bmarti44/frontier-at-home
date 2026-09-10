"""Serial fresh-process containment; never starts a serving model."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CONTROL={'HOME':'/home/bmarti44','USER':'bmarti44','LOGNAME':'bmarti44','LANG':'C.UTF-8','PATH':'/usr/bin:/bin','XDG_RUNTIME_DIR':'/run/user/1000','DBUS_SESSION_BUS_ADDRESS':'unix:path=/run/user/1000/bus'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--frozen',type=Path,required=True);args=ap.parse_args();out=args.frozen.resolve()
    if Path(__file__).resolve()!=out/'code/run.py':raise ValueError('execute the frozen controller copy')
    sys.path.insert(0,str(out/'code/scripts/lib'))
    from glm53_contract import sha256_file,strict_json,verify_inventory
    from glm53_probe_capture import capture_wrapper,inference_lock,write
    from glm53_host_evidence import score_host_observations,require
    spec=importlib.util.spec_from_file_location('workspace_controller_probe',out/'code/probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    def host():
        m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());v=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
        return {'time_unix':time.time(),'available_kib':int(m['MemAvailable'].split()[0]),'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'used_swap_kib':int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0])}
    def stable(path):
        identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
        before=identity(path.stat());data=path.read_bytes();require(identity(path.stat())==before,'document changed during read')
        return json.loads(data),hashlib.sha256(data).hexdigest()
    failure=None;results=[];m=None;bindings={}
    with inference_lock() as lock:
        try:
            m,bindings['manifest.json']=stable(out/'manifest.json');r,bindings['randomness.json']=stable(out/'randomness.json')
            def verify():
                for name,h in bindings.items():require(sha256_file(out/name)==h,'launch document changed')
                probe.verify_frozen(out,m)
            verify();require(m['safety']==probe.SAFETY and m['cases']==probe.CASES and m['budgets_mib']==[512,64],'frozen scope changed')
            frozen=m['frozen_at_unix'];require(type(frozen) in (int,float) and math.isfinite(frozen),'freeze time')
            round_number=int((frozen-1595431050)//30)+2
            require(type(r['round'])is int and r['round']==round_number and r['frozen_at_unix']==frozen and r['publication_unix']==1595431050+(round_number-1)*30,'fresh exact public round')
            v=subprocess.run([m['node'],str(out/'code/scripts/103_verify_drand_receipt_bundle.mjs'),str(r['round']),r['randomness'],r['signature'],r['previous_signature']],env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'},capture_output=True,text=True,timeout=30)
            require(v.returncode==0 and v.stdout=='DRAND_BLS_RECEIPT_OK\n' and r['seed']==int(r['randomness'][:16],16),'public beacon verification')
            write(out/'launch-beacon-verification.json',{'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'time_unix':time.time()})
            seed=r['seed'];schedule=probe.arm_schedule(seed);write(out/'arm-order.json',schedule)
            inventory=strict_json(Path(m['runtime']['inventory']));verify_inventory(Path(m['runtime']['root']),inventory)
            def same_host(observation):
                require(all(observation[k]==m['broad_baseline'][k] for k in ('pswpin','pswpout','used_swap_kib')),'broad baseline swap changed')
                require(observation['available_kib']>=110*1024**2,'110GiB start/recovery required')
            def execute(arm):
                index=len(results);require(arm==schedule[index],'arm order');armroot=out/'arms'/str(index);armroot.mkdir(exist_ok=False)
                state=armroot/'state';shutil.copytree(out/'cache-template',state,symlinks=False)
                before=host();write(armroot/'before-host.json',before);same_host(before);verify()
                env={**m['environment'],'HOME':str(state),'TRITON_CACHE_DIR':str(state/'triton'),'DG_JIT_CACHE_DIR':str(state/'deep-gemm'),'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB':str(arm['budget_mib'])}
                python=Path(m['python']);guard=out/'code/scripts/38_guard_glm53_probe.py';source=out/'code/probe.py';pargs=['--frozen',str(out),'--arm-index',str(index)]
                command=['/usr/bin/env','-i',*(k+'='+v for k,v in sorted(env.items())),str(python),'-I','-B',str(guard),'--output',str(armroot/'identity'),'--',str(source),*pargs]
                tag='glm53-idxws-'+out.name[-10:]+'-'+str(index)
                control={**CONTROL,**lock,'GLM_SAFE_RUN_AS_CURRENT_USER':'1','GLM_SAFE_MIN_START_GIB':'110','GLM_SAFE_KILL_FLOOR_GIB':'40','GLM_SAFE_TIMEOUT_S':'600','GLM_SAFE_MEMORY_HIGH_GIB':'32','GLM_SAFE_DONE_DIGESTS':'1'}
                write(armroot/'invocation.json',{'arm':arm,'command':command,'environment':{k:v for k,v in control.items() if not k.startswith('GLM_SAFE_PARENT_LOCK_')},'manifest_sha256':bindings['manifest.json'],'randomness_sha256':bindings['randomness.json'],'time_unix':time.time()})
                capture_wrapper(armroot,Path(m['wrapper']),tag,command,control,600)
                after=host();write(armroot/'after-host.json',after);same_host(after);verify()
                expected={'binary_sha256':sha256_file(python),'executable':str(python.resolve()),'guard':str(guard),'guard_sha256':sha256_file(guard),'probe':str(source),'probe_sha256':sha256_file(source),'probe_arguments':pargs,'environment_sha256':hashlib.sha256(b'\0'.join(sorted(os.fsencode(k+'='+v) for k,v in env.items()))).hexdigest(),'unit_prefix':'glm52-'+tag+'-','maximum_sample_gap_seconds':2.0,'minimum_start_gib':110,'kill_floor_gib':40,'timeout_seconds':600}
                write(armroot/'host-score-arguments.json',expected);host_result=score_host_observations(armroot,expected)
                checks=armroot/'checks';inner_manifest=strict_json(checks/'manifest.json');require(inner_manifest=={'seed':seed,**arm,'source_sha256':sha256_file(source),'frozen_manifest_sha256':bindings['manifest.json'],'randomness_sha256':bindings['randomness.json']},'inner manifest binding')
                inner=probe.score_arm(checks,seed,arm['case'],arm['budget_mib']);require(strict_json(checks/'summary.json')==inner,'independent inner reduction mismatch')
                cache=[]
                for p in sorted(state.rglob('*')):
                    require(not p.is_symlink(),'new state symlink')
                    if p.is_file():cache.append({'path':str(p.relative_to(state)),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
                write(armroot/'generated-cache-inventory.json',{'scope':'post-run input artifacts for a future frozen replay; not qualification of newly generated binaries','files':cache})
                result={'arm':arm,'root':str(armroot),'host':host_result,'inner':inner};results.append(result);write(armroot/'combined.json',result);return result
            def compare(left,right):
                probe.compare_arms(Path(left['root'])/'checks',Path(right['root'])/'checks',seed,left['arm']['case'])
                write(Path(right['root'])/'paired-comparison.json',{'verdict':'PASS','left_arm':left['arm'],'right_arm':right['arm'],'ordered_indices_and_cache_tail_valid_logit_bytes_equal':True,'scope':'synthetic byte equality only; no model qualification'})
            probe.execute_schedule(seed,execute,compare)
            four=[x for x in results if x['arm']['case']=='prefill-4']
            baseline_peak=min(x['inner']['memory']['cuda_peak_allocated'] for x in four if x['arm']['budget_mib']==512)
            candidate_peak=next(x['inner']['memory']['cuda_peak_allocated'] for x in four if x['arm']['budget_mib']==64)
            require(baseline_peak-candidate_peak>=256*1024**2,'four-request workspace reduction below256MiB')
            write(out/'workspace-delta.json',{'baseline_minimum_cuda_peak_allocated':baseline_peak,'candidate_cuda_peak_allocated':candidate_peak,'delta_bytes':baseline_peak-candidate_peak,'scope':'synthetic observed allocation only; no host saving or model-fit claim'})
        except BaseException as error:failure=repr(error)
        finally:
            try:
                if m is not None:
                    verify()
                    verify_inventory(Path(m['runtime']['root']),strict_json(Path(m['runtime']['inventory'])))
                    last=host();write(out/'terminal-host.json',last)
                    require(all(last[k]==m['broad_baseline'][k] for k in ('pswpin','pswpout','used_swap_kib')),'terminal broad swap changed')
            except BaseException as error:failure=failure or repr(error)
            write(out/'summary.json',{'verdict':'FAIL' if failure else 'NO_RESULT','checks':'FAIL' if failure else 'PASS','scope':'model-free512/64MiB synthetic workspace preparation; no serving, quality, context or performance qualification','failure':failure,'completed_arms':len(results),'arms':results,'kernel_binary_qualification':'NO_RESULT: generated cache inventory is post-run only; prewarm/freeze/sealed replay required before any qualified binary claim','large_model_admission':False,'time_unix':time.time()})
    print(json.dumps(strict_json(out/'summary.json')),flush=True);return 1 if failure else 0

if __name__=='__main__':raise SystemExit(main())
