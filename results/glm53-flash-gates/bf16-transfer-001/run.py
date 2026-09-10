"""Reuse closed capture/scoring for a partial-file HTTP transport diagnostic."""
from pathlib import Path
import gzip,hashlib,importlib.util,json,math,os,subprocess,sys,time
O=Path(sys.argv[1]).resolve();assert Path(__file__).resolve()==O/'code/run.py'
sys.path.insert(0,str(O/'code/scripts/lib'))
from glm53_contract import sha256_file,strict_json,verify_inventory
from glm53_probe_capture import capture_wrapper,inference_lock,write
from glm53_host_evidence import score_host_observations,require
CONTROL={'HOME':'/home/bmarti44','USER':'bmarti44','LOGNAME':'bmarti44','LANG':'C.UTF-8','PATH':'/usr/bin:/bin','XDG_RUNTIME_DIR':'/run/user/1000','DBUS_SESSION_BUS_ADDRESS':'unix:path=/run/user/1000/bus'}
def host():
    m=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());v=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines());return {'time_unix':time.time(),'pswpin':int(v['pswpin']),'pswpout':int(v['pswpout']),'used_swap_kib':int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0]),'available_kib':int(m['MemAvailable'].split()[0])}
def bound_read(path):
    identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    with path.open('rb') as stream:
        before=identity(os.fstat(stream.fileno()));data=stream.read();after=identity(os.fstat(stream.fileno()))
    require(before==after==identity(path.stat()),'launch document changed while reading: '+path.name)
    class Snapshot:
        def read_bytes(self):return data
    return strict_json(Snapshot()),{'sha256':hashlib.sha256(data).hexdigest(),'identity':before}

def check_bindings():
    for name,binding in bindings.items():
        _,current=bound_read(O/name)
        require(current==binding,'launch document changed: '+name)

m=None;failure=None;host_result=None;inner=None;started=False;bindings={}
with inference_lock() as lock:
    try:
        m,bindings['manifest.json']=bound_read(O/'manifest.json')
        r,bindings['randomness.json']=bound_read(O/'randomness.json')
        for row in m['files']:
            p=Path(row['path']);require(p.stat().st_size==row['size_bytes'] and sha256_file(p)==row['sha256'],'frozen input changed: '+str(p))
        require(m['safety']=={'minimum_start_gib':110,'kill_floor_gib':64,'timeout_seconds':180,'memory_high_gib':32,'memory_max_gib':34},'frozen safety mismatch')
        before=host();write(O/'preload-host.json',before);require(all(before[k]==m['broad_baseline'][k] for k in ['pswpin','pswpout','used_swap_kib']),'broad preflight host swap changed');require(before['available_kib']>=110*1024**2,'start memory')
        frozen=m['frozen_at_unix'];require(type(frozen) in (int,float) and math.isfinite(frozen) and frozen>=1595431050,'frozen timestamp')
        expected_round=int((frozen-1595431050)//30)+2;expected_publication=1595431050+(expected_round-1)*30
        require(type(r['round']) is int and r['round']==expected_round and r['publication_unix']==expected_publication and r['frozen_at_unix']==frozen and r['verification']=='DRAND_BLS_RECEIPT_OK','public beacon exact round/publication')
        cmd=[m['node'],str(O/'code/scripts/103_verify_drand_receipt_bundle.mjs'),str(r['round']),r['randomness'],r['signature'],r['previous_signature']];v=subprocess.run(cmd,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'},capture_output=True,text=True,timeout=30);require(v.returncode==0 and v.stdout=='DRAND_BLS_RECEIPT_OK\n','public beacon verification')
        require(r['seed']==int(r['randomness'][:16],16),'public seed conversion')
        write(O/'launch-beacon-verification.json',{'exit_code':v.returncode,'stdout':v.stdout,'stderr':v.stderr,'time_unix':time.time()})
        python=Path(m['python']);guard=O/'code/scripts/38_guard_glm53_probe.py';probe=O/'code/probe.py';args=['--frozen',str(O)];environment=m['environment'];command=['/usr/bin/env','-i',*(k+'='+v for k,v in sorted(environment.items())),str(python),'-I','-B',str(guard),'--output',str(O/'identity'),'--',str(probe),*args]
        control={**CONTROL,**lock,'GLM_SAFE_RUN_AS_CURRENT_USER':'1','GLM_SAFE_MIN_START_GIB':'110','GLM_SAFE_KILL_FLOOR_GIB':'64','GLM_SAFE_TIMEOUT_S':'180','GLM_SAFE_MEMORY_HIGH_GIB':'32','GLM_SAFE_DONE_DIGESTS':'1'}
        write(O/'invocation.json',{'command':['/usr/bin/bash',m['wrapper'],'--tag',m['tag'],'--',*command],'environment':{k:v for k,v in control.items() if not k.startswith('GLM_SAFE_PARENT_LOCK_')},'time_unix':time.time(),'manifest':{'sha256':bindings['manifest.json']['sha256']},'randomness':{'sha256':bindings['randomness.json']['sha256']}})
        check_bindings()
        started=True;capture_wrapper(O,Path(m['wrapper']),m['tag'],command,control,180)
        check_bindings()
        expected={'binary_sha256':sha256_file(python),'executable':str(python.resolve()),'guard':str(guard),'guard_sha256':sha256_file(guard),'probe':str(probe),'probe_sha256':sha256_file(probe),'probe_arguments':args,'environment_sha256':hashlib.sha256(b'\0'.join(sorted(os.fsencode(k+'='+v) for k,v in environment.items()))).hexdigest(),'unit_prefix':'glm52-'+m['tag']+'-','maximum_sample_gap_seconds':2.0,'minimum_start_gib':110,'kill_floor_gib':64,'timeout_seconds':180}
        host_result=score_host_observations(O,expected)
        spec=importlib.util.spec_from_file_location('frozen_layer_probe',probe);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
        inner=scorer.score(O)
        check_bindings()
        write(O/'host-score-arguments.json',expected)
    except BaseException as error:failure=repr(error)
    finally:
        after=host();write(O/'terminal-host.json',after)
        if m is not None and any(after[k]!=m['broad_baseline'][k] for k in ['pswpin','pswpout','used_swap_kib']):failure=failure or 'broad terminal host swap changed'
        try:
            check_bindings()
            for row in m['files']:
                p=Path(row['path']);require(p.stat().st_size==row['size_bytes'] and sha256_file(p)==row['sha256'],'post-run frozen input changed')
            check_bindings()
        except Exception as error:failure=failure or repr(error)
        last=host();write(O/'post-verification-host.json',last)
        if m is not None and any(last[k]!=m['broad_baseline'][k] for k in ['pswpin','pswpout','used_swap_kib']):failure=failure or 'post-verification broad host swap changed'
        write(O/'summary.json',{'verdict':'FAIL' if failure else 'PASS','scope':'Partial-file HTTP byte equivalence/transport diagnostic only; no model, GPU, full-shard or serving qualification','failure':failure,'wrapper_started':started,'host':host_result,'inner':inner,'end_unix':time.time()})
print(json.dumps(strict_json(O/'summary.json')),flush=True);raise SystemExit(1 if failure else 0)
