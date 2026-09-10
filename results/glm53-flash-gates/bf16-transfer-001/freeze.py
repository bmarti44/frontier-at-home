"""Prepare a fresh, limited HTTP transport diagnostic; no network payload here."""
from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys,time
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-transfer-001';S=Path(__file__).parent
sys.path.insert(0,str(R/'scripts/lib'))
from glm53_probe_capture import inference_lock
from glm53_contract import sha256_file
spec=importlib.util.spec_from_file_location('transport',S/'probe.py');probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
with inference_lock():
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=R,text=True),'clean source required'
    baseline=probe.host();assert baseline['available_kib']>=110*1024**2
    O.mkdir(exist_ok=False);(O/'code/scripts').mkdir(parents=True);(O/'state').mkdir()
    for name in ['probe.py','test_probe.py','test_controller.py','freeze.py','run.py','PROTOCOL.md']:shutil.copyfile(S/name,O/'code'/name)
    for name in ['103_verify_drand_receipt_bundle.mjs','38_guard_glm53_probe.py']:shutil.copyfile(R/'scripts'/name,O/'code/scripts'/name)
    shutil.copyfile(B/'context-freeze-003/fetch-randomness.py',O/'fetch-randomness.py')
    lib=O/'code/scripts/lib';lib.mkdir();shutil.copyfile(R/'scripts/lib/glm53_probe_capture.py',lib/'glm53_probe_capture.py');shutil.copyfile(R/'scripts/lib/glm53_host_evidence.py',lib/'glm53_host_evidence.py');shutil.copyfile(R/'scripts/lib/glm53_contract.py',lib/'glm53_contract.py')
    shutil.copyfile(B/'bf16-local-feasibility-001/streaming-layout/model-api-blobs.json',O/'model-api-blobs.json')
    api=json.loads((O/'model-api-blobs.json').read_text());assert api['sha']==probe.REVISION;assert next(x for x in api['siblings'] if x['rfilename']==probe.SHARD)['size']==probe.TOTAL
    wrapper=R/'results/glm52-gates/harness/glm_cgroup_run.sh'
    python=Path('/usr/bin/python3').resolve();node=Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node');paths=[p for p in sorted((O/'code').rglob('*')) if p.is_file()]+[O/'fetch-randomness.py',O/'model-api-blobs.json',python,node,wrapper,R/'results/glm52-gates/harness/glm_safe_run.sh',R/'scripts/03_memory_guard.py',Path('/usr/lib/aarch64-linux-gnu/libc.so.6')]
    manifest={'scope':'HTTP byte transport only; no model/GPU/qualification result','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'probe':{'sha256':sha256_file(O/'code/probe.py')},'node':str(node),'python':str(python),'tag':'glm53-transfer-001','wrapper':str(wrapper),'broad_baseline':baseline,'safety':{'minimum_start_gib':110,'kill_floor_gib':64,'timeout_seconds':180,'memory_high_gib':32,'memory_max_gib':34},'environment':{'HOME':str(O/'state'),'LANG':'C.UTF-8','PATH':'/usr/bin:/bin','CUDA_VISIBLE_DEVICES':''},'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)} for p in paths],'frozen_at_unix':time.time()}
    (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'output':str(O),'frozen_at_unix':manifest['frozen_at_unix']}),flush=True)
