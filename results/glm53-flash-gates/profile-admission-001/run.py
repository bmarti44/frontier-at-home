import hashlib,json,os,subprocess,sys,time
from pathlib import Path
if sys.flags.optimize: raise RuntimeError('assertions required')
root=Path('/home/bmarti44/spark-deepseek-v4-flash')
out=Path('/home/bmarti44/.cache/glm53-flash/profile-admission-001')
out.mkdir(exist_ok=False)
sys.path.insert(0,str(root/'scripts/lib'))
import glm53_profile as api
profile='glm-5.3-flash/cuda-spark-128g-agent-fast'
requested=out/'must-not-launch'
unit='glm52-glm53-context-1770728-1770746.service'
record_path=api.lifecycle_path(profile)
def write(name,data): (out/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def observe():
 memory=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
 return {'time_unix':time.time(),'mem_available_kb':int(memory['MemAvailable'].split()[0]),
         'unit':api.unit_properties(unit),
         'processes':[api.process_identity(p) for p in (1770728,1770801,1771589)],
         'guard_services':subprocess.check_output(['systemctl','is-active','dsv4-engine-restore.service','dsv4-guard.timer'],text=True).splitlines()}
command=[sys.executable,'-B',str(root/'scripts/47_run_glm53_dev.py'),'--start','--profile',profile,'--output',str(requested)]
paths=[root/'scripts/47_run_glm53_dev.py',root/'scripts/lib/glm53_profile.py',root/'scripts/lib/profile_resolver.py',root/'configs/profiles/glm-5.3-flash/cuda-spark-128g-agent-fast.json']
write('manifest.json',{'scope':'real low-memory admission rejection only; no second model or qualification claim','command':command,'acceptance_formula':'nonzero exit with explicit 110 GiB rejection, no output/identity/wrapper artifacts created, unchanged existing model process identities and service InvocationID, both guard services active, memory above existing run floor','sources':[{'path':str(p),'sha256':sha(p)} for p in paths]})
before=observe();write('before.json',before)
assert 18*1024**2 <= before['mem_available_kb'] < 110*1024**2
assert before['unit']['ActiveState']=='active'
assert not record_path.exists(),'preserve an existing real profile lifecycle record'
result=subprocess.run(command,capture_output=True,text=True,timeout=20)
(out/'stdout.txt').write_text(result.stdout);(out/'stderr.txt').write_text(result.stderr)
after=observe();write('after.json',after)
record=json.loads(record_path.read_text());write('lifecycle-record.json',record)
checks={'rejected':result.returncode!=0,'reason':'GLM start requires 110 GiB available; stop the active model first' in result.stderr,
        'no_launch_directory':not requested.exists(),'preparation_exited_before_unit':record['phase']=='exited' and record['unit'] is None,
        'existing_processes_unchanged':before['processes']==after['processes'],
        'existing_unit_unchanged':before['unit']==after['unit'],
        'guards_active':before['guard_services']==after['guard_services']==['active','active'],
        'available_memory_above_floor':min(before['mem_available_kb'],after['mem_available_kb'])>=18*1024**2}
summary={'scope':'real named-profile low-memory admission; not actual launch qualification','returncode':result.returncode,'checks':checks,'verdict':'PASS' if all(checks.values()) else 'FAIL'}
write('summary.json',summary)
print(json.dumps(summary),flush=True)
assert all(checks.values())
