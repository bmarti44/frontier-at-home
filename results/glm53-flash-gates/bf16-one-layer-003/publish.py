"""Preserve the actual native-layer replay, including any failed or null outcome."""
from pathlib import Path
import gzip,hashlib,json,re,shutil,tarfile
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-one-layer-003';S=R/'results/glm53-flash-gates/bf16-one-layer-003'
def read(p):return json.loads(p.read_bytes())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=read(O/'manifest.json');outer=read(O/'summary.json');capture=read(O/'capture.json');terminal=read(O/'post-terminal-observation.json')
assert not read(O/'cgroup-after.json')['exists']
assert 'ActiveState=inactive' in read(O/'unit-after.json')['stdout']
assert terminal['matching_process_group_rows']==[] and not terminal['proc_pid_exists']
assert not terminal['identity_summary_present'] and outer['verdict']=='FAIL' and capture['wrapper_exit_code']==15
for row in frozen['files']:
 p=Path(row['path']);assert p.stat().st_size==row['size_bytes'] and digest(p)==row['sha256'],str(p)
S.mkdir(exist_ok=False)
external=[B/('bf16-one-layer-003-'+stage+'.'+stream) for stage in ['freeze','beacon','controller'] for stream in ['stdout','stderr']]
entries=[(p,'attempt/'+p.relative_to(O).as_posix()) for p in sorted(O.rglob('*')) if p.is_file()]+[(p,'controller/'+p.name) for p in external]
with tarfile.open(S/'attempt.tar.gz','w:gz') as archive:
 for p,name in entries:archive.add(p,arcname=name)
with tarfile.open(S/'attempt.tar.gz') as archive:
 assert len(archive.getmembers())==len(entries)
 for p,name in entries:assert archive.extractfile(name).read()==p.read_bytes()
raw=[]
for name in ['checks/raw.jsonl','identity/raw.jsonl','samples.log']:
 for line in (O/name).read_text().splitlines():raw.append({'stream':name,'original_line':line})
(S/'raw.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(x)+'\n' for x in raw)).encode(),mtime=0))
shutil.copyfile(O/'summary.json',S/'summary.json');shutil.copyfile(__file__,S/'publish.py')
samples=(O/'samples.log').read_text().splitlines();before=frozen['broad_baseline'];after=read(O/'post-verification-host.json');probe=[json.loads(x) for x in (O/'checks/raw.jsonl').read_text().splitlines()]
publication={'scope':'One real native BF16 KDA layer with synthetic activations only; no full-model native reference, fidelity, context or serving-speed qualification','verdict':outer['verdict'],'archive_member_bytes_verified':len(entries),'raw_lines':len(raw),'frozen_bindings_verified':len(frozen['files']),'wrapper_exit_code':capture['wrapper_exit_code'],'identity_verdict':'NO_RESULT: terminal guard summary missing','identity_samples':terminal['identity_samples'],'separate_postterminal_process_group_rows':[],'cgroup_gone':True,'unit_inactive':True,'external_memory_samples':len(samples),'minimum_available_kib':min(int(re.search(r'mem_avail_kb=(\d+)',x)[1]) for x in samples),'maximum_cgroup_swap_bytes':max(int(re.search(r'cgroup_swap_current_bytes=(\d+)',x)[1]) for x in samples),'broad_swap_deltas':{k:after[k]-before[k] for k in ['pswpin','pswpout','used_swap_kib']},'completed_shard_receipts':sum(x['kind']=='verified_shard' for x in probe),'GPU_weights_receipt':any(x['kind']=='weights_on_gpu' for x in probe),'forward_completion_receipt':any(x['kind']=='forward_complete' for x in probe),'formulas':{'minimum_available_kib':'min mem_avail_kb over all external samples','maximum_cgroup_swap_bytes':'max cgroup_swap_current_bytes over all external samples','broad_swap_deltas':'post-verification minus frozen broad baseline'}}
(S/'publication.json').write_text(json.dumps(publication,indent=2)+'\n')
manifest={'source_revision':frozen['source_revision'],'scope':publication['scope'],'frozen_manifest':{'sha256':digest(O/'manifest.json')},'public_randomness_receipt':{'sha256':digest(O/'randomness.json')},'files':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(S.iterdir())],'archive_members':[{'path':name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p,name in entries]}
(S/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(publication),flush=True)
