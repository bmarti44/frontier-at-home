"""Retain actual native005 result; never infer full-model fidelity from one layer."""
from pathlib import Path
import gzip,hashlib,json,re,shutil,tarfile
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-one-layer-005';S=R/'results/glm53-flash-gates/bf16-one-layer-005'
def read(p):return json.loads(p.read_bytes())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
frozen=read(O/'manifest.json');outer=read(O/'summary.json');capture=read(O/'capture.json');terminal=read(O/'post-terminal-observation.json')
assert outer['verdict'] in ('PASS','FAIL') and not read(O/'cgroup-after.json')['exists']
assert 'ActiveState=inactive' in read(O/'unit-after.json')['stdout']
assert terminal['matching_process_group_rows']==[] and not terminal['proc_pid_exists']
assert len(frozen['files'])==41
for row in frozen['files']:
 p=Path(row['path']);assert p.stat().st_size==row['size_bytes'] and digest(p)==row['sha256'],str(p)
guard=read(O/'identity/summary.json') if (O/'identity/summary.json').exists() else None
inner=read(O/'checks/summary.json') if (O/'checks/summary.json').exists() else None
if outer['verdict']=='PASS':
 assert capture['wrapper_exit_code']==0 and guard['verdict']==inner['verdict']=='PASS' and outer['failure'] is None
 assert outer['host'] is not None and outer['inner']['verdict']=='PASS'
S.mkdir(exist_ok=False)
entries=[(p,'attempt/'+p.relative_to(O).as_posix()) for p in sorted(O.rglob('*')) if p.is_file()]
entries += [(Path(__file__),'publication/publish.py'),(Path('/tmp/glm53-native005-terminal.py'),'publication/terminal.py')]
with tarfile.open(S/'attempt.tar.gz','w:gz') as archive:
 for p,name in entries:archive.add(p,arcname=name)
with tarfile.open(S/'attempt.tar.gz') as archive:
 assert len(archive.getmembers())==len(entries)
 for p,name in entries:assert hashlib.file_digest(archive.extractfile(name),'sha256').hexdigest()==digest(p)
raw=[]
for name in ['checks/raw.jsonl','identity/raw.jsonl','samples.log']:
 for line in (O/name).read_text().splitlines():raw.append({'stream':name,'original_line':line})
(S/'raw.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(x)+'\n' for x in raw)).encode(),mtime=0))
shutil.copyfile(O/'summary.json',S/'summary.json');shutil.copyfile(__file__,S/'publish.py')
samples=(O/'samples.log').read_text().splitlines();before=frozen['broad_baseline'];after=read(O/'post-verification-host.json');probe=[json.loads(x) for x in (O/'checks/raw.jsonl').read_text().splitlines()]
publication={'scope':'One real native BF16 KDA layer with synthetic activations only; no full-model reference, fidelity, context or serving-speed qualification','verdict':outer['verdict'],'archive_member_bytes_verified':len(entries),'raw_lines':len(raw),'frozen_bindings_verified':len(frozen['files']),'wrapper_exit_code':capture['wrapper_exit_code'],'identity_verdict':guard['verdict'] if guard else None,'inner_probe_verdict':inner['verdict'] if inner else None,'outer_failure':outer['failure'],'identity_samples':terminal['identity_samples'],'separate_postterminal_process_group_rows':[],'cgroup_gone':True,'unit_inactive':True,'external_memory_samples':len(samples),'minimum_available_kib':min(int(re.search(r'mem_avail_kb=(\d+)',x)[1]) for x in samples),'maximum_cgroup_swap_bytes':max(int(re.search(r'cgroup_swap_current_bytes=(\d+)',x)[1]) for x in samples),'broad_swap_deltas':{k:after[k]-before[k] for k in ['pswpin','pswpout','used_swap_kib']},'completed_shard_receipts':sum(x['kind']=='verified_shard' for x in probe),'GPU_weights_receipt':any(x['kind']=='weights_on_gpu' for x in probe),'forward_completion_receipt':any(x['kind']=='forward_complete' for x in probe),'formulas':{'minimum_available_kib':'min mem_avail_kb over all external samples','maximum_cgroup_swap_bytes':'max cgroup_swap_current_bytes over all external samples','broad_swap_deltas':'post-verification minus frozen broad baseline'}}
(S/'publication.json').write_text(json.dumps(publication,indent=2)+'\n')
(S/'README.md').write_text('# Native BF16 layer005: '+outer['verdict']+'\n\nThis fresh confirmation uses the reviewed terminal-coverage correction with\nunchanged native computation, source/runtime bindings, pinned staging and safety\nlimits. The original 004 result remains FAIL. The archived original summary and\nraw observations determine this attempt\'s result.\n\nScope: one real layer with synthetic 516-token activations. This does not supply\nfull-model reference probabilities or complete the 100-case fidelity comparison.\n')
manifest={'source_revision':frozen['source_revision'],'scope':publication['scope'],'frozen_manifest':{'sha256':digest(O/'manifest.json')},'public_randomness_receipt':{'sha256':digest(O/'randomness.json')},'files':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(S.iterdir())],'archive_members':[{'path':name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p,name in entries]}
(S/'manifest.json.gz').write_bytes(gzip.compress((json.dumps(manifest,indent=2)+'\n').encode(),mtime=0));print(json.dumps(publication),flush=True)
