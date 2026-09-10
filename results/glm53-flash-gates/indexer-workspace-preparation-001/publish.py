"""Archive the actual model-free baseline mismatch, without revising its verdict."""
from pathlib import Path
import gzip,hashlib,json,tarfile,shutil,re
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path('/home/bmarti44/.cache/glm53-flash')
O=B/'indexer-workspace-preparation-001';S=R/'results/glm53-flash-gates/indexer-workspace-preparation-001'
def read(p):return json.loads(p.read_bytes())
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
outer=read(O/'summary.json');frozen=read(O/'manifest.json');after=read(O/'operator-after.json')
assert outer['verdict']=='FAIL' and outer['completed_arms']==4 and outer['failure']=="ValueError('ordered artifact bytes differ')"
assert sorted(p.name for p in (O/'arms').iterdir())==['0','1','2','3']
assert all(not r['proc_exists'] and not r['cgroup_exists'] for r in after['arms'])
assert not (O/'terminal-host.json').exists()
assert read(O/'separate-post-verification.json')['runtime_inventory']['verdict']=='FAIL'
assert (O/'runtime-changed-bytecode.tar.gz').is_file()
accounting=read(O/'runtime-extras-accounting.json')
S.mkdir(exist_ok=False)
entries=[(p,'attempt/'+p.relative_to(O).as_posix()) for p in sorted(O.rglob('*')) if p.is_file()]
entries += [(Path(__file__),'publication/publish.py')]
archive=B/'indexer-workspace-preparation-001-attempt.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 for p,name in entries:
  assert not p.is_symlink();tf.add(p,arcname=name,recursive=False)
with tarfile.open(archive,'r:gz') as tf:
 assert len(tf.getmembers())==len(entries)
 for p,name in entries:assert hashlib.file_digest(tf.extractfile(name),'sha256').hexdigest()==digest(p),name
parts=[]
with archive.open('rb') as f:
 while data:=f.read(32*1024**2):
  p=S/f'attempt.tar.gz.part{len(parts):02d}';p.write_bytes(data);parts.append(p)
raw=[]
for a in sorted((O/'arms').iterdir()):
 for name in ['checks/raw.jsonl','identity/raw.jsonl','samples.log']:
  p=a/name
  for line in p.read_text().splitlines():raw.append({'stream':p.relative_to(O).as_posix(),'original_line':line})
(S/'raw.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(x)+'\n' for x in raw)).encode(),mtime=0))
shutil.copyfile(O/'summary.json',S/'summary.json');shutil.copyfile(__file__,S/'publish.py')
arms=[]
for a in sorted((O/'arms').iterdir()):
 d=read(a/'combined.json');h=d['host'];arms.append({'index':int(a.name),**d['arm'],'host_verdict':h['verdict'],'inner_verdict':d['inner']['verdict'],'minimum_available_kib':h['minimum_mem_available_kib'],'memory_samples':h['memory_samples'],'identity_samples':h['identity_samples'],'wrapper_exit_code':read(a/'capture.json')['wrapper_exit_code']})
diagnosis=read(O/'order-diagnosis.json')
publication={'scope':'Model-free baseline repeatability falsifier only; no serving, quality, context or performance qualification','verdict':'FAIL','completed_arms':arms,'candidate_64mib_arms_started':False,'controller_exit_code':read(O/'controller-exit.json')['exit_code'],'failure':outer['failure'],'prefill_1_baseline_comparison':'PASS','prefill_4_baseline_comparison':'FAIL','order_diagnosis_file':'attempt/order-diagnosis.json','downstream_attention_equivalence':'NOT_MEASURED','minimum_available_kib':min(a['minimum_available_kib'] for a in arms),'formulas':{'minimum_available_kib':'minimum of exact host-scorer minimum_mem_available_kib over all four arms'},'all_four_processes_and_cgroups_gone':True,'original_terminal_host_record':'MISSING','separate_frozen_binding_check':read(O/'separate-post-verification.json')['frozen_bindings'],'original_runtime_inventory':'FAIL:149 changed inventoried bytecode files and830 extra bytecode files','original_61401_files_matched':accounting['original_verified_files'],'original_file_errors':len(accounting['original_file_errors']),'archive_member_bytes_verified':len(entries),'raw_lines':len(raw),'profile_or_default_changed':False,'smaller_workspace_qualification':'NO_RESULT: no candidate arm admitted; this branch stops without a new seed/retry'}
save(S/'publication.json',publication)
(S/'README.md').write_text('# Indexer workspace preparation001: FAIL\n\nFour fresh512MiB baseline arms passed their individual correctness, identity and\nhost checks. The single-request pair was byte-identical; the four-request pair\ndiffered in ordered indices. Cache, tail, valid logits and consumed pool sets\nmatched, but downstream attention-output equivalence was not measured. The fixed\nprotocol stopped before either64MiB arm. No smaller-buffer saving, speed,\nfull-model fidelity or context result is established.\n\nOriginal terminal-host.json is missing. Separate post-run checks verified all98\nfrozen file bindings but found149 changed inventoried bytecode files and830\nextra bytecode files. The original runtime inventory failed; full accounting\nand changed/extra bytes are retained. All four processes/cgroups were gone and host\nmemory recovered. The original failure is preserved, including the secondary\ninventory failure. No serving profile or default changed.\n\nThe split archive reconstructs in numeric part order. Its member hashes and the\nwhole compressed hash are recorded in manifest.json.gz.\n')
manifest={'source_revision':frozen['source_revision'],'scope':publication['scope'],'frozen_manifest_sha256':digest(O/'manifest.json'),'randomness_receipt_sha256':digest(O/'randomness.json'),'archive':{'size_bytes':archive.stat().st_size,'sha256':digest(archive),'parts':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in parts]},'files':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(S.iterdir()) if p.is_file()],'archive_members':[{'path':name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p,name in entries]}
(S/'manifest.json.gz').write_bytes(gzip.compress((json.dumps(manifest,indent=2)+'\n').encode(),mtime=0))
print(json.dumps({'verdict':'FAIL','archive_bytes':archive.stat().st_size,'parts':len(parts),'members':len(entries),'raw_lines':len(raw)}),flush=True)
