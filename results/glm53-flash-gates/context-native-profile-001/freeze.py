"""Freeze this named-profile candidate before fetching its confirmation seed."""
import hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/home/bmarti44/spark-deepseek-v4-flash');base=Path('/home/bmarti44/.cache/glm53-flash');out=base/'context-freeze-005'
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True):raise ValueError('freeze requires clean source')
out.mkdir(exist_ok=False)
sys.path.insert(0,str(repo/'scripts/lib'));import glm53_profile as api
spec=importlib.util.spec_from_file_location('launcher',repo/'scripts/47_run_glm53_dev.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
snapshot=api.resolve_profile('glm-5.3-flash/cuda-spark-128g-1m-experimental',None,out/'run-root-placeholder')
print('verifying closed runtime and model inventories',flush=True);api.verify_artifacts(snapshot)
print('closed runtime and model inventories verified',flush=True)
state=out/'prepared-state';launcher.reuse_prepared_kernels(state)
cache=[]
for p in sorted(state.rglob('*')):
 if not p.is_file():continue
 row={'path':str(p.relative_to(state)),'size_bytes':p.stat().st_size,'sha256':sha(p)}
 if p.name.startswith('__grp__') and p.suffix=='.json':
  data=json.loads(p.read_text());children={}
  for k,v in data['child_paths'].items():
   q=Path(v)
   if not q.is_relative_to(state) or not q.is_file():raise ValueError('prepared cache child is not closed')
   children[k]='{state_root}/'+str(q.relative_to(state))
  normalized=json.dumps({'child_paths':children},sort_keys=True,separators=(',',':')).encode()
  row['relocated_group']={'sha256':hashlib.sha256(normalized).hexdigest()}
 cache.append(row)
save(out/'prepared-cache-inventory.json',{'normalization':'For __grp__*.json only: replace the state-root prefix with {state_root}/, JSON sort_keys with compact separators; all other bytes must match exactly.','files':cache})
save(out/'profile-snapshot-template.json',snapshot)
code=out/'code/scripts';code.mkdir(parents=True)
shutil.copyfile(repo/'scripts/103_verify_drand_receipt_bundle.mjs',code/'103_verify_drand_receipt_bundle.mjs')
shutil.copyfile(base/'context-freeze-003/fetch-randomness.py',out/'fetch-randomness.py')
shutil.copyfile(__file__,out/'freeze.py')
paths=[repo/p for p in ['scripts/93_profile_serve.sh','scripts/47_run_glm53_dev.py','scripts/48_probe_glm53_context.py','scripts/57_dsv4_context_probe.py','scripts/38_guard_glm53_probe.py','scripts/lib/profile_resolver.py','scripts/lib/glm53_profile.py','scripts/lib/glm53_contract.py','scripts/lib/glm53_runtime_policy.py','scripts/lib/glm53_flashinfer_cache.py','scripts/lib/glm53_worker.py','configs/hosts/spark-aba1.json','configs/profiles/glm-5.3-flash/model.json','configs/profiles/glm-5.3-flash/cuda-spark-128g-1m-experimental.json','configs/profiles/glm-5.3-flash/cuda-spark-128g-1m.json','configs/build-manifests/glm53-flash-local.json','results/glm52-gates/harness/glm_safe_run.sh','results/glm52-gates/harness/glm_cgroup_run.sh','results/glm53-flash-gates/context-native-profile-001/prepare_inputs.py','results/glm53-flash-gates/context-native-profile-001/PROTOCOL.md','results/glm53-flash-gates/profile-launch-001/ACCEPTANCE.md','fixtures/ctx-32k.txt','results/glm53-flash-gates/reference-binding-001/summary.json']]
paths += [Path(r['path']) for r in snapshot['digest_checks']]
paths += [out/'freeze.py',out/'fetch-randomness.py',out/'prepared-cache-inventory.json',out/'profile-snapshot-template.json',code/'103_verify_drand_receipt_bundle.mjs',Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node')]
manifest={'scope':'native named-profile direct context candidate; fidelity/performance/switching gates separate','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'profile':snapshot['profile_id'],'request_configuration':{'max_tokens':2048,'input_tokens_per_slot':250128,'slots':4},'ephemeral_path':'run-root-placeholder is replaced by the actual unique operator launch directory; exact argv/env and relocated cache bindings must be retained before requests','closed_runtime_model_verification':True,'prepared_cache_files':len(cache),'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in dict.fromkeys(paths)],'frozen_at_unix':time.time()}
save(out/'manifest.json',manifest)
print(json.dumps({'freeze_complete':True,'prepared_cache_files':len(cache),'frozen_at_unix':manifest['frozen_at_unix']}),flush=True)
