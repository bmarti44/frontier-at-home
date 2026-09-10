"""Freeze this named-profile candidate before fetching its confirmation seed."""
import hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
repo=Path('/home/bmarti44/spark-deepseek-v4-flash');base=Path('/home/bmarti44/.cache/glm53-flash');out=base/'qualification-freeze-003'
if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True):raise ValueError('freeze requires clean source')
if shutil.disk_usage(base).free < 2 * 1024**3:raise ValueError('freeze requires at least 2 GiB disk headroom')
mem=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines())
if int(mem['SwapTotal'].split()[0])!=0 or len(Path('/proc/swaps').read_text().splitlines())!=1:raise ValueError('owner-approved temporary swap pause required')
out.mkdir(exist_ok=False)
sys.path.insert(0,str(repo/'scripts/lib'));import glm53_profile as api
spec=importlib.util.spec_from_file_location('launcher',repo/'scripts/47_run_glm53_dev.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
snapshot=api.resolve_profile('glm-5.3-flash/cuda-spark-128g-1m-experimental',None,out/'run-root-placeholder')
print('verifying closed runtime and model inventories',flush=True);api.verify_artifacts(snapshot)
print('closed runtime and model inventories verified',flush=True)
binder=repo/'results/glm53-flash-gates/startup-heap-trim-001/bind_libc.py'
libc=json.loads(subprocess.check_output([snapshot['binary'],'-I','-B',str(binder)],text=True))
provider=Path(libc['provider']['path'])
assert provider.stat().st_size==libc['provider']['size_bytes'] and sha(provider)==libc['provider']['sha256']
save(out/'libc-provider.json',libc)
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
planned=out/'fixture-profile';planned.mkdir()
save(planned/'launch.json',{'scope':'declared fixture configuration only; no running server observation','arguments':snapshot['argv'][4:]})
code=out/'code/scripts';code.mkdir(parents=True)
shutil.copyfile(repo/'scripts/103_verify_drand_receipt_bundle.mjs',code/'103_verify_drand_receipt_bundle.mjs')
shutil.copyfile(base/'context-freeze-003/fetch-randomness.py',out/'fetch-randomness.py')
shutil.copyfile(__file__,out/'freeze.py')
paths=[repo/p for p in ['scripts/93_profile_serve.sh','scripts/47_run_glm53_dev.py','scripts/48_probe_glm53_context.py','scripts/57_dsv4_context_probe.py','scripts/38_guard_glm53_probe.py','scripts/lib/profile_resolver.py','scripts/lib/glm53_profile.py','scripts/lib/glm53_contract.py','scripts/lib/glm53_runtime_policy.py','scripts/lib/glm53_flashinfer_cache.py','scripts/lib/glm53_worker.py','configs/hosts/spark-aba1.json','configs/profiles/glm-5.3-flash/model.json','configs/profiles/glm-5.3-flash/cuda-spark-128g-1m-experimental.json','configs/profiles/glm-5.3-flash/cuda-spark-128g-1m.json','configs/build-manifests/glm53-flash-local.json','results/glm52-gates/harness/glm_safe_run.sh','results/glm52-gates/harness/glm_cgroup_run.sh','results/glm53-flash-gates/context-clear-instruction-001/prepare_inputs.py','results/glm53-flash-gates/context-clear-instruction-001/PROTOCOL.md','results/glm53-flash-gates/profile-launch-001/ACCEPTANCE.md','fixtures/ctx-32k.txt','results/glm53-flash-gates/reference-binding-001/summary.json']]
paths += [repo/p for p in ['results/glm53-flash-gates/soak-native-001/run.py', 'results/glm53-flash-gates/soak-native-001/test_run.py', 'results/glm53-flash-gates/soak-native-001/test_prepare.py', 'results/glm53-flash-gates/soak-native-001/PROTOCOL.md', 'scripts/35_soak.py']]
paths += [repo/'results/glm53-flash-gates/startup-cache-release-001/PROTOCOL.md', repo/'results/glm53-flash-gates/startup-cache-release-001/test_worker.py']
paths += [repo/'results/glm53-flash-gates/startup-file-cache-001/PROTOCOL.md', repo/'results/glm53-flash-gates/startup-file-cache-001/test_serve.py']
paths += [repo/'results/glm53-flash-gates/soak-scheduler-001'/name for name in ['candidate.py','test_candidate.py','test_terminal.py','PROTOCOL.md','cpu-audit-candidate2.json']]
paths += [repo/'scripts/tests/test_glm53_experimental_profiles.py']
paths += [repo/'results/glm53-flash-gates/soak-scheduler-002'/name for name in ['candidate.py','test_candidate.py','PROTOCOL.md','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/soak-native-004/summary.json', repo/'results/glm53-flash-gates/soak-native-004/manifest.json']
paths += [repo/'results/glm53-flash-gates/soak-cache-replay-001'/name for name in ['test_reuse.py','PROTOCOL.md','cpu-audit.json','phase.py']]
paths += [repo/'results/glm53-flash-gates/swap-observation-001'/name for name in ['observe.py','PROTOCOL.md','test_observe.py']]
paths += [repo/'results/glm53-flash-gates/soak-native-005'/name for name in ['summary.json','manifest.json']]
paths += [repo/'results/glm53-flash-gates/startup-final-warmup-001'/name for name in ['PROTOCOL.md','test_worker.py','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/soak-native-006'/name for name in ['summary.json','manifest.json']]
paths += [repo/'results/glm53-flash-gates/soak-cache-replay-002'/name for name in ['PROTOCOL.md','test_reuse.py','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/soak-native-007'/name for name in ['summary.json','manifest.json']]
paths += [repo/'results/glm53-flash-gates/startup-heap-trim-001'/name for name in ['PROTOCOL.md','test_worker.py','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/soak-native-008'/name for name in ['summary.json','manifest.json']]
paths += [Path(r['path']) for r in snapshot['digest_checks']]
paths += [binder,binder.with_name('test_libc_binding.py'),provider,out/'libc-provider.json']
paths += [repo/'results/glm53-flash-gates/context-native-profile-002/bind_launch.py',out/'fixture-profile/launch.json',out/'freeze.py',out/'fetch-randomness.py',out/'prepared-cache-inventory.json',out/'profile-snapshot-template.json',code/'103_verify_drand_receipt_bundle.mjs',Path('/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node')]
paths += [repo/'results/glm53-flash-gates/host-swap-accounting-001'/name for name in ['observe.py','test_observe.py','PROTOCOL.md','REPLAY.md','freeze_retry.py','WARM-CONTROL-REPLAY.md','freeze_warm_replay.py','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/host-swap-accounting-001/unloaded-control-001'/name for name in ['manifest.json','summary.json']]
paths += [repo/'results/glm53-flash-gates/startup-heap-trim-001/freeze.py']
paths += [repo/'docs/GLM53-DOCKER-ISOLATION.md',repo/'results/glm53-flash-gates/docker-isolation-001/PROTOCOL.md',repo/'results/glm53-flash-gates/docker-isolation-001/freeze.py']
paths += [repo/'results/glm53-flash-gates/containerd-isolation-001/PROTOCOL.md',repo/'results/glm53-flash-gates/containerd-isolation-001/freeze.py',repo/'results/glm53-flash-gates/runtime-access-001/installed-verification.json']
paths += [repo/'results/glm53-flash-gates/context-scheduler-003'/name for name in ['adapter.py','test_adapter.py','PROTOCOL.md','cpu-audit.json']]
paths += [repo/'results/glm53-flash-gates/context-clear-instruction-001/run_short.py',repo/'results/glm53-flash-gates/swap-paused-qualification-001/PROTOCOL.md',repo/'results/glm53-flash-gates/host-swap-pause-001/owner-execution-observed.json']
paths += [repo/'results/glm53-flash-gates/context-short-padding-001'/name for name in ['test_prepare.py','failed-beacon.json.gz','PROTOCOL.md']]
paths += [repo/'results/glm53-flash-gates/startup-serialized-replay-001/PROTOCOL.md']
manifest={'scope':'Owner-paused swap and stopped container runtimes; current full-context profile with unchanged numerical code and closed direct/durability scorers. Direct invocation uses the reviewed512/128 validator adapter. Separate fresh server attempts for direct context and durability; no default or production promotion.','source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'profile':snapshot['profile_id'],'request_configuration':{'max_tokens':2048,'input_tokens_per_request':4224,'workers':4,'admission_seconds':1800,'drain_limit_seconds':2400,'first_window_admission_seconds':300,'first_window_requests_per_worker':5,'first_window_drain_limit_seconds':900},'ephemeral_path':'run-root-placeholder is replaced by the actual unique operator launch directory; exact argv/env and relocated cache bindings must be retained before requests','closed_runtime_model_verification':True,'prepared_cache_files':len(cache),'files':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in dict.fromkeys(paths)],'frozen_at_unix':time.time()}
manifest['direct_context_configuration']={'slots':4,'context_per_slot':262144,'aggregate_cap':1048576,'input_tokens_per_request':250128,'aggregate_input_tokens':1000512,'max_output_tokens':2048,'adapter':str(repo/'results/glm53-flash-gates/context-scheduler-003/adapter.py')}
save(out/'manifest.json',manifest)
print(json.dumps({'freeze_complete':True,'prepared_cache_files':len(cache),'frozen_at_unix':manifest['frozen_at_unix']}),flush=True)
