from pathlib import Path
import json,hashlib,gzip,tarfile,re,time,subprocess
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';O=B/'bf16-one-layer-002';S=R/'results/glm53-flash-gates/bf16-one-layer-002';S.mkdir()
def j(p):return json.loads(p.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
frozen=j(O/'manifest.json');outer=j(O/'summary.json');capture=j(O/'capture.json');identity=j(O/'identity/summary.json');base=frozen['broad_baseline'];after=j(O/'post-verification-host.json');inner=j(O/'checks/summary.json')
assert outer['verdict']==inner['verdict']=='FAIL' and capture['wrapper_exit_code']==1 and capture['capture_failure'] is None
assert identity['verdict']=='FAIL' and identity['probe_exit_code']==1 and identity['live_process_group_after']==[]
assert not j(O/'cgroup-after.json')['exists'] and 'ActiveState=inactive' in j(O/'unit-after.json')['stdout']
assert after['pswpin']==base['pswpin'] and after['pswpout']-base['pswpout']==5 and after['used_swap_kib']-base['used_swap_kib']==20
probe=[json.loads(x) for x in (O/'checks/raw.jsonl').read_text().splitlines()]
assert probe[-1]['kind']=='failure' and 'broad host swap changed' in probe[-1]['error']
assert not any(x['kind'] in ('verified_shard','weights_on_gpu','forward_complete') for x in probe)
samples=(O/'samples.log').read_text().splitlines();memory=[int(re.search(r'mem_avail_kb=(\d+)',s)[1]) for s in samples];swaps=[int(re.search(r'cgroup_swap_current_bytes=(\d+)',s)[1]) for s in samples]
assert not any(swaps) and min(memory)>=40*1024**2
for row in frozen['files']:
 p=Path(row['path']);assert p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256'],str(p)
external=[B/'bf16-one-layer-002-freeze.stdout',B/'bf16-one-layer-002-freeze.stderr',B/'bf16-one-layer-002-controller.stdout',B/'bf16-one-layer-002-controller.stderr']
entries=[(p,'attempt/'+p.relative_to(O).as_posix()) for p in sorted(O.rglob('*')) if p.is_file()]+[(p,'controller/'+p.name) for p in external]
with tarfile.open(S/'attempt.tar.gz','w:gz') as a:
 for p,name in entries:a.add(p,arcname=name)
with tarfile.open(S/'attempt.tar.gz') as a:
 assert len(a.getmembers())==len(entries)
 for p,name in entries:assert a.extractfile(name).read()==p.read_bytes()
raw=[]
for name in ['checks/raw.jsonl','identity/raw.jsonl','samples.log']:
 for line in (O/name).read_text().splitlines():raw.append({'stream':name,'original_line':line})
(S/'raw.jsonl.gz').write_bytes(gzip.compress((''.join(json.dumps(x)+'\n' for x in raw)).encode(),mtime=0))
summary={'verdict':'FAIL','scope':'One-layer native BF16 feasibility attempt; no GPU computation, native reference, fidelity, context or serving-performance result','failure':'Broad host swap-out changed before first verified-shard receipt could be recorded','native_feasibility':'NO_RESULT','measurements':{'global_swap_in_pages_delta':after['pswpin']-base['pswpin'],'global_swap_out_pages_delta':after['pswpout']-base['pswpout'],'used_swap_kib_delta':after['used_swap_kib']-base['used_swap_kib'],'external_memory_samples':len(samples),'minimum_mem_available_kib':min(memory),'minimum_mem_available_gib':min(memory)/1024**2,'maximum_observed_cgroup_swap_bytes':max(swaps),'maximum_observed_cgroup_peak_bytes':max(int(re.search(r'cgroup_peak_bytes=(\d+)',s)[1]) for s in samples),'identity_samples':identity['identity_samples']},'formulas':{'swap_deltas':'post-verification host counter minus broad pre-freeze baseline','memory_floor':'minimum mem_avail_kb over retained external samples','cgroup_swap':'maximum cgroup_swap_current_bytes over retained external samples'},'checks':{'frozen_file_bindings':len(frozen['files']),'frozen_files_unchanged':True,'archive_member_bytes_verified':len(entries),'wrapper_exit_code':1,'capture_failure':None,'identity_verdict':'FAIL: missing success handshake after probe exception','live_process_group_after':[],'cgroup_gone':True,'unit_inactive':True,'no_verified_shard_receipt':True,'no_GPU_forward_or_output':True},'limits':['Transfer reached first-shard verification/copy source line, but no completed verified-shard receipt was retained. Do not claim a completed real-layer check.','Global swap owner and causal trigger are unknown. Zero sampled probe cgroup swap does not attribute the global writes.','Native GPU memory feasibility and the full 100-case native reference remain unmeasured.']}
(S/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
manifest={'source_revision':frozen['source_revision'],'scope':'Publication of failed real-layer attempt002; original frozen manifest and raw files preserved byte-for-byte in archive','files':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in [S/'attempt.tar.gz',S/'raw.jsonl.gz',S/'summary.json']],'frozen_manifest':{'sha256':sha(O/'manifest.json')},'public_randomness_receipt':{'sha256':sha(O/'randomness.json')},'archive_members':[{'path':name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p,name in entries]}
(S/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(S/'README.md').write_text('''# Native BF16 layer attempt 002: FAIL; feasibility NO_RESULT\n\nThe reviewed probe passed the corrected runtime freeze and later verified public\nseed. It then failed the unchanged host gate at the first shard's verification/\ncopy stage: five host pages (20 KiB) were written to swap. No completed shard\nreceipt, GPU forward or output was recorded. This is not native reference data,\na fidelity result, a context result or a serving-speed result.\n\nExternal memory stayed above 100 GiB. Sampled probe cgroup swap stayed zero,\nbut the owner and trigger of the global writes are unknown. The probe exited 1;\nthe identity guard correctly reported missing successful completion. The process\ngroup and cgroup were gone afterward and memory recovered. Keep the overall\nFAIL; individual cleanup observations do not promote the result.\n\nThe archive retains the frozen source/configuration/runtime inventory, pinned\nmodel shard metadata, later public seed, controller invocation, original raw\nlogs, memory/identity sampling, failures and terminal cleanup. Publication\nverified every archived member byte against the original and all frozen file\nbindings. Raw source lines are also provided in raw.jsonl.gz.\n\nDocker service/socket and containerd remained stopped. Earlier isolated attempts\nalso failed global swap gates. No new warmup or additional daemon-stop variant\nis proposed; native GPU feasibility and full qualification remain unresolved.\n''')
print(json.dumps({'verdict':'FAIL','archive_members':len(entries),'archive_bytes':(S/'attempt.tar.gz').stat().st_size,'raw_rows':len(raw),'measurements':summary['measurements']}))
