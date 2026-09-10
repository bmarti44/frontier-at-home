from pathlib import Path
import json,importlib.util,hashlib,time
R=Path('/home/bmarti44/spark-deepseek-v4-flash');B=Path.home()/'.cache/glm53-flash';old=B/'soak-native-011-observations/host-accounting/raw.jsonl'
last=None
for line in old.open():
 row=json.loads(line)
 if row.get('kind')=='sample':last=row
assert last is not None
source=R/'results/glm53-flash-gates/host-swap-accounting-001/observe.py';spec=importlib.util.spec_from_file_location('existing',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
before=mod.previous.swap_counters();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip();groups={};errors=[]
for name,prior in last['groups'].items():
 p=Path('/sys/fs/cgroup')/name
 try:
  a=p.stat();raw=(p/'memory.stat').read_text();z=p.stat();identity=(z.st_dev,z.st_ino);assert (a.st_dev,a.st_ino)==identity
  current=mod.counters(raw);stable=identity==(prior['device'],prior['inode']) and boot==last['boot_id']
  groups[name]={'device':z.st_dev,'inode':z.st_ino,'raw':raw,'counters':current,'prior_identity_matches':stable,'delta':{k:current[k]-prior['counters'][k] for k in ('pswpin','pswpout')} if stable else None}
 except Exception as e:errors.append({'group':name,'error':repr(e)})
after=mod.previous.swap_counters();value={'scope':'Posthoc interval comparison only. Includes all activity since the older observation, not just the native attempt. Stable cgroup counters identify accounting ownership, never causal trigger. Missing/recreated groups are retained as incomplete coverage.','time_unix':time.time(),'source':{'path':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},'prior_source':{'path':str(old),'sha256':hashlib.sha256(old.read_bytes()).hexdigest()},'prior_sample':last,'current':{'before':before,'groups':groups,'after':after,'boot_id':boot,'errors':errors}}
out=B/'bf16-one-layer-002-posthoc-swap.json';out.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps({'prior_global':last['after'],'current_global':after,'group_deltas':{k:v['delta'] for k,v in groups.items()},'errors':errors}))
