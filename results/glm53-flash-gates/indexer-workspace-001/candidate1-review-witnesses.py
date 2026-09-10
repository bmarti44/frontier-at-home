"""CPU synthetic evidence witnesses only; no model, GPU or network execution."""
import copy,gzip,hashlib,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
R=Path('/home/bmarti44/spark-deepseek-v4-flash');P=R/'results/glm53-flash-gates/indexer-workspace-001/probe.py'
s=importlib.util.spec_from_file_location('workspace_review',P);p=importlib.util.module_from_spec(s);s.loader.exec_module(p)
results=[]
with tempfile.TemporaryDirectory(prefix='glm53-workspace-binding-review-') as d:
 root=Path(d)
 for name in p.REQUIRED_CODE:
  q=root/'code'/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(name)
 (root/'metadata').mkdir();(root/'cache-template').mkdir()
 for q in [root/'metadata/config.json',root/'metadata/runtime-inventory.json',root/'cache-template-inventory.json']:q.write_text('{}')
 rows=[{'path':str(q),'size_bytes':q.stat().st_size,'sha256':p.sha(q)} for q in root.rglob('*') if q.is_file()]
 m={'files':rows,'safety':p.SAFETY,'cases':p.CASES,'budgets_mib':[512,64], 'python':'/unfrozen/python','node':'/unfrozen/node','wrapper':'/unfrozen/wrapper','runtime':{'root':'/unfrozen/runtime','inventory':'/unfrozen/inventory'}}
 p.verify_frozen(root,m)
 results.append({'witness':'selected executable/node/wrapper/runtime inventory paths absent from frozen files','accepted':True,'scope':'Actual verify_frozen with complete required synthetic local tree; no executable launched.'})
with tempfile.TemporaryDirectory(prefix='glm53-workspace-row-review-') as d:
 root=Path(d);seed=42;case='prefill-4';budget=512;cfg=p.fixture.case_config(seed,case)
 indices=np.full((512,2048),-1,dtype='<i4')
 for i,(request,position,old) in enumerate(zip(cfg['row_requests'],cfg['positions'],cfg['history_counts'])):
  pools=p.fixture.top_history(seed,int(request),int(old))[:511]
  indices[i,:2044]=(pools[:,None]*4+np.arange(4)).reshape(-1)
  count=(int(position)+1)%4;indices[i,2044:2044+count]=np.arange(int(position)//4*4,int(position)//4*4+count)
 cache,tail=p.fixture.cache_and_tail(seed,case,final=True)
 for name,v in [('indices.i32.gz',indices),('cache.u8.gz',cache),('tail.bf16.gz',tail)]:
  with gzip.open(root/name,'wb') as f:f.write(v.tobytes())
 with gzip.open(root/'logits.f32.gz','wb') as f:
  for request,position,old in zip(cfg['row_requests'],cfg['positions'],cfg['history_counts']):f.write(p.fixture.valid_logits(seed,int(request),int(old),(int(position)+1)//4).tobytes())
 mem={'cuda_allocated':p.BASE_BYTES+1024,'cuda_reserved':p.BASE_BYTES+512*1024**2+4096,'cuda_peak_allocated':p.BASE_BYTES+512*1024**2+2048,'cuda_peak_reserved':p.BASE_BYTES+512*1024**2+4096,'device_free':100*1024**3,'device_total':120*1024**3,'workspace_bytes':p.WORKSPACE}
 rows=[{'time_unix':1,'event':'configured',**p.geometry(seed,case,budget)},{'time_unix':2,'event':'profiled','workspace_bytes':p.WORKSPACE,'cuda_peak_allocated':mem['cuda_peak_allocated'],'cuda_reserved':mem['cuda_reserved'],'device_total':mem['device_total']},{'time_unix':3,'event':'start','case':case},{'time_unix':4,'event':'output','case':case,'budget_mib':budget,'artifacts':{n:p.sha(root/n) for n in p.artifact_sizes(seed,case)},'calls':[{**p.call_specs(case,budget)[0],'cuda_allocated_at_return':mem['cuda_peak_allocated']}],'metadata':p.expected_metadata(seed,case,budget),'memory':mem,'cuda_elapsed_ms':1,'cache_stride':[8448,132,1],'tail_stride':[1024,512,128,1],'output_alias':True,'input_digests':p.expected_inputs(seed,case)}]
 (root/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(root/'traceback.log').write_text('');(root/'summary.json').write_text('{}');(root/'manifest.json').write_text('{}')
 result=p.score_arm(root,seed,case,budget)
 results.append({'witness':'full-geometry synthetic arm with times1,2,3,4 outside any current guard interval','accepted':result['verdict'],'scope':'Actual score_arm, actual complete CPU fixture bytes and oracle, no patched dependency and no GPU claim; controller currently adds no guard-window join.'})
print(json.dumps({'probe_sha256':p.sha(P),'witnesses':results},indent=2))
