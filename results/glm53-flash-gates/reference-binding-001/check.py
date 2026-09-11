import hashlib,json,struct,time,urllib.request
from pathlib import Path
BASE=Path('/home/bmarti44/.cache/glm53-flash');OUT=BASE/'reference-binding-001';META=BASE/'reference-audit-001';MODEL=BASE/'model-weights-001'
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_exact(stream,n):
 data=bytearray()
 while len(data)<n:
  block=stream.read(n-len(data))
  if not block:raise ValueError('short read')
  data.extend(block)
 return bytes(data)
def header(f):
 prefix=read_exact(f,8);n=struct.unpack('<Q',prefix)[0]
 if not 0<n<=1048576:raise ValueError('invalid header length')
 raw=read_exact(f,n);return prefix+raw,json.loads(raw)
api=json.loads((META/'malaiwah-api.json').read_text());revision=api['sha'];assert revision=='6a6cea7adb38b5ff5d38979cbd4334b89fa4e069'
references={r['rfilename']:r for r in api['siblings']};suite=json.loads((META/'malaiwah/suite/suite-manifest.json').read_text());index=json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
results=[]
with (OUT/'raw.jsonl').open('w') as raw:
 def emit(event):raw.write(json.dumps({'time_unix':time.time(),**event})+'\n');raw.flush()
 tokenizer=sha(MODEL/'tokenizer.json');assert tokenizer==suite['model_identity']['tokenizer_sha256'];emit({'event':'tokenizer_match','sha256':tokenizer})
 for remote_name,local_key in [('head/head.safetensors','lm_head.weight'),('head/final_norm.safetensors','model.language_model.norm.weight')]:
  expected=references[remote_name];local=MODEL/index[local_key];before=local.stat();assert not local.is_symlink()
  url=f'https://huggingface.co/datasets/malaiwah/GLM-5.3-Flash-fidelity-suite-v1/resolve/{revision}/{remote_name}?download=true&binding={time.time_ns()}'
  full_hash=hashlib.sha256();remote_hash=hashlib.sha256();local_hash=hashlib.sha256();equal=True;total=0;first_difference=None
  with local.open('rb') as target,urllib.request.urlopen(url,timeout=90) as source:
   local_header,local_entries=header(target);entry=local_entries[local_key]
   remote_header,remote_entries=header(source);full_hash.update(remote_header)
   tensors=[(k,v) for k,v in remote_entries.items() if k!='__metadata__'];assert len(tensors)==1
   remote_key,ref=tensors[0];assert entry['shape']==ref['shape'] and entry['dtype']==ref['dtype']=='BF16'
   size=ref['data_offsets'][1];assert ref['data_offsets'][0]==0 and size==entry['data_offsets'][1]-entry['data_offsets'][0]
   assert expected['size']==len(remote_header)+size
   target.seek(len(local_header)+entry['data_offsets'][0]);emit({'event':'start_tensor','name':local_key,'size_bytes':size,'reference_file':remote_name})
   while total<size:
    count=min(8*1024*1024,size-total);a=read_exact(source,count);b=read_exact(target,count);full_hash.update(a);remote_hash.update(a);local_hash.update(b)
    if a!=b:
     equal=False
     if first_difference is None:first_difference=total+next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    total+=count
    if total%(64*1024*1024)==0:emit({'event':'progress','name':local_key,'bytes_compared':total})
   assert not source.read(1),'trailing source bytes'
  after=local.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
  assert full_hash.hexdigest()==expected['lfs']['sha256'],'reference full-file hash mismatch'
  row={'name':local_key,'reference_key':remote_key,'size_bytes':total,'byte_identical':equal,'first_difference':first_difference,'reference_file':{'sha256':full_hash.hexdigest()},'reference_tensor':{'sha256':remote_hash.hexdigest()},'candidate_tensor':{'sha256':local_hash.hexdigest()}};results.append(row);emit({'event':'tensor_result',**row});print(json.dumps(row),flush=True)
summary={'verdict':'PASS' if all(r['byte_identical'] for r in results) else 'FAIL','scope':'tokenizer and BF16 output-head/final-norm byte binding only; not native reference equivalence or fidelity qualification','tokenizer':{'sha256':tokenizer},'tensors':results}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(OUT/'manifest.json').write_text(json.dumps({'source':{'sha256':sha(OUT/'check.py')},'reference_repo':api['id'],'revision':revision,'reference_api':{'sha256':sha(META/'malaiwah-api.json')},'candidate_inventory':{'sha256':sha(MODEL/'inventory.json')},'raw':{'sha256':sha(OUT/'raw.jsonl')}},indent=2)+'\n')
print(json.dumps({'verdict':summary['verdict']}),flush=True)
