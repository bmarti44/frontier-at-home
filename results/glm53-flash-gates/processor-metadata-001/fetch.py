import hashlib,json,pathlib,time,urllib.request
base=pathlib.Path('/home/bmarti44/.cache/glm53-flash')
api_path=base/'metadata/k2-api.json'
api=json.loads(api_path.read_bytes())
output=base/'processor-metadata-001'
output.mkdir(exist_ok=False)
entries={r['rfilename']:r for r in api['siblings']}
files=['processor_config.json','tokenizer_config.json','tokenizer.json','chat_template.jinja','generation_config.json','config.json']
raw=[]
for name in files:
 entry=entries[name]
 url=f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{name}"
 started=time.time()
 with urllib.request.urlopen(url,timeout=60) as response:
  payload=response.read(entry['size']+1)
 (output/name).write_bytes(payload)
 blob=hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()
 if len(payload)!=entry['size'] or blob!=entry['blobId']:
  raise ValueError('metadata source hash/size mismatch: '+name)
 raw.append({'path':name,'url':url,'size_bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'git_blob':blob,'start_unix':started,'end_unix':time.time(),'source_verified':True})
 (output/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in raw))
manifest={'schema_version':1,'source_revision':api['sha'],'repository':api['id'],'api':{'path':str(api_path),'sha256':hashlib.sha256(api_path.read_bytes()).hexdigest()},'files':raw}
(output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(output/'summary.json').write_text(json.dumps({'verdict':'PASS','qualification':'processor_tokenizer_metadata_only','files':len(raw),'weight_payload_downloaded':False,'model_loaded':False},indent=2)+'\n')
print((output/'processor_config.json').read_text())
