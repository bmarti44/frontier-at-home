import json,sys,time,urllib.request,urllib.error
from pathlib import Path
root=Path(sys.argv[1]);base='http://127.0.0.1:8015';key=(root/'api-key').read_text().strip()
try:
 urllib.request.urlopen(base+'/v1/models',timeout=10)
except urllib.error.HTTPError as error:
 if error.code!=401:raise
else:raise RuntimeError('unauthenticated models request accepted')
headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
with urllib.request.urlopen(urllib.request.Request(base+'/v1/models',headers=headers),timeout=30) as response:
 models=json.load(response)
assert any(row['id']=='glm-5.3-flash' for row in models['data'])
(root/'models.json').write_text(json.dumps(models,indent=2)+'\n')
body={'model':'glm-5.3-flash','messages':[{'role':'user','content':'What is 2 + 2? Give the answer in one short sentence.'}],'temperature':0,'max_tokens':256,'stream':True,'stream_options':{'include_usage':True}}
(root/'first-request.json').write_text(json.dumps(body,indent=2)+'\n')
request=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers=headers)
answer='';reasoning='';finished=False;start=time.time()
with urllib.request.urlopen(request,timeout=300) as response,(root/'first-response.jsonl').open('w') as raw:
 for line in response:
  if not line.startswith(b'data: '):continue
  value=line[6:].strip()
  if value==b'[DONE]':finished=True;break
  row=json.loads(value)
  raw.write(json.dumps({'time_unix':time.time(),'chunk':row})+'\n');raw.flush()
  for choice in row.get('choices',[]):
   delta=choice.get('delta',{})
   answer+=delta.get('content') or ''
   reasoning+=delta.get('reasoning_content') or delta.get('reasoning') or ''
summary={'scope':'first authenticated text response only','completed':finished,'answer':answer,'reasoning':reasoning,'authentication_rejection':401,'start_unix':start,'end_unix':time.time()}
(root/'first-response-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
assert finished and answer.strip() and ('4' in answer or 'four' in answer.lower())
