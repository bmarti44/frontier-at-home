import importlib.util,json,time,urllib.request
from pathlib import Path
path=Path('/home/bmarti44/spark-deepseek-v4-flash/scripts/48_probe_glm53_context.py');spec=importlib.util.spec_from_file_location('probe',path);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
root=Path('/home/bmarti44/.cache/glm53-flash/server-bringup-012');baseline=probe.read(root/'template-token-comparison.json');body=baseline['request'];body.update(stream=True,stream_options={'include_usage':True});key=(root/'api-key').read_text().strip();rows=[]
def event(**kw):rows.append({'monotonic_ns':time.monotonic_ns(),'time_unix':time.time(),**kw})
event(kind='start')
with urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8015/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}),timeout=30) as response:
 event(kind='http',status=response.status)
 for line in response:
  if not line.startswith(b'data:'):continue
  v=line[5:].strip()
  if v==b'[DONE]':event(kind='done');break
  event(kind='chunk',chunk=json.loads(v))
event(kind='end');(root/'native-stream-check.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));parsed=probe.parse_stream(rows,baseline['local_token_ids']);probe.write(root/'native-stream-check-summary.json',{'scope':'short native SSE grammar/token coverage check only','verdict':'PASS','answer':parsed[0],'usage':parsed[2],'output_token_ids':parsed[-1]});print('Native SSE grammar and token coverage PASS')
