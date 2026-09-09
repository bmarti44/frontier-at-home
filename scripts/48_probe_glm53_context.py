#!/usr/bin/env python3
"""Direct four-slot GLM context probe using existing retrieval fixtures/scorer."""
import argparse,concurrent.futures,hashlib,importlib.util,json,math,re,threading,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
MODEL=Path('/home/bmarti44/.cache/glm53-flash/model-weights-001')
DS=ROOT/'scripts/57_dsv4_context_probe.py'
BINDING=ROOT/'results/glm53-flash-gates/reference-binding-001/summary.json'
spec=importlib.util.spec_from_file_location('retrieval',DS);retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text(),parse_constant=lambda v:(_ for _ in ()).throw(ValueError(v)))
def prepare(out,seed,server):
 from jinja2.sandbox import ImmutableSandboxedEnvironment
 from tokenizers import Tokenizer
 assert re.fullmatch('[0-9a-f]{64}',seed)
 launch=launch_check(server)
 out.mkdir(exist_ok=False,parents=True);write(out/'server-launch.json',launch)
 tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
 assert sha(MODEL/'tokenizer.json')==read(BINDING)['tokenizer']['sha256']
 template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols']).from_string((MODEL/'chat_template.jinja').read_text())
 inputs=[]
 for slot in range(4):
  case_seed=hashlib.sha256(f'{seed}:slot:{slot}'.encode()).hexdigest();target=250000
  for _ in range(8):
   art=retrieval.build_request_artifacts(tokenizer,target=target,seed_sha256=case_seed)
   body=art['payload'];body.update(model='glm-5.3-flash',return_token_ids=True,chat_template_kwargs={'reasoning_effort':'low','clear_thinking':True})
   rendered=template.render(messages=body['messages'],tools=[],add_generation_prompt=True,reasoning_effort='low',clear_thinking=True)
   ids=tokenizer.encode(rendered,add_special_tokens=False).ids
   if len(ids)==250128:break
   target+=250128-len(ids)
  else:raise RuntimeError('exact prompt length did not converge')
  meta={k:v for k,v in art['fixture'].items() if k!='text'};meta.update(slot=slot,seed=case_seed,input_tokens=len(ids))
  for name,data in [('request',body),('fixture',meta),('input-token-ids',ids)]:write(out/f'{slot}-{name}.json',data)
  inputs.append({'slot':slot,'input_tokens':len(ids)})
 files=[out/'server-launch.json']+[out/f'{s}-{name}.json' for s in range(4) for name in ('request','fixture','input-token-ids')]
 manifest={'scope':'direct aggregate-context probe; production qualification requires complete freeze/host review','prepared_unix':time.time(),'seed':seed,'inputs':inputs,'files':{p.name:sha(p) for p in files},'sources':{str(p):sha(p) for p in [Path(__file__),DS,BINDING,MODEL/'tokenizer.json',MODEL/'chat_template.jinja',ROOT/'fixtures/ctx-32k.txt']}}
 write(out/'manifest.json',manifest);print(json.dumps(inputs),flush=True)
def verify(out):
 m=read(out/'manifest.json')
 assert set(m['files'])=={f'{slot}-{name}.json' for slot in range(4) for name in ('request','fixture','input-token-ids')}|{'server-launch.json'}
 assert set(m['sources'])=={str(p) for p in (Path(__file__),DS,BINDING,MODEL/'tokenizer.json',MODEL/'chat_template.jinja',ROOT/'fixtures/ctx-32k.txt')}
 for p,h in m['files'].items():assert sha(out/p)==h,p
 for p,h in m['sources'].items():assert sha(p)==h,p
 return m
def check_launch(launch):
 argv=launch['arguments']
 for flag,value in [('--max-model-len','262144'),('--max-num-seqs','4'),('--max-num-batched-tokens','128'),('--long-prefill-token-threshold','32')]:
  assert flag in argv and argv[argv.index(flag)+1]==value,(flag,value)
 assert '--no-enable-prefix-caching' in argv
 return launch
def launch_check(server):return check_launch(read(server/'launch.json'))
def run(out,server):
 verify(out);launch=launch_check(server);assert launch==read(out/'server-launch.json')
 key=(server/'api-key').read_text().strip();headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
 barrier=threading.Barrier(4);stop=threading.Event();ready=threading.Event()
 def request(path,body=None,timeout=30):return urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8015'+path,data=None if body is None else json.dumps(body,separators=(',',':')).encode(),headers=headers),timeout=timeout)
 def metric_sample(f):
  start=time.monotonic_ns()
  try:
   with request('/metrics') as response:text=response.read().decode()
   row={'start_ns':start,'end_ns':time.monotonic_ns(),'text':text}
  except Exception as error:row={'start_ns':start,'end_ns':time.monotonic_ns(),'error':str(error)}
  f.write(json.dumps(row)+'\n');f.flush()
 def metrics():
  with (out/'metrics.jsonl').open('x') as f:
   metric_sample(f);ready.set()
   while not stop.wait(.25):metric_sample(f)
   metric_sample(f)
 def stream(slot):
  body=read(out/f'{slot}-request.json');barrier.wait(timeout=60)
  with (out/f'{slot}-raw.jsonl').open('x') as raw:
   def event(**data):raw.write(json.dumps({'monotonic_ns':time.monotonic_ns(),'time_unix':time.time(),**data})+'\n');raw.flush()
   event(kind='start')
   try:
    with request('/v1/chat/completions',body,timeout=8500) as response:
     event(kind='http',status=response.status)
     for line in response:
      if not line.startswith(b'data:'):continue
      value=line[5:].strip()
      if value==b'[DONE]':event(kind='done');break
      event(kind='chunk',chunk=json.loads(value))
   except Exception as error:event(kind='error',error=str(error),body=error.read().decode() if hasattr(error,'read') else None)
   event(kind='end')
 monitor=threading.Thread(target=metrics);monitor.start()
 try:
  assert ready.wait(35),'metrics baseline timed out'
  baseline=json.loads((out/'metrics.jsonl').read_text().splitlines()[0])
  assert metric(baseline['text'],'num_requests_running')==0 and metric(baseline['text'],'num_requests_waiting')==0
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(stream,range(4)))
 finally:stop.set();monitor.join(timeout=35)
 verify(out)
 print(json.dumps(score(out)),flush=True)
def metric(text,name):
 values=[float(v) for v in re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})? ([^\n]+)$',text,re.M)]
 assert len(values)==1 and math.isfinite(values[0]),(name,values)
 return values[0]
def parse_stream(rows,ids):
 assert rows and rows[0]['kind']=='start' and rows[-1]['kind']=='end'
 times=[r['monotonic_ns'] for r in rows]
 assert all(type(t) is int and t>0 for t in times) and all(a<b for a,b in zip(times,times[1:]))
 assert all(type(t) is int and t>=0 for t in ids)
 state='http';content='';reasoning='';usage=None;finish=None;observed=None;first=None;terminal=None;response_id=None;outputs=[]
 for r in rows[1:-1]:
  if state=='http':
   assert r['kind']=='http' and r['status']==200
   state='stream';continue
  if r['kind']=='done':
   assert state=='usage';state='done';continue
  assert r['kind']=='chunk' and state in ('stream','terminal'),(state,r['kind'])
  c=r['chunk'];rid=c.get('id');assert isinstance(rid,str) and rid
  if response_id is None:response_id=rid
  assert response_id==rid
  if c.get('prompt_token_ids') is not None:
   assert observed is None and state=='stream' and first is None
   observed=c['prompt_token_ids'];assert all(type(t) is int and t>=0 for t in observed)
  choices=c.get('choices');assert isinstance(choices,list)
  if state=='terminal':
   assert choices==[] and isinstance(c.get('usage'),dict)
   usage=c['usage'];state='usage';continue
  assert len(choices)==1 and type(choices[0].get('index')) is int and choices[0]['index']==0
  assert c.get('usage') is None
  choice=choices[0];delta=choice.get('delta');assert isinstance(delta,dict)
  for name in ('content','reasoning','reasoning_content'):assert delta.get(name) is None or isinstance(delta[name],str)
  content+=delta.get('content') or '';reasoning+=delta.get('reasoning') or delta.get('reasoning_content') or ''
  tokens=choice.get('token_ids') or [];assert isinstance(tokens,list) and all(type(t) is int and t>=0 for t in tokens)
  if tokens:
   outputs.extend(tokens)
   if first is None:first=r['monotonic_ns']
  if choice.get('finish_reason') is not None:
   finish=choice['finish_reason'];terminal=r['monotonic_ns'];state='terminal'
 assert state=='done' and first is not None and terminal is not None
 assert observed==ids
 assert all(type(usage.get(k)) is int for k in ('prompt_tokens','completion_tokens'))
 assert usage['prompt_tokens']==len(ids) and usage['completion_tokens']==len(outputs)
 return content,reasoning,usage,finish,first,terminal,response_id,outputs

def score(out):
 verify(out);check_launch(read(out/'server-launch.json'));results=[];firsts=[];finishes=[];starts=[];ends=[];response_ids=[]
 for slot in range(4):
  rows=[json.loads(line) for line in (out/f'{slot}-raw.jsonl').read_text().splitlines()]
  ids=read(out/f'{slot}-input-token-ids.json');fixture=read(out/f'{slot}-fixture.json')
  content,reasoning,usage,finish,first,terminal,rid,outputs=parse_stream(rows,ids)
  starts.append(rows[0]['monotonic_ns']);ends.append(rows[-1]['monotonic_ns']);response_ids.append(rid)
  completion=retrieval.validate_completion(content=content,reasoning_content=reasoning,finish_reason=finish,done=True,records=fixture['records'],absent_value=fixture['absent_value'])
  checks={'completion':completion['pass'],'stream_and_tokens_valid':True,'direct_size':len(ids)==250128}
  results.append({'slot':slot,'checks':checks,'content':content,'reasoning':reasoning,'usage':usage,'first_ns':first,'finish_ns':terminal,'output_tokens_observed':len(outputs),'retrieval':completion})
  if first is not None:firsts.append(first)
  if terminal is not None:finishes.append(terminal)
 samples=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
 overlap=len(firsts)==len(finishes)==4 and max(firsts)<min(finishes)
 try:
  assert len(samples)>=3
  assert all(type(r.get(k)) is int and r[k]>0 for r in samples for k in ('start_ns','end_ns'))
  assert all(r['start_ns']<r['end_ns'] for r in samples)
  assert all(a['end_ns']<b['start_ns'] for a,b in zip(samples,samples[1:]))
  assert samples[0]['end_ns']<min(starts) and samples[-1]['start_ns']>max(ends)
  assert all(metric(r['text'],name)==0 for r in (samples[0],samples[-1]) for name in ('num_requests_running','num_requests_waiting'))
  preempt=[metric(r['text'],'num_preemptions_total') for r in samples]
  no_preemption=bool(preempt) and max(preempt)==min(preempt)
  common=overlap and any(r['start_ns']>=max(firsts) and r['end_ns']<=min(finishes) and metric(r['text'],'num_requests_running')==4 and metric(r['text'],'num_requests_waiting')==0 for r in samples)
 except (KeyError,AssertionError,ValueError):no_preemption=False;common=False
 checks={'four_valid_cases':len(results)==4 and len(set(response_ids))==4 and all(all(r['checks'].values()) for r in results),'generated_streams_overlap':overlap,'metrics_corrobates_four_running':common,'no_preemption':no_preemption,'aggregate_input_tokens':sum(len(read(out/f'{s}-input-token-ids.json')) for s in range(4))>=1000000}
 summary={'scope':'direct aggregate context API checks; host/freeze/lifecycle verdict remains separate','verdict':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'cases':results,'full_qualification':'NO_RESULT until host/freeze/lifecycle checks and review'}
 write(out/'summary.json',summary);return summary
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','score','launch-check']);p.add_argument('--output',type=Path);p.add_argument('--seed');p.add_argument('--server',type=Path);a=p.parse_args()
 if a.action=='prepare':prepare(a.output,a.seed,a.server)
 elif a.action=='run':run(a.output,a.server)
 elif a.action=='launch-check':launch_check(a.server);print('launch configuration PASS')
 else:print(json.dumps(score(a.output)))
