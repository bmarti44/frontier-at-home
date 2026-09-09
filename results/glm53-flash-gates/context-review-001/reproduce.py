# Synthetic scorer mutation only; never an actual model attempt.
import ast,subprocess,tempfile,json,math,re,importlib.util
from pathlib import Path
root=Path('/home/bmarti44/spark-deepseek-v4-flash')
spec=importlib.util.spec_from_file_location('retrieval',root/'scripts/57_dsv4_context_probe.py'); retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
source=subprocess.check_output(['git','show','04bc66ef:scripts/48_probe_glm53_context.py'],cwd=root,text=True)
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in ('read','write','metric','score')]
# Bypass unrelated manifest verification to isolate the exact unmodified scorer.
ns=dict(Path=Path,json=json,math=math,re=re,retrieval=retrieval,verify=lambda p: {})
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<frozen context scorer>','exec',optimize=0),ns)
with tempfile.TemporaryDirectory(prefix='glm53-context-mutation-') as directory:
 out=Path(directory)
 for s in range(4):
  ids=[1]*250128
  records=[{'value':f'RECORD_{name}_{s:016x}'} for name in ('ALPHA','BRAVO','CHARLIE')]
  answer=','.join([x['value'] for x in records]+['NO_EXTRA_RECORD'])
  ns['write'](out/f'{s}-input-token-ids.json',ids);ns['write'](out/f'{s}-fixture.json',{'records':records,'absent_value':'RECORD_DELTA_ffffffffffffffff'})
  def chunk(t,choices,**more):return {'kind':'chunk','monotonic_ns':t,'chunk':dict(id=f'chatcmpl-{s}',choices=choices,**more)}
  rows=[{'kind':'start','monotonic_ns':1},{'kind':'http','monotonic_ns':2,'status':200},chunk(10+s*10,[{'index':0,'delta':{'content':answer},'token_ids':[1]}],prompt_token_ids=ids)]
  if s==0:rows.append(chunk(11,[{'index':0,'delta':{},'finish_reason':'stop'}]))
  rows += [chunk(100+s,[{'index':0,'delta':{},'finish_reason':'stop'}]),chunk(110+s,[],usage={'prompt_tokens':len(ids)}),{'kind':'done','monotonic_ns':120+s},{'kind':'end','monotonic_ns':130+s}]
  (out/f'{s}-raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 metrics='vllm:num_preemptions_total 0\nvllm:num_requests_running 4\nvllm:num_requests_waiting 0\n'
 (out/'metrics.jsonl').write_text(json.dumps({'start_ns':50,'end_ns':51,'text':metrics})+'\n')
 result=ns['score'](out)
 print(json.dumps({'kind':'synthetic mutation, not an actual attempt','source':'04bc66ef','actual_first_terminal_ns':11,'last_first_token_ns':40,'scored_first_terminal_ns':min(r['finish_ns'] for r in result['cases']),'verdict':result['verdict'],'checks':result['checks']},indent=2))
 assert result['verdict']=='FAIL','BUG: non-overlapping terminal chronology and missing metrics baseline accepted'
