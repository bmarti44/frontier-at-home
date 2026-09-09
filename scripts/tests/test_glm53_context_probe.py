"""Synthetic scorer mutations only; none of these rows are model evidence."""
import importlib.util,json,subprocess,sys,tempfile,tracemalloc,unittest
from pathlib import Path
from unittest.mock import patch
PATH=Path(__file__).resolve().parents[1]/'48_probe_glm53_context.py'
spec=importlib.util.spec_from_file_location('context_probe',PATH);probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
class ContextEvidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
  launch={'arguments':['--max-model-len','262144','--max-num-seqs','4','--max-num-batched-tokens','128','--long-prefill-token-threshold','32','--no-enable-prefix-caching']};probe.write(self.out/'server-launch.json',launch)
  for slot in range(4):
   ids=[1]*250128;records=[{'value':f'RECORD_{name}_{slot:016x}'} for name in ('ALPHA','BRAVO','CHARLIE')];answer=','.join([r['value'] for r in records]+['NO_EXTRA_RECORD'])
   probe.write(self.out/f'{slot}-input-token-ids.json',ids);probe.write(self.out/f'{slot}-fixture.json',{'records':records,'absent_value':'RECORD_DELTA_absent'})
   def c(t,choices,**kw):return {'kind':'chunk','monotonic_ns':t,'chunk':{'id':f'case-{slot}','choices':choices,**kw}}
   rows=[{'kind':'start','monotonic_ns':100},{'kind':'http','monotonic_ns':101,'status':200},c(200+slot*10,[{'index':0,'delta':{'content':answer},'token_ids':[1]}],prompt_token_ids=ids),c(1000+slot,[{'index':0,'delta':{},'finish_reason':'stop'}]),c(1100+slot,[],usage={'prompt_tokens':len(ids),'completion_tokens':1}),{'kind':'done','monotonic_ns':1200+slot},{'kind':'end','monotonic_ns':1300+slot}]
   self.save(slot,rows)
  samples=[{'start_ns':a,'end_ns':b,'text':f'vllm:num_preemptions_total 0\nvllm:num_requests_running {running}\nvllm:num_requests_waiting 0\n'} for a,b,running in ((10,20,0),(500,510,4),(2000,2010,0))]
  (self.out/'metrics.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples))
 def rows(self,slot=0):return [json.loads(s) for s in (self.out/f'{slot}-raw.jsonl').read_text().splitlines()]
 def save(self,slot,rows):(self.out/f'{slot}-raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
 def score(self):
  # Isolate parser/scorer from hash binding; test closed binding separately.
  with patch.object(probe,'verify',return_value={}):return probe.score(self.out)
 def reject(self):
  try:result=self.score()
  except (AssertionError,KeyError,ValueError):return
  self.assertEqual(result['verdict'],'FAIL')
 def test_valid_synthetic_control(self):self.assertEqual(self.score()['verdict'],'PASS')
 def test_metrics_scoring_memory_does_not_scale_with_log_length(self):
  def peak():
   tracemalloc.start()
   try:
    self.assertEqual(self.score()['verdict'],'PASS')
    return tracemalloc.get_traced_memory()[1]
   finally:tracemalloc.stop()
  baseline=peak();p=self.out/'metrics.jsonl'
  def sample(a,b,running,padding=''):
   return json.dumps({'start_ns':a,'end_ns':b,'text':f'vllm:num_preemptions_total 0\nvllm:num_requests_running {running}\nvllm:num_requests_waiting 0\n'+padding})+'\n'
  with p.open('w') as f:
   f.write(sample(10,20,0))
   for i in range(1000):f.write(sample(500+2*i,501+2*i,4,'#'+('x'*16384)+'\n'))
   f.write(sample(4000,4010,0))
  self.assertLess(peak()-baseline,5*1024*1024,'scoring retained a growing metrics log in memory')
 def test_invalid_trailing_metrics_cannot_be_ignored(self):
  with (self.out/'metrics.jsonl').open('a') as f:f.write('{"error":"late malformed sample"}\n')
  self.reject()
 def test_repeated_terminal_cannot_manufacture_overlap(self):
  rows=self.rows();r=json.loads(json.dumps(rows[3]));r['monotonic_ns']=211;rows.insert(3,r);self.save(0,rows);self.reject()
 def test_missing_metrics_baseline_and_end(self):
  p=self.out/'metrics.jsonl';p.write_text(p.read_text().splitlines()[1]+'\n');self.reject()
 def test_metrics_error(self):
  p=self.out/'metrics.jsonl';rows=[json.loads(r) for r in p.read_text().splitlines()];rows[1].pop('text');rows[1]['error']='timeout';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));self.reject()
 def test_preemption(self):
  p=self.out/'metrics.jsonl';rows=p.read_text().splitlines();rows[-1]=rows[-1].replace('preemptions_total 0','preemptions_total 1');p.write_text('\n'.join(rows)+'\n');self.reject()
 def test_wrong_choice_index(self):
  rows=self.rows();rows[2]['chunk']['choices'][0]['index']=1;self.save(0,rows);self.reject()
 def test_duplicate_response_id(self):
  rows=self.rows(1)
  for r in rows:
   if r['kind']=='chunk':r['chunk']['id']='case-0'
  self.save(1,rows);self.reject()
 def test_missing_done(self):
  rows=self.rows();rows.pop(-2);self.save(0,rows);self.reject()
 def test_duplicate_timestamp(self):
  rows=self.rows();rows[2]['monotonic_ns']=rows[1]['monotonic_ns'];self.save(0,rows);self.reject()
 def test_nonfinite_timestamp(self):
  rows=self.rows();rows[2]['monotonic_ns']=float('nan');self.save(0,rows);self.reject()
 def test_output_token_coverage(self):
  rows=self.rows();rows[4]['chunk']['usage']['completion_tokens']=2;self.save(0,rows);self.reject()
 def test_wrong_launch(self):
  probe.write(self.out/'server-launch.json',{'arguments':[]});self.reject()
 def test_nonfinite_unused_field(self):
  rows=self.rows();rows[2]['invalid_extra']=float('nan');self.save(0,rows);self.reject()
 def test_optimized_python_rejected(self):
  for flag in ('-O','-OO'):
   result=subprocess.run([sys.executable,'-I','-B',flag,str(PATH),'--help'],capture_output=True,text=True)
   self.assertNotEqual(result.returncode,0)
   self.assertIn('optimized Python is forbidden',result.stderr)
 def test_empty_binding_rejected(self):
  probe.write(self.out/'manifest.json',{'files':{},'sources':{}})
  with self.assertRaises(AssertionError):probe.verify(self.out)
if __name__=='__main__':unittest.main()
