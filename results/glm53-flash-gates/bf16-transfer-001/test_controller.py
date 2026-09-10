"""Synthetic caller-boundary checks; closed capture/guard implementations are reused."""
import ast,contextlib,hashlib,importlib.util,json,math,os,shutil,sys,tempfile,time,types,unittest
from pathlib import Path
from test_probe import Evidence,p
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/lib'))
from glm53_contract import strict_json,sha256_file
class Controller(unittest.TestCase):
    def execute(self,mutation=None,capture_error=False):
        source=Path(__file__).with_name('run.py');calls=[]
        with tempfile.TemporaryDirectory() as d:
            O=Path(d);(O/'code/scripts').mkdir(parents=True);shutil.copyfile(source.with_name('probe.py'),O/'code/probe.py');(O/'code/scripts/38_guard_glm53_probe.py').write_text('# synthetic not executed\n')
            def write(path,value):path.write_text(json.dumps(value))
            host={'time_unix':1.,'available_kib':115*1024**2,'pswpin':0,'pswpout':0,'used_swap_kib':0}
            m={'source_revision':'synthetic','frozen_at_unix':1595431051.,'broad_baseline':host,'safety':{'minimum_start_gib':110,'kill_floor_gib':64,'timeout_seconds':180,'memory_high_gib':32,'memory_max_gib':34},'node':'/not/executed','python':sys.executable,'wrapper':'/not/executed','tag':'synthetic','environment':{},'files':[{'path':str(x),'size_bytes':x.stat().st_size,'sha256':sha256_file(x)} for x in [O/'code/probe.py',O/'code/scripts/38_guard_glm53_probe.py']]};write(O/'manifest.json',m)
            write(O/'randomness.json',{'frozen_at_unix':m['frozen_at_unix'],'round':2,'publication_unix':1595431080,'verification':'DRAND_BLS_RECEIPT_OK','seed':0,'randomness':'0'*64,'signature':'fixture','previous_signature':'fixture'})
            (O/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in Evidence().rows()))
            def capture(*args):
                calls.append(args[-1]);self.assertEqual(args[-2]['GLM_SAFE_MEMORY_HIGH_GIB'],'32');self.assertEqual(args[-2]['GLM_SAFE_KILL_FLOOR_GIB'],'64')
                if mutation:
                    x=json.loads((O/mutation).read_text());x['changed']=True;write(O/mutation,x)
                if capture_error:raise ValueError('closed capture failure after cleanup')
            ns=dict(Path=Path,O=O,hashlib=hashlib,importlib=importlib,json=json,math=math,os=os,sys=sys,time=time,strict_json=strict_json,sha256_file=sha256_file,require=p.require,write=write,host=lambda:host,inference_lock=lambda:contextlib.nullcontext({}),capture_wrapper=capture,score_host_observations=lambda *args:{'verdict':'PASS'},subprocess=types.SimpleNamespace(run=lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='DRAND_BLS_RECEIPT_OK\n',stderr='')))
            body=[]
            for node in ast.parse(source.read_text()).body:
                if isinstance(node,ast.FunctionDef) and node.name!='host':body.append(node)
                elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CONTROL','m','failure','host_result','inner','started','bindings') for t in node.targets):body.append(node)
                elif isinstance(node,ast.With):body.append(node)
            exec(compile(ast.Module(body=body,type_ignores=[]),str(source),'exec'),ns)
            self.assertEqual(calls,[180]);return json.loads((O/'summary.json').read_text())
    def test_reused_capture_configuration(self):self.assertEqual(self.execute()['verdict'],'PASS')
    def test_capture_failure_preserved(self):self.assertEqual(self.execute(capture_error=True)['verdict'],'FAIL')
    def test_changed_launch_documents_rejected(self):
        for name in ['manifest.json','randomness.json']:
            with self.subTest(name=name):self.assertEqual(self.execute(mutation=name)['verdict'],'FAIL')
if __name__=='__main__':unittest.main()
