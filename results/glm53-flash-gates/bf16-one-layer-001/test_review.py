"""Synthetic regressions for H1–H3; never native/model acceptance evidence."""
import ast,contextlib,hashlib,importlib.util,json,os,shutil,sys,time,types,unittest
from pathlib import Path
from test_probe import Evidence,p

class Review(Evidence):
    def test_reject_missing_phase_data(self):
        for row in self.rows:
            if row['kind'] in ('weights_on_gpu','forward_complete'):
                for key in list(row):
                    if key not in ('kind','time_unix'):del row[key]
        self.raw()
        with self.assertRaises((ValueError,KeyError)):p.score(self.root)
    def test_reject_contradictory_phase_data(self):
        for row in self.rows:
            if row['kind'] in ('weights_on_gpu','forward_complete'):row.update(cuda_allocated=0,cuda_reserved=0,cuda_peak_allocated=0,elapsed_seconds=-1,host={'available_kib':0,'pswpin':1,'pswpout':1,'used_swap_kib':1})
        self.raw()
        with self.assertRaises((ValueError,KeyError)):p.score(self.root)
    def test_reject_missing_input_digest(self):
        path=self.out/'output.json';d=json.loads(path.read_text());del d['input']['sha256'];path.write_text(json.dumps(d))
        with self.assertRaises((ValueError,KeyError)):p.score(self.root)
    def test_reject_missing_selector(self):
        (self.out/'stock-torch-selector.json').unlink(missing_ok=True)
        with self.assertRaises((ValueError,KeyError,FileNotFoundError)):p.score(self.root)
    def test_reject_premature_completion(self):
        self.rows.insert(0,self.rows.pop());self.raw()
        with self.assertRaises((ValueError,KeyError)):p.score(self.root)

    def test_each_bad_measurement_is_rejected(self):
        import copy
        original=copy.deepcopy(self.rows)
        changes=[('cuda_allocated',0),('cuda_reserved',0),('cuda_peak_allocated',0),('cuda_peak_reserved',0),('cuda_reserved',65*1024**3),('elapsed_seconds',-1),('elapsed_seconds',1e309),('elapsed_seconds',True)]
        for key,value in changes:
            with self.subTest(key=key,value=value):
                self.rows=copy.deepcopy(original);self.rows[-1][key]=value;self.raw()
                with self.assertRaises((ValueError,KeyError)):p.score(self.root)
        for key,value in [('pswpin',1),('pswpout',1),('used_swap_kib',1),('available_kib',0),('available_kib',True)]:
            with self.subTest(key=key):
                self.rows=copy.deepcopy(original);self.rows[0]['host'][key]=value;self.raw()
                with self.assertRaises((ValueError,KeyError)):p.score(self.root)
    def test_changed_selector_rejected(self):
        self.selector['functions'][0]['source']['sha256']='0'*64;self.dump('checks/stock-torch-selector.json',self.selector)
        with self.assertRaisesRegex(ValueError,'selector'):p.score(self.root)
    def test_noncanonical_input_rejected(self):
        d=json.loads((self.out/'output.json').read_text());d['input']['sha256']=self.digest;self.dump('checks/output.json',d)
        with self.assertRaisesRegex(ValueError,'input binding'):p.score(self.root)
    def test_canonical_input_definition(self):
        data=p.canonical_input(42)
        expected=bytearray(hashlib.shake_256(b'glm53-bf16-layer-input-v1\0'+bytes.fromhex('000000000000002a')).digest(16))
        for i in range(0,16,2):expected[i]&=127;expected[i+1]=0x3c|(expected[i+1]&128)
        self.assertEqual(data[:16],expected);self.assertEqual(len(data),16908288)
        self.assertNotEqual(p.sha(data),p.sha(p.canonical_input(43)))

class Controller(unittest.TestCase):
    def execute(self,mutation=None,old_beacon=False,mutation_phase="capture"):
        f=Evidence();f.setUp();O=f.root
        try:
            source=Path(__file__).with_name('run.py');(O/'code/scripts').mkdir(parents=True)
            shutil.copyfile(source.with_name('probe.py'),O/'code/probe.py');(O/'code/scripts/38_guard_glm53_probe.py').write_text('# synthetic fixture\n');(O/'inventory.json').write_text('{}')
            def sha256_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
            def strict_json(path):return p.strict(path.read_bytes())
            def write(path,value):Path(path).write_text(json.dumps(value))
            baseline={'time_unix':1.,'pswpin':0,'pswpout':0,'used_swap_kib':0,'available_kib':115*1024**2}
            m=strict_json(O/'manifest.json');m.update(source_revision='frozen',safety={'minimum_start_gib':110,'kill_floor_gib':40,'timeout_seconds':600,'memory_high_gib':62,'memory_max_gib':64},broad_baseline=baseline,node='/not/executed',python=sys.executable,wrapper='/not/executed',tag='synthetic-fixture',environment={},runtime={'root':str(O),'inventory':str(O/'inventory.json')},frozen_at_unix=1595431051.,files=[{'path':str(x),'size_bytes':x.stat().st_size,'sha256':sha256_file(x)} for x in [O/'code/probe.py',O/'code/scripts/38_guard_glm53_probe.py']]);write(O/'manifest.json',m)
            r=strict_json(O/'randomness.json');r.update(verification='DRAND_BLS_RECEIPT_OK',publication_unix=1595431080,frozen_at_unix=m['frozen_at_unix'],round=1 if old_beacon else 2,signature='fixture',previous_signature='fixture');write(O/'randomness.json',r)
            def mutate(phase):
                if mutation and phase==mutation_phase:
                    doc=strict_json(O/mutation);doc['mutation']='changed-during-capture';write(O/mutation,doc)
            def capture(*args):mutate('capture')
            def score_host(*args):mutate('score');return {'verdict':'PASS'}
            def inventory(*args):mutate('inventory')
            ns=dict(Path=Path,O=O,hashlib=hashlib,importlib=importlib,json=json,math=__import__('math'),os=os,sys=sys,time=time,strict_json=strict_json,sha256_file=sha256_file,require=p.require,write=write,host=lambda:baseline,inference_lock=lambda:contextlib.nullcontext({}),capture_wrapper=capture,score_host_observations=score_host,verify_inventory=inventory,subprocess=types.SimpleNamespace(run=lambda *a,**k:types.SimpleNamespace(returncode=0,stdout='DRAND_BLS_RECEIPT_OK\n',stderr='')))
            tree=ast.parse(source.read_text());selected=[]
            for node in tree.body:
                if isinstance(node,ast.FunctionDef) and node.name!='host':selected.append(node)
                elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('CONTROL','m','failure','host_result','inner','started','bindings') for t in node.targets):selected.append(node)
                elif isinstance(node,ast.With):selected.append(node)
            exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),ns)
            return strict_json(O/'summary.json')
        finally:f.tearDown()
    def test_valid_outer_control(self):self.assertEqual(self.execute()['verdict'],'PASS')
    def test_preknown_beacon_rejected(self):self.assertEqual(self.execute(old_beacon=True)['verdict'],'FAIL')
    def test_manifest_mutation_rejected(self):self.assertEqual(self.execute('manifest.json')['verdict'],'FAIL')
    def test_randomness_mutation_rejected(self):self.assertEqual(self.execute('randomness.json')['verdict'],'FAIL')
    def test_changes_through_postinventory_rejected(self):
        for phase in ['score','inventory']:
            for name in ['manifest.json','randomness.json']:
                with self.subTest(phase=phase,name=name):self.assertEqual(self.execute(name,mutation_phase=phase)['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
