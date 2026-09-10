"""CPU-only controls for new workspace evidence, never native execution."""
import ast
import gzip
import hashlib
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import copy

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('workspace_probe_test',HERE/'probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Tests(unittest.TestCase):
    def test_native_splitter_geometry(self):
        path=Path('/home/bmarti44/.cache/glm53-flash/native-runtime-002/runtime/lib/python3.12/site-packages/vllm/v1/attention/backends/mla/indexer.py')
        node=next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='split_indexer_prefill_chunks')
        scope={'torch':SimpleNamespace(Tensor=object)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        class Number(int):
            def item(self):return int(self)
        for case,lengths in probe.CASES.items():
            for budget in (512,64):
                actual=scope[node.name]([Number(65536)]*len(lengths),[Number(x) for x in lengths],10485760,budget*1024**2)
                starts=[sum(lengths[:i]) for i in range(len(lengths))]
                expected=probe.call_specs(case,budget)
                got=[(starts[r.start]+q.start,starts[r.start]+q.stop,(r.stop-r.start)*65536,q.start>0) for r,q in actual]
                self.assertEqual(got,[(x['start'],x['stop'],x['columns'],x['skip_gather']) for x in expected])

    def test_private_fixture_geometry(self):
        f=probe.fixture_api()
        for seed in (0,42,2**64-1):
            for case,lengths in probe.CASES.items():
                cfg=f.case_config(seed,case)
                self.assertEqual(cfg['lengths'],lengths)
                self.assertEqual(len(cfg['positions']),512)
                self.assertEqual(cfg['ends'],[262144]*len(lengths))
                for start,length in zip(cfg['starts'],lengths):
                    self.assertEqual(int(cfg['positions'][int(start)+length-1]),262143)
        self.assertEqual(f.CASES,probe.CASES)
        self.assertEqual(probe.load_module('unchanged_fixture_check',probe.PRIOR_FIXTURE).CASES['prefill-1'],[2048])

    def test_two_baselines_before_candidate(self):
        schedule=probe.arm_schedule(42)
        self.assertEqual([x['budget_mib'] for x in schedule[:4]],[512]*4)
        self.assertEqual([x['budget_mib'] for x in schedule[4:]],[64]*2)
        for case in probe.CASES:
            self.assertEqual([x['arm'] for x in schedule if x['case']==case],['baseline-a','baseline-b','candidate'])

    def test_ordered_byte_comparison_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.gz',Path(d)/'b.gz'
            a.write_bytes(gzip.compress(b'\x01\x00\x02\x00',mtime=0))
            b.write_bytes(gzip.compress(b'\x01\x00\x02\x00',mtime=9))
            probe.compare_files(a,b,4)
            for bad in (b'\x02\x00\x01\x00',b'\x01\x00',b'\x01\x00\x02\x00\x00'):
                b.write_bytes(gzip.compress(bad))
                with self.assertRaises(ValueError):probe.compare_files(a,b,4)

    def test_budget_is_closed(self):
        for budget in (0,True,63,128,513):
            with self.assertRaises(ValueError):probe.call_specs('prefill-4',budget)
        with self.assertRaises(ValueError):probe.call_specs('decode-4',64)

    def test_baseline_mismatch_aborts_before_candidate(self):
        history=[]
        def execute(arm):history.append(arm);return arm
        def compare(a,b):raise ValueError('ordered baseline bytes differ')
        with self.assertRaisesRegex(ValueError,'baseline'):
            probe.execute_schedule(42,execute,compare)
        self.assertEqual(len(history),2)
        self.assertTrue(all(x['budget_mib']==512 for x in history))

    def test_success_compares_both_baselines_and_candidates(self):
        history=[];pairs=[]
        probe.execute_schedule(1,lambda arm:history.append(arm) or arm,lambda a,b:pairs.append((a,b)))
        self.assertEqual(len(history),6);self.assertEqual(len(pairs),4)
        self.assertTrue(all(x['budget_mib']==512 for x in history[:4]))
        self.assertTrue(all(a['case']==b['case'] for a,b in pairs))

class ReceiptControls(unittest.TestCase):
    def test_memory_cross_counter_mutations(self):
        good={'cuda_allocated':probe.BASE_BYTES+1024,'cuda_reserved':probe.BASE_BYTES+2048,
              'cuda_peak_allocated':probe.BASE_BYTES+2048,'cuda_peak_reserved':probe.BASE_BYTES+4096,
              'device_free':40*1024**3,'device_total':120*1024**3,'workspace_bytes':probe.WORKSPACE}
        probe.validate_memory(good)
        for key,value in [('cuda_allocated',0),('cuda_reserved',1),('cuda_peak_allocated',2**63),
                          ('cuda_peak_reserved',35*1024**3),('workspace_bytes',probe.WORKSPACE-1),
                          ('device_free',float('nan')),('device_total',True)]:
            with self.assertRaises(ValueError):probe.validate_memory({**good,key:value})

    def test_logit_bounds_retain_global_row_order(self):
        for case in probe.CASES:
            cfg=probe.fixture_api().case_config(42,case)
            all_counts=[]
            for budget in (512,64):
                counts=[]
                for i,spec in enumerate(probe.call_specs(case,budget)):
                    lo,hi=probe.bounds(42,case,budget,i)
                    self.assertTrue((lo>=0).all());self.assertTrue((hi<=spec['columns']).all())
                    counts.extend((hi-lo).tolist())
                self.assertEqual(len(counts),512)
                self.assertEqual(counts,((cfg['positions']+1)//4).tolist())
                all_counts.append(counts)
            self.assertEqual(*all_counts)

    def test_complete_raw_receipt_mutations(self):
        case='prefill-4';seed=42;budget=512
        memory={'cuda_allocated':probe.BASE_BYTES+1024,'cuda_reserved':probe.BASE_BYTES+512*1024**2+4096,
                'cuda_peak_allocated':probe.BASE_BYTES+512*1024**2+2048,'cuda_peak_reserved':probe.BASE_BYTES+512*1024**2+4096,
                'device_free':100*1024**3,'device_total':120*1024**3,'workspace_bytes':probe.WORKSPACE}
        rows=[{'time_unix':1,'event':'configured',**probe.geometry(seed,case,budget)},
              {'time_unix':2,'event':'profiled','workspace_bytes':probe.WORKSPACE,'cuda_peak_allocated':memory['cuda_peak_allocated'],'cuda_reserved':memory['cuda_reserved'],'device_total':memory['device_total']},
              {'time_unix':3,'event':'start','case':case},
              {'time_unix':4,'event':'output','case':case,'budget_mib':budget,'metadata':probe.expected_metadata(seed,case,budget),
               'input_digests':{'test':'a'*64},'memory':memory,'cuda_elapsed_ms':1,'cache_stride':[8448,132,1],
               'tail_stride':[1024,512,128,1],'output_alias':True,
               'calls':[{**probe.call_specs(case,budget)[0],'cuda_allocated_at_return':memory['cuda_peak_allocated']}],
               'artifacts':{x:'a'*64 for x in probe.artifact_sizes(seed,case)}}]
        with patch.object(probe,'expected_inputs',return_value={'test':'a'*64}):
            probe.validate_rows(rows,seed,case,budget)
            for mutate in [lambda r:r.pop(),lambda r:r[3].update(calls=[]),lambda r:r[3].update(input_digests={}),
                           lambda r:r[3]['calls'][0].update(cuda_allocated_at_return=2**63),
                           lambda r:r[3]['metadata']['chunks'][0].update(skip_gather=True),
                           lambda r:r[3].update(cuda_elapsed_ms=float('nan')),
                           lambda r:r[3].update(time_unix=2),lambda r:r[0].update(budget_mib=64)]:
                bad=copy.deepcopy(rows);mutate(bad)
                with self.assertRaises(ValueError):probe.validate_rows(bad,seed,case,budget)

    def test_native_entrypoint_is_explicit_frozen_only(self):
        source=(HERE/'probe.py').read_text();tree=ast.parse(source)
        self.assertIn("os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB')",source)
        main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        self.assertIn("Path(__file__).resolve() == frozen / 'code/probe.py'",ast.unparse(main))
        self.assertFalse(any(isinstance(x,(ast.Import,ast.ImportFrom)) and ('torch' in ast.unparse(x)) for x in tree.body))

class BindingControls(unittest.TestCase):
    def test_frozen_binding_missing_duplicate_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            helper=probe.load_module('workspace_review_fixture_for_original_test',HERE/'test_review.py')
            manifest=helper.frozen_fixture(root)
            probe.verify_frozen(root,manifest)
            for change in [lambda m:m['files'].pop(0),lambda m:m['files'].append(m['files'][0]),
                           lambda m:m.update(budgets_mib=[512,128]),lambda m:m['safety'].update(kill_floor_gib=18)]:
                bad=copy.deepcopy(manifest);change(bad)
                with self.assertRaises(ValueError):probe.verify_frozen(root,bad)
            (root/'code/native.py').write_text('changed')
            with self.assertRaises(ValueError):probe.verify_frozen(root,manifest)

    def test_controller_and_child_use_same_binding_validator(self):
        self.assertIn('probe.verify_frozen(out,m)',(HERE/'run.py').read_text())
        self.assertIn('verify_frozen(frozen,m)',(HERE/'probe.py').read_text())

if __name__=='__main__':unittest.main()
