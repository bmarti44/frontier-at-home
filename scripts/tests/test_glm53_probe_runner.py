"""Reject incomplete native-probe verdicts; synthetic CPU records only."""
import importlib.util
import json
from pathlib import Path
import random
import os
import shutil
from unittest import mock
from types import SimpleNamespace
import tempfile
import unittest
SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
runner = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(runner)

class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name); (self.root/'checks').mkdir()
        self.seed=123
        self.binding={'scorer_sha256':'a'*64, 'binary_sha256':'b'*64, 'expected_checks':14}
        order=runner.native_ids(); random.Random(self.seed).shuffle(order)
        self.rows=[{'check_order':order}]
        for name in order:
            self.rows.extend([{'event':'start','check_id':name}, {'event':'pass','check_id':name}])
        (self.root/'checks/assertion-output.log').write_text('')
        self.seal()

    def seal(self):
        p=self.root/'checks';(p/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (p/'summary.json').write_text(json.dumps({'verdict':'PASS','qualification':'synthetic_native_smoke_only','model_loaded':False,'checks_completed':14,
            'raw_sha256':runner.sha256_file(p/'raw.jsonl'),'test_output_sha256':runner.sha256_file(p/'assertion-output.log')}))
        (p/'manifest.json').write_text(json.dumps({'seed':self.seed,**self.binding}))

    def test_complete_native_record_accepts(self):
        self.assertEqual(runner.score_inner(self.root,'native',self.seed,self.binding)['checks_completed'],14)

    def test_missing_duplicate_reordered_or_failed_native_records_reject(self):
        original=list(self.rows)
        variants=[original[:-1], original+[original[-1]], original[:1]+list(reversed(original[1:])), original+[{'event':'failure'}]]
        for rows in variants:
            self.rows=rows;self.seal()
            with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

    def test_stale_interpreter_or_probe_binding_rejects(self):
        p=self.root/'checks/manifest.json'; original=json.loads(p.read_text())
        for key in ('scorer_sha256','binary_sha256'):
            changed={**original,key:'c'*64};p.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

    def test_seed_and_unbound_raw_reject(self):
        with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed+1,self.binding)
        with (self.root/'checks/raw.jsonl').open('a') as f:f.write('{}\n')
        with self.assertRaises(ValueError):runner.score_inner(self.root,'native',self.seed,self.binding)

class CacheRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'checks').mkdir();self.seed=123
        self.binding={'scorer_sha256':'a'*64,'binary_sha256':'b'*64}
        order=[f'cache-preflight-{i}' for i in range(5)];random.Random(self.seed).shuffle(order)
        self.rows=[{'event':'allocated_and_zeroed','unique_backing_bytes':9565304320}]
        for i,name in enumerate(order[:4]):
            ids=list(range(i*36+1,(i+1)*36+1))
            self.rows.append({'event':'reserve','request_id':name,'block_ids':[ids[:31],*[ids[j:j+1] for j in range(31,36)]],'free_blocks':144-36*(i+1)})
        self.rows += [{'event':'fifth_rejected','live_requests':4,'distinct_blocks':144},{'event':'restored','free_blocks':144}]
        self.seal()
    def seal(self):
        p=self.root/'checks';(p/'raw.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        (p/'manifest.json').write_text(json.dumps({'seed':self.seed,**self.binding}))
        (p/'summary.json').write_text(json.dumps({'verdict':'PASS','qualification':'model_free_cache_allocation_only','actual_input_tokens_processed':0,'model_loaded':False,'raw_sha256':runner.sha256_file(p/'raw.jsonl')}))
    def test_cache_reservations_must_match_frozen_physical_contract(self):
        original=json.loads(json.dumps(self.rows))
        changes=[('request_id','duplicate'),('block_ids',[[1001]]),('free_blocks',999)]
        for key,value in changes:
            self.rows=json.loads(json.dumps(original));self.rows[1][key]=value;self.seal()
            with self.assertRaises(ValueError):runner.score_inner(self.root,'cache',self.seed,self.binding)
    def test_cache_block_reuse_and_seed_order_reject(self):
        self.rows[2]['block_ids']=self.rows[1]['block_ids'];self.seal()
        with self.assertRaises(ValueError):runner.score_inner(self.root,'cache',self.seed,self.binding)

class LaunchBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def test_later_beacon_cannot_replace_predetermined_round(self):
        source=Path(__file__).resolve().parents[2]/'results/glm53-flash-gates/native-smoke-003/randomness.json'
        receipt=json.loads(source.read_text());(self.root/'randomness.json').write_text(json.dumps(receipt))
        manifest={'frozen_at_unix':receipt['publication_unix']-45,'node':'/pinned/node','tools':{'/pinned/node':{}},'code':{'scripts/103_verify_drand_receipt_bundle.mjs':{}}}
        with mock.patch.object(runner.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='DRAND_BLS_RECEIPT_OK\n',stderr='')):
            with self.assertRaises(ValueError):runner.verify_beacon(self.root,manifest)
    def test_accepted_input_files_must_stay_byte_identical(self):
        for name in ('manifest.json','randomness.json'):(self.root/name).write_text('{}\n')
        accepted={name:runner.sha256_file(self.root/name) for name in ('manifest.json','randomness.json')}
        (self.root/'manifest.json').write_text('{"changed":true}\n')
        with self.assertRaises(ValueError):runner.verify_accepted_inputs(self.root,accepted)
    def test_outer_shell_environment_excludes_ambient_startup_hooks(self):
        with mock.patch.dict(os.environ,{'BASH_ENV':'/bad','LD_PRELOAD':'/bad','PATH':'/bad','PYTHONPATH':'/bad'}):
            environment=runner.wrapper_environment({'GLM_SAFE_PARENT_LOCK_PID':'123'})
        self.assertEqual(environment['PATH'],'/usr/bin:/bin')
        self.assertTrue({'BASH_ENV','LD_PRELOAD','PYTHONPATH'}.isdisjoint(environment))

if __name__=='__main__':unittest.main()
