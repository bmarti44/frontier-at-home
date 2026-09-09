"""Reject incomplete native-probe verdicts; synthetic CPU records only."""
import importlib.util
import json
from pathlib import Path
import random
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

if __name__=='__main__':unittest.main()
