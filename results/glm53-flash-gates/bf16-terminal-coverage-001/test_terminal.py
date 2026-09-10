"""Synthetic teardown-boundary regressions; never model qualification evidence."""
import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path('/home/bmarti44/spark-deepseek-v4-flash')
spec = importlib.util.spec_from_file_location('closed_host_tests', ROOT/'scripts/tests/test_glm53_host_evidence.py')
closed = importlib.util.module_from_spec(spec); spec.loader.exec_module(closed)

class TerminalCoverageTests(closed.HostEvidenceTests):
    def delayed_teardown(self, delay=3.5):
        rows = [json.loads(x) for x in (self.root/'identity/raw.jsonl').read_text().splitlines()]
        terminal = rows[-2]
        rows[-1]['time_unix'] = terminal['time_unix'] + delay
        rows[-1]['monotonic_ns'] = terminal['monotonic_ns'] + int(delay*1e9)
        self.write_rows(rows)
        end = 1.1 + delay
        for step in range(3, int(end*2)+1): self.append_rss_sample(step/2, 100000)
        p = self.root/'main.log'
        stamp = closed.datetime.fromtimestamp(self.base+end+0.5, closed.timezone.utc).isoformat()
        p.write_text(p.read_text().replace('2023-11-14T22:13:22+00:00', stamp))
        for name, delta in [('unit-after.json',0.6),('cgroup-after.json',0.7),('swap-after.json',0.8)]:
            p=self.root/name; data=json.loads(p.read_text()); data['observed_at']=self.base+end+delta; closed.write_json(p,data)
        self.seal()

    def test_verified_teardown_with_continuous_memory_is_not_active_sampling(self):
        self.delayed_teardown()
        result=closed.score_host_observations(self.root,self.expected)
        self.assertEqual(result['verdict'],'PASS')
        self.assertLessEqual(result['maximum_identity_sample_gap_seconds'],2)
        self.assertEqual(result['terminal_cleanup_seconds'],3.5)

    def test_delayed_teardown_requires_memory_through_cleanup(self):
        self.delayed_teardown()
        p=self.root/'samples.log'; p.write_text(''.join(p.read_text().splitlines(keepends=True)[:3])); self.seal()
        with self.assertRaisesRegex(ValueError,'coverage|window'): closed.score_host_observations(self.root,self.expected)

    def test_teardown_over_five_seconds_rejects(self):
        self.delayed_teardown(5.01)
        with self.assertRaisesRegex(ValueError,'teardown|coverage'): closed.score_host_observations(self.root,self.expected)

    def test_teardown_does_not_waive_active_identity_gap(self):
        self.delayed_teardown()
        rows=[json.loads(x) for x in (self.root/'identity/raw.jsonl').read_text().splitlines()]
        rows[-2]['time_unix']+=2; rows[-2]['monotonic_ns']+=2_000_000_000
        self.write_rows(rows); self.seal()
        with self.assertRaisesRegex(ValueError,'coverage'): closed.score_host_observations(self.root,self.expected)

    def test_teardown_requires_terminal_filter_and_zero_exit(self):
        self.delayed_teardown()
        rows=[json.loads(x) for x in (self.root/'identity/raw.jsonl').read_text().splitlines()]
        rows[-2]['terminal_exec_filter_verified']=False; self.write_rows(rows); self.seal()
        with self.assertRaisesRegex(ValueError,'filter|completion|coverage'): closed.score_host_observations(self.root,self.expected)

    def test_teardown_clock_disagreement_rejects(self):
        self.delayed_teardown()
        rows=[json.loads(x) for x in (self.root/'identity/raw.jsonl').read_text().splitlines()]
        rows[-1]['monotonic_ns']+=100_000_000; self.write_rows(rows); self.seal()
        with self.assertRaisesRegex(ValueError,'clock|coverage'): closed.score_host_observations(self.root,self.expected)

if __name__=='__main__':unittest.main()
