"""Validate native counter availability and reject malformed observations."""
import copy
import importlib.util
from pathlib import Path
import unittest
spec = importlib.util.spec_from_file_location('memcg_observer', Path(__file__).with_name('observe.py'))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class AccountingTests(unittest.TestCase):
    def rows(self):
        row = api.snapshot()
        # Synthetic timestamps test scorer rejection, never used as measurement.
        row['before']['observed_ns'] = 0
        row['after']['observed_ns'] = 1
        last = copy.deepcopy(row)
        last['before']['observed_ns'] = 1_000_000_000
        last['after']['observed_ns'] = 1_000_000_001
        return [{'kind': 'sample', 'index': 0, **row},
                {'kind': 'sample', 'index': 1, **last}, {'kind': 'end', 'observed_ns': 1_000_000_002}]

    def test_installed_counters_and_valid_shape(self):
        result = api.score(self.rows(), 1)
        self.assertEqual(result['observation_integrity'], 'PASS')
        self.assertEqual(result['verdict'], 'NO_RESULT')

    def test_malformed_or_missing_counter_rejected(self):
        for text in ['pswpin 0\n', 'pswpin -1\npswpout 0\n',
                     'pswpin 0\npswpin 1\npswpout 0\n', 'pswpin NaN\npswpout 0\n']:
            with self.subTest(text=text), self.assertRaises((ValueError, KeyError)):
                api.counters(text)

    def test_broken_sample_evidence_rejected(self):
        mutations = [lambda r: r.pop(1), lambda r: r[1].update(index=0),
                     lambda r: r[1]['after'].update(time_unix=float('nan')),
                     lambda r: r[1]['groups']['.'].update(inode=-1),
                     lambda r: r[1]['after'].update(pswpin=-1),
                     lambda r: r[-1].update(kind='error'),
                     lambda r: r[-1].pop('observed_ns')]
        for mutate in mutations:
            rows = self.rows()
            mutate(rows)
            with self.assertRaises((ValueError, KeyError)):
                api.score(rows, 1)

    def test_cgroup_counter_reset_rejected_even_when_raw_matches(self):
        rows = self.rows()
        group = rows[1]['groups']['.']
        old = group['counters']['pswpin']
        self.assertGreater(old, 0)
        group['raw'] = group['raw'].replace(f'pswpin {old}\n', f'pswpin {old - 1}\n')
        group['counters']['pswpin'] -= 1
        with self.assertRaisesRegex(ValueError, 'cgroup counter reset'):
            api.score(rows, 1)


if __name__ == '__main__': unittest.main()
