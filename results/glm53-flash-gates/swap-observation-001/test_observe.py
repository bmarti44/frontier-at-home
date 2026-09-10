"""Counter positions and identity parsing for external diagnostic observations."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('observer',Path(__file__).with_name('observe.py'))
api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)

class Observation(unittest.TestCase):
    def test_complex_task_name_does_not_shift_counters(self):
        fields=['S']+['0']*21;fields[7]='11';fields[9]='7';fields[19]='12345'
        raw='42 (name with (parentheses)) '+' '.join(fields)
        self.assertEqual(api.stat_fields(raw),{'pid':42,'name':'name with (parentheses)',
            'start_ticks':12345,'major_faults':7,'minor_faults':11})
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp);(p/'stat').write_text(raw);(p/'status').write_text('Uid:\t0 0 0 0\n');(p/'cgroup').write_text('0::/synthetic\n')
            self.assertIsNone(api.process(p)['vm_swap_kib'])
            (p/'status').write_text('Uid:\t1000 1000 1000 1000\nVmSwap:\t4 kB\n')
            self.assertEqual(api.process(p)['vm_swap_kib'],4)
            (p/'status').write_text('Uid:\t1000 1000 1000 1000\nVmSwap:\t4 MB\n')
            with self.assertRaises(ValueError):api.process(p)

    def test_actual_self_identity_and_global_counter_coverage(self):
        row=api.process(Path('/proc/self'))
        self.assertGreater(row['pid'],0);self.assertGreater(row['start_ticks'],0)
        self.assertGreaterEqual(row['major_faults'],0);self.assertGreaterEqual(row['vm_swap_kib'],0)
        counters=api.swap_counters()
        self.assertGreater(counters['MemAvailable_kib'],0)
        self.assertGreaterEqual(counters['pswpin'],0);self.assertGreaterEqual(counters['pswpout'],0)

if __name__=='__main__':unittest.main()
