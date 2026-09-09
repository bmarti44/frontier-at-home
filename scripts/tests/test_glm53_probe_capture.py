"""CPU controls for raw probe observation capture; no GPU or systemd mutation."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))
import glm53_probe_capture as capture

class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_parent_lock_is_real_and_excludes_second_opener(self):
        path = self.root / 'lock'; path.touch()
        with mock.patch.object(capture, 'LOCK', path), capture.inference_lock() as env:
            self.assertEqual(env['GLM_SAFE_PARENT_LOCK_PID'], str(os.getpid()))
            fd = int(env['GLM_SAFE_PARENT_LOCK_FD'])
            self.assertIn('FLOCK', Path(f'/proc/self/fdinfo/{fd}').read_text())
            with self.assertRaises(BlockingIOError):
                with capture.inference_lock(): pass
        with mock.patch.object(capture, 'LOCK', path), capture.inference_lock(): pass

    def test_lock_symlink_rejects(self):
        path = self.root / 'lock'; target = self.root / 'target'; target.touch(); path.symlink_to(target)
        with mock.patch.object(capture, 'LOCK', path), self.assertRaises(OSError):
            with capture.inference_lock(): pass

    def test_unit_capture_preserves_failed_actual_query(self):
        result = SimpleNamespace(returncode=1, stdout='Id=actual\n', stderr='query failed')
        with mock.patch.object(capture.subprocess, 'run', return_value=result):
            capture.capture_unit(self.root / 'unit.json', 'glm52-glm53-native-smoke-004-321.service')
        row = json.loads((self.root / 'unit.json').read_text())
        self.assertEqual((row['returncode'], row['stdout'], row['stderr']), (1, 'Id=actual\n', 'query failed'))
        self.assertEqual(row['command'][0], '/usr/bin/systemctl')

    def test_cgroup_permission_error_is_not_absence(self):
        with mock.patch.object(Path, 'lstat', side_effect=PermissionError('denied')), self.assertRaises(PermissionError):
            capture.capture_cgroup(self.root / 'cg.json', self.root / 'cgroup')
        self.assertFalse((self.root / 'cg.json').exists())

    def test_vmstat_capture_keeps_raw_text(self):
        path = self.root / 'vmstat'; path.write_text('nr_free_pages 1\npswpin 2\npswpout 3\n')
        with mock.patch.object(capture, 'VMSTAT', path): capture.capture_swap(self.root / 'swap.json')
        row = json.loads((self.root / 'swap.json').read_text())
        self.assertEqual(row['text'], path.read_text())

    def run_cpu_wrapper(self, exit_code):
        tag = 'glm53-native-smoke-004'
        crash_root = self.root / 'crashes'; crash = crash_root / ('20260909-120000-' + tag); crash.mkdir(parents=True)
        for name in ('main.log', 'samples.log', 'kernel.log', 'cmd.log'): (crash / name).write_text('raw bytes\n')
        script = self.root / 'wrapper.sh'
        script.write_text("sleep 0.3\nprintf '%s\\n' 'SAFE_RUN_DONE rc=" + str(exit_code) + " killed=no dir=" + str(crash) + "'\nexit " + str(exit_code) + "\n")
        (self.root / 'identity').mkdir(); (self.root / 'identity/raw.jsonl').write_text('{"event":"identity"}\n')
        lock = self.root / 'lock'; lock.touch()
        def unit(path, name):
            self.assertIn('FLOCK', Path('/proc/self/fdinfo/' + environment['GLM_SAFE_PARENT_LOCK_FD']).read_text())
            capture.write(path, {'unit': name, 'actual_query_mocked_for_CPU_control': True})
        with mock.patch.object(capture, 'LOCK', lock), capture.inference_lock() as environment:
            with mock.patch.object(capture, 'CRASH_ROOT', crash_root), mock.patch.object(capture, 'capture_unit', side_effect=unit):
                return capture.capture_wrapper(self.root, script, tag, [], {**os.environ, **environment}, 10)

    def test_cpu_wrapper_captures_raw_logs_and_holds_lock_through_cleanup(self):
        result = self.run_cpu_wrapper(0)
        self.assertEqual(result['wrapper_exit_code'], 0)
        self.assertEqual((self.root / 'main.log').read_text(), 'raw bytes\n')
        self.assertTrue((self.root / 'unit-live.json').exists())
        self.assertTrue((self.root / 'unit-after.json').exists())
        self.assertFalse(json.loads((self.root / 'cgroup-after.json').read_text())['exists'])

    def test_failed_cpu_wrapper_keeps_logs_and_rejects(self):
        with self.assertRaisesRegex(ValueError, 'failed'): self.run_cpu_wrapper(1)
        self.assertEqual(json.loads((self.root / 'capture.json').read_text())['wrapper_exit_code'], 1)
        self.assertEqual((self.root / 'cmd.log').read_text(), 'raw bytes\n')

    def test_crash_copy_rejects_outside_path_and_duplicate_receipts(self):
        for log in ('SAFE_RUN_DONE rc=0 killed=no dir=/etc\n',
                    'SAFE_RUN_DONE rc=0 killed=no dir=/a\nSAFE_RUN_DONE rc=1 killed=yes dir=/b\n'):
            (self.root / 'wrapper.log').write_text(log)
            with self.assertRaises(ValueError): capture.copy_crash(self.root, 'glm53-native-smoke-004')

if __name__ == '__main__': unittest.main()
