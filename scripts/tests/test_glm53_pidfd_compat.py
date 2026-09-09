"""Exercise real pidfds with both CPython and libc entry points; CPU only."""
import importlib.util
import os
from pathlib import Path
import signal
import sys
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))
import glm53_probe_capture as capture
spec=importlib.util.spec_from_file_location('guard_pidfd',Path(__file__).resolve().parents[1]/'38_guard_glm53_probe.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)

class PidfdCompatibilityTests(unittest.TestCase):
    def test_real_pidfd_open_signal_zero_and_closed_descriptor(self):
        for module in (capture,guard):
            with self.subTest(module=module.__name__):
                fd=module.pidfd_open(os.getpid())
                self.assertIn('Pid:',Path(f'/proc/self/fdinfo/{fd}').read_text())
                self.assertIsNone(module.pidfd_send_signal(fd,0));os.close(fd)
                with self.assertRaises(OSError):module.pidfd_send_signal(fd,0)

    def test_missing_python_entry_points_use_same_kernel_pidfd(self):
        for module in (capture,guard):
            with self.subTest(module=module.__name__), mock.patch.object(os,'pidfd_open',None,create=True), mock.patch.object(signal,'pidfd_send_signal',None,create=True):
                fd=module.pidfd_open(os.getpid())
                try:self.assertIsNone(module.pidfd_send_signal(fd,0))
                finally:os.close(fd)
                with self.assertRaises(OSError):module.pidfd_open(2147483647)

    def test_invalid_integer_arguments_cannot_wrap_to_another_process(self):
        for module in (capture,guard):
            for pid in (True,-1,0,2**32+os.getpid()):
                with self.subTest(module=module.__name__,pid=pid),self.assertRaises(ValueError):module.pidfd_open(pid)

if __name__=='__main__':unittest.main()
