"""Real child-process enforcement for the evidence-only frozen cache boundary."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

LIB = Path(__file__).resolve().parents[1] / 'lib'
PREAMBLE = '''
import json, mmap, os, pathlib, subprocess, sys, threading
sys.path.insert(0, sys.argv[1])
from glm53_replay_write_guard import activate
root = pathlib.Path(sys.argv[2])
cache, output = root/'cache', root/'output'
kernel = cache/'kernel.cubin'
def guard(**kwargs):
    return activate([cache], [output], enabled=True, **kwargs)
def rejected(action):
    try: action()
    except (ValueError, PermissionError): return
    raise AssertionError('forbidden action succeeded')
'''


class ReplayWriteGuardTests(unittest.TestCase):
    def child(self, body):
        with tempfile.TemporaryDirectory(prefix='glm53-write-guard-') as tmp:
            root = Path(tmp)
            (root/'cache').mkdir(); (root/'output').mkdir()
            (root/'cache/kernel.cubin').write_bytes(b'frozen kernel bytes')
            run = subprocess.run([sys.executable, '-I', '-B', '-c',
                                  textwrap.dedent(PREAMBLE) + textwrap.dedent(body), str(LIB), tmp],
                                 capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            return json.loads(run.stdout)

    def test_disabled_requires_no_existing_paths_or_restrictions(self):
        result = self.child('''
            receipt = activate(['/absent'], ['/absent'])
            kernel.write_bytes(b'disabled remains writable')
            print(json.dumps(receipt))
        ''')
        self.assertEqual(result, {'selection': 'disabled'})

    def test_real_write_denials_and_compilation_directory_seam(self):
        result = self.child('''
            receipt = guard(writable_devices=['/dev/null'])
            assert kernel.read_bytes() == b'frozen kernel bytes'
            (output/'new').write_bytes(b'evidence')
            (output/'directory').mkdir()
            (output/'new').rename(output/'renamed')
            with open('/dev/null', 'wb') as sink: sink.write(b'ok')
            def compile_miss():
                (cache/'tmp'/'uuid').mkdir(parents=True)
                (output/'compiler-entered').write_text('bad')
            for action in [compile_miss, lambda: kernel.write_bytes(b'changed'),
                           lambda: os.truncate(kernel, 0), lambda: kernel.unlink(),
                           lambda: (output/'renamed').replace(kernel),
                           lambda: os.link(kernel, output/'alias'),
                           lambda: cache.rename(root/'moved')]:
                rejected(action)
            assert not (output/'compiler-entered').exists()
            assert kernel.read_bytes() == b'frozen kernel bytes'
            print(json.dumps(receipt))
        ''')
        self.assertEqual(result['selection'], 'landlock_frozen_cache_writes')
        self.assertGreaterEqual(result['abi'], 3)
        self.assertEqual(result['no_new_privs'], 1)

    def test_ancestor_symlink_and_hardlink_allowances_rejected(self):
        self.child('''
            rejected(lambda: activate([cache], [root], enabled=True))
            (root/'link').symlink_to(output, target_is_directory=True)
            rejected(lambda: activate([cache], [root/'link'], enabled=True))
            os.link(kernel, output/'alias')
            rejected(guard)
            print('{}')
        ''')

    def test_open_file_descriptor_rejected_even_through_alias(self):
        self.child('''
            with kernel.open('r+b') as opened: rejected(guard)
            os.link(kernel, output/'alias')
            with (output/'alias').open('r+b') as opened: rejected(guard)
            print('{}')
        ''')

    def test_shared_writable_mapping_survives_closed_fd_and_is_rejected(self):
        self.child('''
            with kernel.open('r+b') as opened:
                mapping = mmap.mmap(opened.fileno(), 0, flags=mmap.MAP_SHARED)
            rejected(guard)
            mapping.close()
            print('{}')
        ''')

    def test_existing_thread_rejected(self):
        self.child('''
            stop = threading.Event()
            worker = threading.Thread(target=stop.wait); worker.start()
            try: rejected(guard)
            finally: stop.set(); worker.join()
            print('{}')
        ''')

    def test_future_thread_exec_child_and_completion_pipes(self):
        self.child('''
            read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
            guard()
            errors = []
            def thread_body():
                try: kernel.write_bytes(b'bad')
                except PermissionError: return
                errors.append('thread escaped')
            worker = threading.Thread(target=thread_body); worker.start(); worker.join()
            assert not errors
            child = subprocess.run([sys.executable, '-I', '-B', '-c',
                'import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b"bad")', str(kernel)],
                capture_output=True, text=True, timeout=5)
            assert child.returncode != 0 and 'PermissionError' in child.stderr
            os.write(write_fd, b'completed'); assert os.read(read_fd, 9) == b'completed'
            os.close(read_fd); os.close(write_fd)
            print('{}')
        ''')


if __name__ == '__main__': unittest.main()
