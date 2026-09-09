"""Direct host capture for model-free probes using existing GLM containment.

The caller freezes/verifies code, runtime, seed and arguments separately and
holds inference_lock through capture, scoring and post-run inventory checks.
No serving path imports this module.
"""
from contextlib import contextmanager
import fcntl
import ctypes
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import time

from glm53_host_evidence import read, require, unit_query

LOCK = Path('/run/lock/frontier-at-home/inference.lock')
VMSTAT = Path('/proc/vmstat')
CRASH_ROOT = Path('/home/bmarti44/.local/state/glm52-crashlog')
CLEANUP_SIGNALS = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}


def pidfd_open(pid):
    """Use the same Linux pidfd ABI when standalone CPython omits its binding."""
    require(type(pid) is int and 0 < pid <= 2147483647, 'invalid pidfd process ID')
    native = getattr(os, 'pidfd_open', None)
    if native is not None: return native(pid)
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]; function.restype = ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    return descriptor


def pidfd_send_signal(descriptor, number):
    require(type(descriptor) is int and 0 <= descriptor <= 2147483647 and isinstance(number, int) and
            not isinstance(number, bool) and 0 <= number < signal.NSIG, 'invalid pidfd signal arguments')
    native = getattr(signal, 'pidfd_send_signal', None)
    if native is not None: return native(descriptor, number)
    function = ctypes.CDLL(None, use_errno=True).pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]; function.restype = ctypes.c_int
    result = function(descriptor, number, None, 0)
    if result < 0:
        error = ctypes.get_errno(); raise OSError(error, os.strerror(error))
    require(result == 0, 'unexpected pidfd signal result')


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def inference_lock():
    fd = os.open(LOCK, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
    previous_handlers = {}
    def interrupted(number, frame):
        raise InterruptedError(f'probe controller received signal {number}')
    try:
        info = os.fstat(fd)
        require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and
                (info.st_dev, info.st_ino) == (LOCK.stat().st_dev, LOCK.stat().st_ino), 'inference lock identity changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for number in CLEANUP_SIGNALS:
            previous_handlers[number] = signal.signal(number, interrupted)
        ticks = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
        yield {'GLM_SAFE_PARENT_LOCK_PID': str(os.getpid()), 'GLM_SAFE_PARENT_LOCK_START_TICKS': ticks,
               'GLM_SAFE_PARENT_LOCK_FD': str(fd), 'GLM_SAFE_PARENT_LOCK_DEV_INO': f'{info.st_dev}:{info.st_ino}',
               'GLM_SAFE_PARENT_LOCK_KERNEL_KEY': f'{os.major(info.st_dev):02x}:{os.minor(info.st_dev):02x}:{info.st_ino}'}
    finally:
        os.close(fd)
        for number, handler in previous_handlers.items(): signal.signal(number, handler)


def capture_unit(path, unit):
    command = unit_query(unit)
    result = subprocess.run(command, capture_output=True, text=True, timeout=5)
    write(path, {'command': command, 'returncode': result.returncode, 'stdout': result.stdout,
                 'stderr': result.stderr, 'observed_at': time.time()})


def capture_swap(path):
    text = VMSTAT.read_text()
    write(path, {'path': str(VMSTAT), 'text': text, 'observed_at': time.time()})


def capture_cgroup(path, cgroup):
    try:
        cgroup.lstat()
    except FileNotFoundError:
        exists = False
    else:
        exists = True
    write(path, {'path': str(cgroup), 'exists': exists, 'observed_at': time.time()})
    return exists


def copy_crash(root, tag):
    text = read(root / 'wrapper.log').decode()
    records = [line for line in text.splitlines() if line.startswith('SAFE_RUN_DONE ')]
    require(len(records) == 1, 'missing or duplicated wrapper receipt')
    match = re.search(r'\bdir=(\S+)', records[0]); require(match is not None, 'missing crash path')
    path = Path(match.group(1))
    require(path.parent == CRASH_ROOT and re.fullmatch(r'[0-9]{8}-[0-9]{6}-' + re.escape(tag), path.name), 'unexpected crash path')
    for name in ('main.log', 'samples.log', 'kernel.log', 'cmd.log'):
        with (root / name).open('xb') as stream:
            stream.write(read(path / name)); stream.flush(); os.fsync(stream.fileno())
    return str(path)


def capture_wrapper(root, wrapper, tag, command, environment, timeout):
    """Capture actual observations; the wrapper exclusively owns containment.

    Caller must hold inference_lock and pass its binding in environment. On an
    observation failure, signal the identity-bound wrapper and wait for its
    cleanup. Never return while its exact cgroup still exists, so the caller
    cannot release the inference lock ahead of descendant cleanup.
    """
    require(re.fullmatch(r'glm53-[a-z0-9-]{1,30}', tag), 'invalid probe tag')
    require(0 < timeout <= 600, 'invalid probe timeout')
    require(environment.get('GLM_SAFE_PARENT_LOCK_PID') == str(os.getpid()), 'caller must hold inference lock')
    require(not (root / 'unit-live.json').exists(), 'attempt already used')
    capture_swap(root / 'swap-before.json')
    process = None; pidfd = None; failure = None; unit = None; cgroup = None
    try:
        with (root / 'wrapper.log').open('xb') as log:
            deferred = []
            handlers = {number: signal.signal(number, lambda number, frame: deferred.append(number))
                        for number in CLEANUP_SIGNALS}
            try:
                process = subprocess.Popen(['/usr/bin/bash', str(wrapper), '--tag', tag, '--', *command],
                                           env=environment, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(int(environment['GLM_SAFE_PARENT_LOCK_FD']),))
                unit = f'glm52-{tag}-{process.pid}.service'
                cgroup = Path('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice') / unit
                pidfd = pidfd_open(process.pid)
            finally:
                for number, handler in handlers.items(): signal.signal(number, handler)
            if deferred: raise InterruptedError(f'probe controller received signal {deferred[0]} during launch')
            deadline = time.monotonic() + timeout + 120
            while process.poll() is None:
                identity_raw = root / 'identity/raw.jsonl'
                if not (root / 'unit-live.json').exists() and identity_raw.exists() and identity_raw.stat().st_size:
                    capture_unit(root / 'unit-live.json', unit)
                require(time.monotonic() < deadline, 'wrapper exceeded containment deadline')
                time.sleep(0.1)
            log.flush(); os.fsync(log.fileno())
    except BaseException as error:
        failure = repr(error)
        if pidfd is not None:
            try: pidfd_send_signal(pidfd, signal.SIGTERM)
            except ProcessLookupError: pass
    finally:
        prior_mask = signal.pthread_sigmask(signal.SIG_BLOCK, CLEANUP_SIGNALS)
        try:
            if process is not None:
                # The wrapper signal trap and systemd RuntimeMaxSec own shutdown.
                # Uncertain observations never authorize lock release.
                while process.poll() is None: time.sleep(0.2)
                if cgroup is not None:
                    while True:
                        try: cgroup.lstat()
                        except FileNotFoundError: break
                        except OSError as error:
                            failure = failure or repr(error)
                            try:
                                if not (root / 'cleanup-observation-error.json').exists():
                                    write(root / 'cleanup-observation-error.json', {'failure': repr(error), 'time_unix': time.time()})
                            except OSError: pass
                        time.sleep(0.2)
                    try:
                        capture_unit(root / 'unit-after.json', unit)
                        capture_cgroup(root / 'cgroup-after.json', cgroup)
                    except Exception as error: failure = failure or repr(error)
                try: capture_swap(root / 'swap-after.json')
                except Exception as error: failure = failure or repr(error)
        finally:
            if pidfd is not None: os.close(pidfd)
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_mask)
    result = {'wrapper_exit_code': process.returncode if process else None, 'capture_failure': failure, 'unit': unit}
    try: result['crash_directory'] = copy_crash(root, tag)
    except Exception as error: result['capture_failure'] = result['capture_failure'] or repr(error)
    write(root / 'capture.json', result)
    require(result['wrapper_exit_code'] == 0 and result['capture_failure'] is None, 'wrapper or host capture failed: ' + str(result))
    return result
