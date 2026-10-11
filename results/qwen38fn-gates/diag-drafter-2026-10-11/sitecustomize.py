# DIAGNOSTIC only (diag-drafter-2026-10-11): reached through a scratch
# PYTHONPATH, never the production scripts/lib. SIGUSR1 dumps every
# thread's Python stack to stderr (the server log).
import faulthandler
import signal
import sys

faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
