#!/usr/bin/env python3
"""Portable setsid(1) for hosts without util-linux (macOS).

Usage: setsid.py CMD [ARG...]

Starts a new session in this process and execs CMD, so the PID the caller
recorded ($!) is the session and process-group leader. Unlike util-linux
setsid it never forks: when this process already leads a group it fails
instead, because a forked child would break the caller's PID == PGID
identity that stop and the memory watchdog signal.
"""

import os
import sys

if len(sys.argv) < 2:
    sys.exit(__doc__)
try:
    os.setsid()
except OSError as error:
    sys.exit(f"setsid.py: cannot start a new session: {error}")
os.execvp(sys.argv[1], sys.argv[1:])
