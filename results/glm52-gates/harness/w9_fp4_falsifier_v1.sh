#!/bin/bash
# Frozen CPU-only launcher for the W9 real-tensor FP4 falsifier.
set -Eeuo pipefail
umask 077

[[ $# == 2 && -f $1 && ! -L $1 && $2 =~ ^attempt-[a-z0-9][a-z0-9-]{0,79}$ ]]
exec /usr/bin/sudo -n /usr/local/sbin/glm52-w9-submit run "$1" "$2"
