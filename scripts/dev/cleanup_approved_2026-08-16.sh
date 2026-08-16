#!/usr/bin/env bash
# Owner-approved disk cleanup (options B/C/D from 2026-08-16 audit).
# Option A (177 GB rejected slab) is already deleted.
# Run: sudo bash scripts/dev/cleanup_approved_2026-08-16.sh
set -uo pipefail

echo "== B: root-owned smoke training checkpoints + Trash (~102 GB)"
rm -rf /home/bmarti44/distiallation/runs/r2e-3b-smoke \
       /home/bmarti44/distiallation/runs/gsm8k-smoke/ckpt \
       /home/bmarti44/.local/share/Trash/files \
       /home/bmarti44/.local/share/Trash/info

echo "== C: pre-p15 GLM campaign/corpus/smoke state (~130 GB)"
echo "   (keeps glm52-decisive-p15-r6379759, glm52-crashlog, and all non-GLM dirs)"
cd /home/bmarti44/.local/state
rm -rf glm52-p0-corpus-long8000-r99-pcache glm52-p0-corpus-long8000-r97b \
  glm52-p0-corpus-long8000-r98 glm52-p0-shards glm52-p0-corpus-v2-r88-6b685fc \
  glm52-p0-corpus-r93-1069149610 glm52-p0-corpus-r85-468e14f \
  glm52-w7-equivalence glm52-w7-red glm52-w7-cache-generation \
  glm52-w7-production-equivalence glm52-decisive-p10-r6379414 \
  glm52-c13-fault-smoke2.a00d73kj glm52-c13-fault-smoke.5yv26ips \
  glm52-c8-final-smoke.6ywwhu88 glm52-c8-named-smoke.krfzol8f \
  glm52-c8-runtime-smoke.21110a1q glm52-c8-runtime-smoke.9ptb0tz3 \
  glm52-c8-runtime-smoke.ivvndogk glm52-c8-runtime-smoke.p1jb174h \
  glm52-c9-train-smoke.7ma89t_5 \
  glm52-confirm-w1-affine-0b14242-3685845-r6329478 \
  glm52-confirm-w1-affine-0b14242-5cfc1d7-r6329464 \
  glm52-confirm-w1-affine-23671c629dde-68ed773974ab \
  glm52-controller-W1-66d936e-0b14242 glm52-debug-startup

echo "== D: approved media/model dirs (~330 GB)"
rm -rf /home/bmarti44/ComfyUI/models \
       /home/bmarti44/models/ltx-training \
       /home/bmarti44/.local/share/producer-at-home

echo "== Result:"
df -h / | tail -1
