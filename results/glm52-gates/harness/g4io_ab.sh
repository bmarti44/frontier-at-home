#!/bin/bash
set -u
SRC=/home/dsv4/ds4-project/src/ds4-upstream-master
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
OUT=/home/dsv4/ds4-project/glm52-io-ab
mkdir -p "$OUT"
P=$(python3 -c "print(' '.join(['word%d'%i for i in range(120)]))")
for T in 0 6 12; do
  T0=$(date +%s)
  DS4_CUDA_FETCH_THREADS=$T DS4_GLM_TP_DEBUG=0 DS4_CUDA_MOE_NO_ATOMIC_DOWN=1 \
    timeout 2400 "$SRC/ds4" --cuda -m "$GGUF" --ssd-streaming \
    --ssd-streaming-cache-experts 40GB -c 4096 -n 8 -p "$P" \
    > "$OUT/run-t$T.log" 2>&1
  echo "threads=$T rc=$? wall=$(( $(date +%s) - T0 ))s $(grep -oE 'prefill: [0-9.]+ t/s, generation: [0-9.]+ t/s' $OUT/run-t$T.log)"
done
for T in 0 6 12; do echo "t$T out_sha=$(grep -A2 'input tokens' $OUT/run-t$T.log | tail -1 | sha256sum | cut -c1-12)"; done
chmod -R a+rX "$OUT"
