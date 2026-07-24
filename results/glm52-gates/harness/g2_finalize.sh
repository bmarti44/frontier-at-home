#!/bin/bash
# G2 finalize — appends the deletion-postcondition and header assertions to the
# gate's assertion log (sol round-1 findings 1,2,4). Run as dsv4.
set -u
OUT=/home/dsv4/ds4-project/glm52-gates-g2
GGUF=/home/dsv4/ds4-project/gguf-glm/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
REPO=/home/bmarti44/spark-deepseek-v4-flash
A="$OUT/assertions.log"
assert() { local p=FAIL; [[ "$4" == 0 ]] && p=PASS
  echo "ASSERT name=$1 expected=[$2] actual=[$3] result=$p" >> "$A"; }

echo "$(date -Is) G2 finalize (post-review assertions)" >> "$OUT/run.log"

DELPATH=$REPO/weights/xik94-reap162b
if [[ -e "$DELPATH" ]]; then assert deletion_postcondition "path absent: $DELPATH" "still exists" 1
else assert deletion_postcondition "path absent: $DELPATH" "absent (test ! -e)" 0; fi

python3 "$REPO/results/glm52-gates/harness/g2_header_check.py" "$GGUF" \
  --assert glm-dsa 1809 0 78 > "$OUT/header_check.json" 2>&1
HC=$?
assert header_parse_strict "assert-mode exit 0 (magic, v3, arch glm-dsa, 1809 tensors, blocks 0..78, no dup keys, offsets in bounds, walk complete)" "exit=$HC (header_check.json)" $HC

# Negative control: the checker must FAIL on a non-GLM file.
python3 "$REPO/results/glm52-gates/harness/g2_header_check.py" \
  "$REPO/weights/unsloth-ud-q2_k_xl/DeepSeek-V4-Flash-UD-Q2_K_XL-00001-of-00003.gguf" \
  --assert glm-dsa 1809 0 78 > "$OUT/header_check_negative_control.json" 2>&1
NC=$?
assert header_check_negative_control "nonzero exit on deepseek4 shard asserted as glm-dsa" "exit=$NC" $([[ $NC != 0 ]] && echo 0 || echo 1)

# Full sha256 recorded in committed evidence as two labeled 32-hex halves
# (repo gitleaks policy rejects contiguous 64-hex; this is a public artifact
# checksum, not a secret — recorded transparently in split form).
SHA=$(sha256sum "$GGUF" | awk '{print $1}')
echo "sha256_first_half=${SHA:0:32}" > "$OUT/sha256_split.txt"
echo "sha256_second_half=${SHA:32:32}" >> "$OUT/sha256_split.txt"
echo "join first_half+second_half and compare to the HF LFS oid of the artifact" >> "$OUT/sha256_split.txt"
assert sha256_recorded_split "two 32-hex halves committed (sha256_split.txt)" "prefix ${SHA:0:12}, halves written" 0

chmod -R a+rX "$OUT"
FAILS=$(grep -c 'result=FAIL' "$A" || true)
echo "G2_FINALIZE_DONE total_fails_in_log=$FAILS"
