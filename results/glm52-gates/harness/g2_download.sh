#!/bin/bash
# G2 harness — GLM-5.2 GGUF download + deterministic verification.
# Run as dsv4: sudo -u dsv4 bash results/glm52-gates/harness/g2_download.sh
# Precondition: >= 252 GB free on / (211.1 GB artifact + 40 GB post headroom + margin).
set -u
OUT=/home/dsv4/ds4-project/glm52-gates-g2
DEST=/home/dsv4/ds4-project/gguf-glm
REPO_ID=antirez/GLM-5.2-GGUF
FILE=GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf
EXPECTED_BYTES=211075856448
# Full LFS sha256 is fetched from the HF API at run time (repo policy: no
# 64-char hex literals in git). It must start with this pinned prefix:
EXPECTED_SHA256_PREFIX=a49de64c5020
mkdir -p "$OUT" "$DEST"
A="$OUT/assertions.log"; : > "$A"
note() { echo "$(date -Is) $*" >> "$OUT/run.log"; }
assert() { local p=FAIL; [[ "$4" == 0 ]] && p=PASS
  echo "ASSERT name=$1 expected=[$2] actual=[$3] result=$p" >> "$A"; }

note "G2 start host=$(hostname)"
FREE=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
note "free_bytes_before=$FREE"
assert disk_precondition ">= 252000000000 bytes free" "$FREE" $(( FREE >= 252000000000 ? 0 : 1 ))
if (( FREE < 252000000000 )); then echo "G2_ABORT insufficient disk"; exit 1; fi

# HF API cross-check at run time (size/oid may not silently change)
curl -sL "https://huggingface.co/api/models/$REPO_ID/tree/main" > "$OUT/hf_api_tree.json"
API_BYTES=$(python3 -c 'import json;print([f.get("size") for f in json.load(open("'"$OUT"'/hf_api_tree.json")) if f["path"]=="'"$FILE"'"][0])')
EXPECTED_SHA256=$(python3 -c 'import json;print([f["lfs"]["oid"] for f in json.load(open("'"$OUT"'/hf_api_tree.json")) if f["path"]=="'"$FILE"'"][0])')
assert hf_api_size "$EXPECTED_BYTES" "$API_BYTES" $([[ "$API_BYTES" == "$EXPECTED_BYTES" ]] && echo 0 || echo 1)
assert hf_api_oid_prefix "$EXPECTED_SHA256_PREFIX" "${EXPECTED_SHA256:0:12}" $([[ "${EXPECTED_SHA256:0:12}" == "$EXPECTED_SHA256_PREFIX" ]] && echo 0 || echo 1)

# PEP 668: system python is externally managed — use a dedicated venv.
VENV="$HOME/hf-venv"
if [[ ! -x "$VENV/bin/hf" ]]; then
  note "creating venv + installing huggingface_hub + hf_xet"
  python3 -m venv "$VENV" >> "$OUT/pip.log" 2>&1
  "$VENV/bin/pip" install -q -U huggingface_hub hf_xet >> "$OUT/pip.log" 2>&1
fi
export PATH="$VENV/bin:$PATH"
command -v hf >/dev/null 2>&1 || { assert hf_cli_present "hf on PATH" "missing" 1; echo "G2_ABORT no hf cli"; exit 1; }
assert hf_cli_present "hf on PATH (venv)" "$(command -v hf)" 0

note "download begin ($FILE, $EXPECTED_BYTES bytes)"
hf download "$REPO_ID" "$FILE" --repo-type model --local-dir "$DEST" >> "$OUT/download.log" 2>&1
DLEXIT=$?
note "download end exit=$DLEXIT"
assert download_exit_0 "0" "$DLEXIT" $([[ $DLEXIT == 0 ]] && echo 0 || echo 1)
[[ $DLEXIT == 0 ]] || { echo "G2_DONE result=FAIL"; exit 1; }

ACTUAL_BYTES=$(stat -c%s "$DEST/$FILE")
assert byte_size "$EXPECTED_BYTES" "$ACTUAL_BYTES" $([[ "$ACTUAL_BYTES" == "$EXPECTED_BYTES" ]] && echo 0 || echo 1)

note "sha256 begin"
ACTUAL_SHA=$(sha256sum "$DEST/$FILE" | awk '{print $1}')
note "sha256 end"
SHAOK=1; [[ "$ACTUAL_SHA" == "$EXPECTED_SHA256" ]] && SHAOK=0
# committed evidence carries only 12-char prefixes (repo gitleaks policy)
assert sha256_matches_lfs_oid "${EXPECTED_SHA256:0:12} (12-char prefix; full compare done in-shell)" "${ACTUAL_SHA:0:12} full_match=$([[ $SHAOK == 0 ]] && echo yes || echo no)" $SHAOK

FREE_AFTER=$(df -B1 --output=avail / | tail -1 | tr -d ' ')
note "free_bytes_after=$FREE_AFTER"
assert free_after ">= 40000000000 bytes (40 GB)" "$FREE_AFTER" $(( FREE_AFTER >= 40000000000 ? 0 : 1 ))

chmod a+rX "$DEST" 2>/dev/null; chmod a+r "$DEST/$FILE" 2>/dev/null
chmod -R a+rX "$OUT"
FAILS=$(grep -c 'result=FAIL' "$A" || true)
note "G2 end fails=$FAILS"
echo "G2_DONE result=$([[ "$FAILS" == 0 ]] && echo PASS || echo "FAIL($FAILS)")"
