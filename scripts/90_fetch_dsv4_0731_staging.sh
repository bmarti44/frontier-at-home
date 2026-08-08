#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C

# Staging-only fetch for the DeepSeek-V4-Flash-0731 UD-Q2_K_XL candidate.
# Mirrors scripts/12_fetch_gguf.sh's verify-then-download-then-verify shape,
# but targets the OUT-OF-TREE staging path (not weights/) so this never
# collides with the qualified v0.4.2 weights, and downloads shards
# sequentially (guardrail: no parallel shard fetches; this is staging, not
# production, so it runs under the operator's own account, never as dsv4,
# and never touches /home/dsv4).
#
# All artifact pins live in configs/pins/unsloth-ud-q2_k_xl-0731.json.

readonly PIN_FILE_REL="configs/pins/unsloth-ud-q2_k_xl-0731.json"
readonly DEFAULT_DESTINATION="/home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl"
readonly MIN_FREE_RATIO_NUM=3   # 1.5x expressed as an integer ratio (3/2) to
readonly MIN_FREE_RATIO_DEN=2   # avoid floating point in bash arithmetic.
readonly MIN_MEM_AVAILABLE_KIB=$((20 * 1024 * 1024))  # 20 GiB courtesy floor

usage() {
    cat <<'EOF'
Usage: scripts/90_fetch_dsv4_0731_staging.sh [--verify-only] [--destination PATH]

Download and verify the pinned unsloth/DeepSeek-V4-Flash-0731-GGUF
UD-Q2_K_XL shards into a staging directory (default
/home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl). Shards download
sequentially. Never writes under /home/dsv4 and never starts any engine.

Options:
  --verify-only        Verify final files without downloading anything.
  --destination PATH   Override the staging destination directory.
  -h, --help            Show this help message.
EOF
}

verify_only=false
destination="$DEFAULT_DESTINATION"
while (($# > 0)); do
    case "$1" in
        --verify-only)
            verify_only=true
            ;;
        --destination)
            [[ $# -ge 2 ]] || { echo "90_fetch_dsv4_0731_staging.sh: --destination requires a path" >&2; exit 2; }
            destination=$2
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            printf '90_fetch_dsv4_0731_staging.sh: unknown argument: %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

repo_root=$(cd "$(dirname "$0")/.." && pwd)
pin_file="$repo_root/$PIN_FILE_REL"

[[ -r $pin_file ]] || { printf '90_fetch_dsv4_0731_staging.sh: pin file missing: %s\n' "$pin_file" >&2; exit 2; }

REPO=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["repo"])' "$pin_file")
REVISION=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$pin_file")
readonly REPO REVISION

declare -a FILE_PATHS=() FILE_BYTES=() FILE_SHA256=()
while IFS=$'\t' read -r p b s; do
    FILE_PATHS+=("$p"); FILE_BYTES+=("$b"); FILE_SHA256+=("$s")
done < <(python3 -c '
import json, sys
for f in json.load(open(sys.argv[1]))["files"]:
    print(f["path"], f["bytes"], f["sha256"], sep="\t")' "$pin_file")
((${#FILE_PATHS[@]} > 0)) || { printf '90_fetch_dsv4_0731_staging.sh: pin file lists no files\n' >&2; exit 2; }

total_pinned_bytes=0
for b in "${FILE_BYTES[@]}"; do
    total_pinned_bytes=$((total_pinned_bytes + b))
done
min_free_bytes=$(( total_pinned_bytes * MIN_FREE_RATIO_NUM / MIN_FREE_RATIO_DEN ))

command -v python3 >/dev/null 2>&1 || { echo "90_fetch_dsv4_0731_staging.sh: python3 is required" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "90_fetch_dsv4_0731_staging.sh: sha256sum is required" >&2; exit 1; }
if [[ $verify_only == false ]]; then
    command -v curl >/dev/null 2>&1 || { echo "90_fetch_dsv4_0731_staging.sh: curl is required" >&2; exit 1; }
fi

df_target=$destination
while [[ ! -e $df_target ]]; do
    df_target=$(dirname "$df_target")
done
free_bytes=$(df -B1 --output=avail "$df_target" 2>/dev/null | awk 'NR == 2 {print $1}') ||
    { echo "90_fetch_dsv4_0731_staging.sh: could not determine free space for $destination" >&2; exit 2; }
[[ $free_bytes =~ ^[0-9]+$ ]] ||
    { echo "90_fetch_dsv4_0731_staging.sh: could not determine free space for $destination" >&2; exit 2; }
if [[ $verify_only == false ]] && ((free_bytes < min_free_bytes)); then
    printf '90_fetch_dsv4_0731_staging.sh: need >= %s bytes free (1.5x pinned %s bytes), have %s\n' \
        "$min_free_bytes" "$total_pinned_bytes" "$free_bytes" >&2
    exit 2
fi

mem_available_kib=$(awk '$1 == "MemAvailable:" {print $2}' /proc/meminfo 2>/dev/null || true)
if [[ $verify_only == false ]] && [[ $mem_available_kib =~ ^[0-9]+$ ]] && ((mem_available_kib < MIN_MEM_AVAILABLE_KIB)); then
    printf '90_fetch_dsv4_0731_staging.sh: MemAvailable %s KiB below 20 GiB courtesy floor; pausing (re-run when memory recovers)\n' \
        "$mem_available_kib" >&2
    exit 3
fi

if [[ $verify_only == false ]]; then
    mkdir -p "$destination"
fi

verification_failed=false
files_present=0
bytes_total=0
for i in "${!FILE_PATHS[@]}"; do
    file_path=${FILE_PATHS[$i]}
    name=${file_path##*/}
    expected_bytes=${FILE_BYTES[$i]}
    expected_sha256=${FILE_SHA256[$i]}
    final_file="$destination/$name"
    partial_file="$final_file.partial"

    if [[ -f $final_file ]]; then
        actual_bytes=$(stat -c %s "$final_file")
        if [[ $actual_bytes != "$expected_bytes" ]]; then
            printf '%s: size mismatch (expected %s, got %s) — removing\n' \
                "$name" "$expected_bytes" "$actual_bytes" >&2
            rm -f "$final_file"
        fi
    fi

    if [[ ! -f $final_file ]]; then
        if [[ $verify_only == true ]]; then
            printf '%s: missing\n' "$name" >&2
            verification_failed=true
            continue
        fi

        url="https://huggingface.co/$REPO/resolve/$REVISION/$file_path"
        printf '%s: downloading (sequential, no parallel shard fetches)\n' "$name" >&2
        attempt=1
        shard_ok=false
        while ((attempt <= 2)); do
            if curl -L --fail --retry 5 --retry-delay 10 --continue-at - \
                --output "$partial_file" "$url"; then
                actual_bytes=$(stat -c %s "$partial_file")
                actual_sha256=$(sha256sum "$partial_file" | awk '{print $1}')
                if [[ $actual_bytes == "$expected_bytes" && $actual_sha256 == "$expected_sha256" ]]; then
                    shard_ok=true
                    break
                fi
                printf '%s: verification failed on attempt %d (bytes=%s sha256=%s)\n' \
                    "$name" "$attempt" "$actual_bytes" "$actual_sha256" >&2
                rm -f "$partial_file"
            else
                printf '%s: download failed on attempt %d\n' "$name" "$attempt" >&2
            fi
            attempt=$((attempt + 1))
        done
        if [[ $shard_ok != true ]]; then
            printf '%s: BLOCKER — failed verification twice, not retrying further\n' "$name" >&2
            exit 1
        fi
        mv "$partial_file" "$final_file"
        printf '%s: installed and verified\n' "$name" >&2
    else
        actual_sha256=$(sha256sum "$final_file" | awk '{print $1}')
        if [[ $actual_sha256 != "$expected_sha256" ]]; then
            printf '%s: sha256 mismatch — removing\n' "$name" >&2
            rm -f "$final_file"
            if [[ $verify_only == true ]]; then
                verification_failed=true
                continue
            fi
            printf '%s: BLOCKER — pinned bytes matched but sha256 did not\n' "$name" >&2
            exit 1
        fi
        printf '%s: already present and verified\n' "$name" >&2
    fi

    files_present=$((files_present + 1))
    bytes_total=$((bytes_total + expected_bytes))
done

if [[ $verification_failed == true ]]; then
    echo "90_fetch_dsv4_0731_staging.sh: verification failed: one or more shards missing or invalid" >&2
    exit 1
fi

printf '{"ok":true,"files_present":%d,"bytes_total":%d,"destination":"%s"}\n' \
    "$files_present" "$bytes_total" "$destination"
