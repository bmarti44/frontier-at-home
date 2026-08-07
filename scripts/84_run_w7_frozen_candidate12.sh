#!/usr/bin/env -S -i HOME=/home/bmarti44 PATH=/usr/bin:/bin /usr/bin/bash --noprofile --norc
# Independent launcher for the reviewed W7 candidate. The candidate cannot
# nominate its own commit or digest: both constants live outside its blob.
set -Eeuo pipefail
umask 077

readonly REPO=/home/bmarti44/spark-deepseek-v4-flash
readonly CANDIDATE_COMMIT=d327d63bee1a93319a05f8f9b467edbae0cf49d8
readonly HARNESS_SHA256=e9521f2b096bee1803f5d65fad81da6d473ce582fdc79c82433fcff1cecc3b45
readonly HARNESS_PATH=results/glm52-gates/harness/w7_resume_compiled_red_v1.sh

[[ $(/usr/bin/id -un) == bmarti44 ]] || exit 2
/usr/bin/git -C "$REPO" cat-file -e "$CANDIDATE_COMMIT^{commit}"
staged=$(/usr/bin/mktemp /tmp/glm52-w7-frozen.XXXXXX)
cleanup() { /usr/bin/rm -f -- "$staged"; }
trap cleanup EXIT
/usr/bin/git -C "$REPO" show "$CANDIDATE_COMMIT:$HARNESS_PATH" >"$staged"
/usr/bin/chmod 0400 "$staged"
exec {harness_fd}<"$staged"
read -r actual_sha256 _ < <(/usr/bin/sha256sum -- "/proc/$$/fd/$harness_fd")
[[ $actual_sha256 == "$HARNESS_SHA256" ]] || exit 2
/usr/bin/rm -f -- "$staged"
trap - EXIT

if [[ ${1:-} == --self-test ]]; then
  [[ $# == 1 ]] || exit 2
  /usr/bin/env -i \
    HOME=/home/bmarti44 PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    W7_EXECUTED_HARNESS_SHA256="$HARNESS_SHA256" \
    W7_FROZEN_CANDIDATE_COMMIT="$CANDIDATE_COMMIT" \
    /usr/bin/bash "/proc/$$/fd/$harness_fd" \
      --validate-execution-authority "$HARNESS_SHA256" "$CANDIDATE_COMMIT"
  echo W7_FROZEN_LAUNCHER_OK
  exit 0
fi

[[ $# == 0 ]] || exit 2
exec /usr/bin/env -i \
  HOME=/home/bmarti44 USER=bmarti44 LOGNAME=bmarti44 \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  W7_EXECUTED_HARNESS_SHA256="$HARNESS_SHA256" \
  W7_FROZEN_CANDIDATE_COMMIT="$CANDIDATE_COMMIT" \
  /usr/bin/bash "/proc/$$/fd/$harness_fd"
