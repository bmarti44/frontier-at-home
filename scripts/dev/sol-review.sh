#!/usr/bin/env bash
# sol-review.sh — hardened codex/sol caller after repeated silent startup
# hangs. Streams --json events so progress is observable, watchdogs the
# stream (kill+retry if quiet too long), resumes the session on retry.
#   sol-review.sh <effort> <max_minutes> <outfile> < prompt.txt
set -u
EFFORT=${1:-xhigh}; MAXMIN=${2:-30}; OUTFILE=${3:-/tmp/sol-out.txt}
PROMPT=$(cat)
EV=$(mktemp /tmp/sol-events-XXXX.jsonl)
run_once() { # $1 = "new"|"resume"
  local args=(exec --json --sandbox read-only -c model_reasoning_effort=$EFFORT -o "$OUTFILE")
  if [[ $1 == resume ]]; then
    codex exec resume --last --json --sandbox read-only -o "$OUTFILE" - <<<"Continue: finish the review you were asked to do and print the final verdict." >>"$EV" 2>&1 &
  else
    codex "${args[@]}" - <<<"$PROMPT" >>"$EV" 2>&1 &
  fi
  local pid=$! quiet=0 total=0
  local last_size=0
  while kill -0 $pid 2>/dev/null; do
    sleep 15; total=$((total+15))
    local size=$(stat -c%s "$EV" 2>/dev/null || echo 0)
    if [[ $size -eq $last_size ]]; then quiet=$((quiet+15)); else quiet=0; last_size=$size; fi
    if [[ $quiet -ge 240 ]]; then echo "SOL_STALL after ${total}s (no events 240s)"; kill $pid 2>/dev/null; return 9; fi
    if [[ $total -ge $((MAXMIN*60)) ]]; then echo "SOL_MAXTIME"; kill $pid 2>/dev/null; return 8; fi
  done
  wait $pid; return $?
}
run_once new; rc=$?
if [[ $rc == 9 ]]; then echo "retrying via resume --last"; run_once resume; rc=$?; fi
if [[ $rc == 9 ]]; then echo "second stall; fresh retry"; run_once new; rc=$?; fi
echo "SOL_RC=$rc events=$(wc -l < "$EV")"
[[ -s "$OUTFILE" ]] && { echo "---FINAL---"; cat "$OUTFILE"; }
exit $rc
