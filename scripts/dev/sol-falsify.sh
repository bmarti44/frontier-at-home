#!/usr/bin/env bash
# sol-falsify.sh <slug> <minutes> — adversarial falsification review of ONE
# finding. Reads the prompt on stdin, writes the verdict to
# results/glm52-gates/audit-2026-07-26/<slug>.md
set -Eeuo pipefail
SLUG="$1"; MINS="${2:-25}"
REPO=/home/bmarti44/spark-deepseek-v4-flash
OUT="$REPO/results/glm52-gates/audit-2026-07-26/$SLUG.md"
TMP=$(mktemp)
cat > "$TMP"
bash "$REPO/scripts/dev/sol-review.sh" xhigh "$MINS" "$OUT.raw" < "$TMP"
{
  echo "# Adversarial falsification review: $SLUG"
  echo
  echo "Task: sol xhigh was asked to PROVE THIS CLAIM FALSE, not to review it."
  echo "Generated $(date -Is)."
  echo
  sed -n '/---FINAL---/,$p' "$OUT.raw" | tail -n +2
} > "$OUT"
rm -f "$OUT.raw" "$TMP"
echo "WROTE $OUT"
