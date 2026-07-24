# Technical review: round-2 hardening (guard circuit breaker, builder .so, overflow bounds)

Reviewing /home/bmarti44/spark-deepseek-v4-flash READ-ONLY. TECHNICAL CORRECTNESS +
SAFETY, NOT security-audit, NOT scored. This closes findings from the prior sol/max
hardening review; check the new fixes are correct and introduce no regressions,
especially the new root-run guard script.

## CRITICAL OUTPUT RULES (prior codex runs here were killed by an output filter — obey)
1. Never write a full 64-char hex digest; first 6 chars + ellipsis only.
2. Never cat/print digest-bearing files (MANIFEST.sha256, results/*.json, configs/build-manifests/*, configs/pins/*) wholesale; compare programmatically, print MATCH/MISMATCH + field.
3. Never paste diffs/patches/file contents from integrity experiments; one neutral sentence each.

## DO NOT
- Do not modify/create/delete tracked files (scratch under /tmp only).
- Do not contact 127.0.0.1:8010-8014; do not start any server/build/model. Read code only.

## Scope: commits `3efe711..HEAD` (ae63360 funnel fix, acbad89 round-2 hardening)
- scripts/03_guard.sh (NEW, runs as ROOT from dsv4-guard.service): consecutive-failure circuit breaker in /run/dsv4/guard-consecutive-failures; latches after 3; resets on healthy.
- configs/systemd/dsv4-guard.service: ExecStart now the script (was inline bash).
- configs/systemd/deepseek-v4-flash-llamacpp.service: StartLimit comment corrected.
- scripts/21_serve_llamacpp.sh: shared_libraries now REQUIRED non-empty; DSV4_BATCH/UBATCH digit-bounded (<=5).
- scripts/13_build_llamacpp.sh: build manifest now records shared_libraries (hashes build/bin *.so).
- scripts/00_preflight.sh: DSV4_MIN_ROOT_FREE_GIB digit-bounded (<=6).
- scripts/42_verify_exposure.sh (ae63360): recursive Funnel detection (Foreground sessions).
- docs/phase-b-fusion + runbook: corrections.

## Review, each with file:line + concrete failure scenario, tagged [critical]/[high]/[medium]/[low]:
1. **Guard circuit breaker (scripts/03_guard.sh) — this runs as root, scrutinize hardest.** Is the latch logic correct and free of races? Consider: the restart is BLOCKING (~600s) while the timer is OnUnitActiveSec=60s — can two guard runs overlap and double-count or corrupt the counter file (no locking)? Is the counter reset-on-health correct, and can a transient healthy blip reset a real failure loop? After the breaker latches (exit 1), does the guard oneshot entering `failed` stop the TIMER from firing again (it should keep firing and keep hitting the OPEN branch — confirm it does NOT restart)? Does `set -Eeuo pipefail` + the `runuser status` nonzero exit behave (status returning nonzero must NOT abort the script before the breaker logic)? STACK/paths/`$COUNTER` quoting; /run/dsv4 writability under the unit sandbox (ReadWritePaths); SCRIPT_DIR resolution as root. Any way it fails-open (keeps restarting forever) or fails-closed wrongly (never restarts a recoverable engine)?
2. **guard.service wiring.** Does removing the inline bash for `ExecStart=@DSV4_REPO@/scripts/03_guard.sh` preserve behavior? Is the script executable/mode correct for systemd? Does EnvironmentFile still supply STACK? @DSV4_REPO@ expansion at install.
3. **.so required + builder emission.** Serve now hard-requires non-empty shared_libraries — does any legitimate path (baseline manifest, fusion manifest, freshly-built manifest) now fail? Does 13_build_llamacpp.sh correctly enumerate/hash exactly the loaded libs (symlinks vs versioned reals — the serve check hashes `bin/<name>` following symlinks; does the builder record the same `<name>` set)? Any mismatch between what the builder records and what serve looks for?
4. **Overflow-safe bounds.** Are the digit caps (batch/ubatch <=5, disk <=6) correct and sufficient to prevent Bash 64-bit arithmetic wrap, while still allowing all intended values? Any remaining bypass?
5. **Funnel recursion (42).** Is the recursive AllowFunnel/Foreground check correct and complete vs Tailscale's IsFunnelOn? Any structure it still misses?
6. **Docs now accurate?** Do phase-b + runbook now correctly describe the breaker (count-based), the repo-landed-not-deployed status, and the recovery steps? Any remaining overstatement?

## Output
Per area: `## <area> — <verdict>` then tagged bullets (file:line + scenario) or explicit justification if clean. End with `## Summary`: remaining GPU-free issues ranked, and a one-line judgement on whether this round-2 hardening is correct and safe. Note sandbox limits; reason from code where blocked.
