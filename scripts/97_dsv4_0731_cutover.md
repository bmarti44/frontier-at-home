# DeepSeek-V4-Flash-0731 cutover + rollback runbook

**Audience:** the owner (Brian), run manually during an announced maintenance
window. Nothing in this document is to be executed by an autonomous agent —
every step below either needs `sudo`, writes under `/home/dsv4`, or touches
the live systemd units, all of which are out of scope for staging agents per
this repo's guardrails.

**Preconditions before starting:**
- `results/dsv4-0731-staging/staging-identity.json` shows all UD-Q2_K_XL
  shards verified (sha256 match) and staged under
  `/home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl/`.
- `scripts/96_requalify_dsv4_0731.md`'s pre-window steps have run and
  produced a `results/decision-0731.json` you're satisfied with (or you are
  knowingly overriding it — record that the way
  `results/DECISION-OVERRIDE.md` did for the v0.4.2→llama.cpp override).
- No maintenance is happening on the box for anything else (`sol`'s
  campaigns, etc.) — coordinate before taking the production engine down.

## 0. Pre-window checklist

```bash
export DSV4_REPO=${DSV4_REPO:-/home/bmarti44/spark-deepseek-v4-flash}
cd "$DSV4_REPO"

# Confirm current production state and take a timestamped note of it
sudo systemctl status deepseek-v4-flash-llamacpp.service dsv4-authhelper.service dsv4-caddy.service dsv4-guard.timer
sudo -u dsv4 -H "$DSV4_REPO/scripts/21_serve_llamacpp.sh" status

# Confirm the staged 0731 shards are still verified (re-hash; don't trust a stale record)
bash scripts/90_fetch_dsv4_0731_staging.sh --verify-only

# Snapshot the current api-key so a rollback client re-key isn't a surprise
sudo cat /etc/deepseek-v4-flash/api-key | sha256sum   # record the hash only, never the key itself, in your notes
```

## 1. Swap procedure (production cutover)

Decide the target stack from `results/decision-0731.json` (mirrors how
`results/DECISION-OVERRIDE.md` picked llama.cpp over the frozen ds4 verdict
for the current deployment — the 0731 decision may or may not agree).

```bash
# Stop the guard first so it doesn't fight you
sudo systemctl stop dsv4-guard.timer

# Stop the currently running production engine unit
sudo systemctl stop deepseek-v4-flash-llamacpp.service   # or -ds4.service, whichever is active

# Move (not copy, to save 90 GiB) the verified staged shards into the
# dsv4-owned weights tree. Run this AS dsv4 or with an explicit chown after,
# since production directories are dsv4:dsv4 0700.
sudo -u dsv4 mkdir -p /home/dsv4/ds4-project/gguf-0731   # or the llamacpp-project equivalent path used by 21_serve_llamacpp.sh's MODEL_PATH default
sudo mv /home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-*.gguf \
    /home/dsv4/ds4-project/gguf-0731/
sudo chown dsv4:dsv4 /home/dsv4/ds4-project/gguf-0731/*.gguf
sudo chmod 444 /home/dsv4/ds4-project/gguf-0731/*.gguf

# Re-verify sha256 AFTER the move (catches any transfer corruption) against
# configs/pins/unsloth-ud-q2_k_xl-0731.json before pointing anything at it.
sudo -u dsv4 sha256sum /home/dsv4/ds4-project/gguf-0731/*.gguf
# compare each hash by hand (or a small owner-run script) against the pin file

# Point the serve wrapper's MODEL_PATH at the new shard (edit the systemd
# unit's Environment= block or configs/profiles/*.env, per docs/runbook.md's
# "Changing serving knobs or profiles" section — this requires the full
# conformance-test + agent-gate + soak sequence documented there, not just
# an env edit).

# Re-run the required profile-change gate sequence from docs/runbook.md:
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
scripts/dev/regression-suite.py agent-gate --base http://127.0.0.1:<production-port>
# If memory-relevant (a new, larger, or differently-quantized weight set is
# always memory-relevant): a 30-minute scripts/35_soak.py run, memwatch log reviewed.

# Rotate the API key (default behavior; add --keep-key only if you have a
# specific reason to keep serving the current key)
sudo "$DSV4_REPO/scripts/41_install_service.sh" llamacpp --acknowledge-decision-override
# (drop --acknowledge-decision-override if 0731's decision.json actually
# selects the frozen-verdict stack, i.e. no override is being made)

# Restart the managed units
sudo systemctl start deepseek-v4-flash-llamacpp.service dsv4-authhelper.service dsv4-caddy.service dsv4-guard.timer
```

**Re-key note:** `41_install_service.sh` rotates `/etc/deepseek-v4-flash/api-key`
on every run unless `--keep-key` is passed. Any client (laptops, scripts)
holding the old key loses access immediately on restart of
`dsv4-authhelper.service`. Deliver the new key to authorized clients
out-of-band before or immediately after cutover; do not commit it anywhere.

## 2. Post-swap verification smoke

```bash
# Confirm the engine itself is not exposed off-loopback
sudo "$DSV4_REPO/scripts/42_verify_exposure.sh"

# Confirm the auth-fronted endpoint answers with the new key
curl -sS -H "Authorization: Bearer $(sudo cat /etc/deepseek-v4-flash/api-key)" \
    https://<tailscale-or-caddy-host>/v1/models

# A single real completion, sanity-read the output
curl -sS -H "Authorization: Bearer $(sudo cat /etc/deepseek-v4-flash/api-key)" \
    -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Say ready."}],"max_tokens":16}' \
    https://<tailscale-or-caddy-host>/v1/chat/completions
```

If any of these fail, go straight to rollback (section 3) rather than
debugging live — restore service first, investigate on staged copies after.

## 3. Rollback to v0.4.2

The v0.4.2 weights and binary are untouched by this whole procedure (nothing
in the staging/requalification work modifies `weights/unsloth-ud-q2_k_xl/`
or the pinned v0.4.2 engine build) unless you deliberately deleted them in
step 1, which you should not have. Rollback is:

```bash
sudo systemctl stop dsv4-guard.timer deepseek-v4-flash-llamacpp.service

# Point the serve wrapper's MODEL_PATH / weights reference back at the
# original v0.4.2 shard set (revert the env/profile edit made in step 1;
# `git diff` the systemd unit / profile files under configs/ to confirm the
# revert is exact before restarting).
git -C "$DSV4_REPO" diff -- configs/systemd/ configs/profiles/
# git checkout the specific files if the diff doesn't match a clean revert

# Re-run the same mandatory gate sequence before trusting the rollback:
python3 -m unittest discover -s scripts/tests -p 'test_*.py'
scripts/dev/regression-suite.py agent-gate --base http://127.0.0.1:<production-port>

# Restart on the known-good v0.4.2 configuration
sudo systemctl start deepseek-v4-flash-llamacpp.service dsv4-authhelper.service dsv4-caddy.service dsv4-guard.timer

# Re-key again if you rotated in step 1 and want a clean boundary between
# the failed 0731 attempt and restored service
sudo "$DSV4_REPO/scripts/41_install_service.sh" llamacpp --acknowledge-decision-override
```

Verify with the same section-2 smoke checks. Record what went wrong in a new
`results/OPERATIONAL-OVERRIDE-<date>.md` or incident note, following the
existing `results/OPERATIONAL-OVERRIDE-2026-07-24.md` precedent, before
attempting cutover again.

## Rollback has NOT been executed or tested by this agent

Everything in this file is a written procedure only. No step here has been
run — the agent that authored this runbook does not have `sudo`, cannot
write to `/home/dsv4`, and was explicitly barred from touching the live
service. Before relying on this runbook during a real maintenance window,
the owner should dry-run the git-diff/revert steps against the actual
current `configs/systemd/` and `configs/profiles/` state, since this
document was written without being able to read those live files' exact
current content beyond what's visible in this repo checkout.
