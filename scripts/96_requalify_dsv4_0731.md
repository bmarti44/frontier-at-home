# DeepSeek-V4-Flash-0731 requalification plan

Status: DRAFT — staging + planning only. No engine has been launched under
this plan. See `results/dsv4-0731-staging/staging-identity.json` for the
verified shard inventory this plan targets.

## Why this document exists

The current production deployment (`docs/runbook.md`, `results/DECISION-OVERRIDE.md`)
serves **llama.cpp** with the `unsloth/DeepSeek-V4-Flash-GGUF` UD-Q2_K_XL
weights on the product override path, with ds4 (DSpark profile) parked as the
frozen-benchmark winner for the ≤28K envelope. Upgrading to
**unsloth/DeepSeek-V4-Flash-0731-GGUF** UD-Q2_K_XL means the whole comparison
in `results/decision.json` / `results/DECISION.md` is stale for the new
weights and must be rerun end-to-end before any cutover: golden correctness,
token parity, accuracy (gsm8k/mmlu-pro/humaneval, dev+holdout), speed (both
stacks), soak, the agent gate, and a fresh `34_decision.py` verdict.

This plan mirrors the v0.4.2 qualification stack script-for-script, but:
- targets the staged 0731 weights at `/home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl/`
  (see staging-identity.json), never `/home/dsv4`.
- uses non-production ports (never 8011, never the auth-helper's 8014).
- runs the engine only through the repo's hardened wrapper pattern
  (`results/glm52-gates/harness/glm_safe_run.sh`), which enforces a
  100+ GiB start-memory floor, an 18 GiB kill floor, a 400 GiB `ulimit -v`
  backstop, and a wall-clock timeout — never a bare `ds4-server` /
  `llama-server` invocation.
- checks engine/fio exclusivity and a resource floor before every launch
  (see "Preflight" below), because another agent (sol) runs build/freeze/review
  campaigns with exclusive-evidence requirements on this same box.

## Inventory of the existing qualification pipeline (recon)

| Stage | Script(s) | Produces |
|---|---|---|
| Host preflight | `scripts/00_preflight.sh` | disk/mem/driver/kernel/ollama/swap checks |
| Engine + weights fetch | `scripts/10_fetch_ds4.sh`, `scripts/12_fetch_gguf.sh`, `scripts/14_fetch_encoder.sh` | pinned, sha256-verified binaries/weights |
| Eval datasets | `scripts/16_fetch_evalsets.py` | `evalsets/*.jsonl` + `evalsets/pins.json` |
| Build | `scripts/11_build_ds4.sh`, `scripts/13_build_llamacpp.sh` | pinned-commit binaries |
| Serve (production wrapper) | `scripts/20_serve_ds4.sh`, `scripts/21_serve_llamacpp.sh` | loopback OpenAI-compatible server under `/run/dsv4` state |
| Speed | `scripts/30_bench_speed.py` | `results/speed-<stack>[-<profile>].json` (context sweep, TTFT + decode tok/s) |
| Accuracy | `scripts/31_bench_accuracy.py` | `results/acc-<suite>-<split>-<stack>.json` |
| Golden correctness | `scripts/32_golden_tests.py` | `results/golden-<stack>[-<profile>].json` |
| Token parity | `scripts/33_token_parity.py` | `results/parity-<stack>.json` |
| Decision | `scripts/34_decision.py` | `results/decision.json`, `results/DECISION.md` |
| Soak | `scripts/35_soak.py` | `results/soak-<stack>.json` (30 min, 10 gates, all must pass) |
| Accuracy re-audit | `scripts/36_audit_accuracy.py` | `results/audit-<stack>.json` (independent re-scoring, binds transcript/evalset hashes) |
| Auth | `scripts/40_auth_helper.py` | bearer-token proxy in front of the loopback engine |
| Service install | `scripts/41_install_service.sh` | systemd units, Caddy, Tailscale Serve, API key rotation |
| Exposure check | `scripts/42_verify_exposure.sh` | confirms the engine itself is not reachable off-loopback |
| Agent gate | `scripts/dev/regression-suite.py agent-gate` | `results/agent-gate-<date>.json` (prefix-cache, turn-continuation, slot-thrash, prefill-throughput) |
| Safe engine wrapper | `results/glm52-gates/harness/glm_safe_run.sh` | hardened process-group isolation for ANY engine invocation |

Ports observed: production upstream engine 8011 (guardrail: never reuse),
auth-helper public listen 8014, ds4-dspark speed test used 8012,
agent-gate probe used 8013. **This plan reserves 8021-8029 for all 0731
requalification engine instances** to avoid any collision with production or
with `sol`'s concurrent work.

## What can run pre-maintenance-window (against staged weights, non-production port)

All of the following are mechanical and require only the staged, verified
0731 shards plus a non-production loopback server — they do **not** touch
`/home/dsv4` or any existing systemd unit:

1. Build/obtain a serving binary capable of loading the 0731 GGUF (the
   pinned llama.cpp commit in `configs/pins/*` should load it unmodified,
   since it's the same UD-Q2_K_XL family; the pinned ds4 commit
   `baa889025b16a7060f5f854226cb0d14e260eb52` needs verification against
   the 0731 GGUF's `deepseek4` architecture tag before assuming compatibility).
2. Golden correctness (`32_golden_tests.py`) against the 0731 server.
3. Token parity (`33_token_parity.py`) — note the tokenizer is pinned at
   `vendor/official-encoding/tokenizer.json`; if 0731's chat template or
   vocabulary differs (the model card shows an updated `chat_template` with
   `｜DSML｜` tool-call tokens, `reasoning_effort` knobs, and `<think>` tags
   not present in the v0.4.2 template), this step must be re-validated
   against the 0731 tokenizer/template, not silently reused.
4. Speed sweep (`30_bench_speed.py`), same context levels `0,4096,16384,28672`.
5. Accuracy suites, dev+holdout, both stacks (`31_bench_accuracy.py`) —
   **must use fresh holdout draws**; `results/holdout-ledger.json` tracks
   which examples have been spent, and reusing spent holdout rows against
   a new model voids the exercise.
6. Independent accuracy audit (`36_audit_accuracy.py`).
7. Soak (`35_soak.py`, 30 minutes, all 10 gates).
8. Agent gate (`scripts/dev/regression-suite.py agent-gate`).
9. Fresh decision run (`34_decision.py`) producing a 0731-specific
   `results/decision-0731.json` / `DECISION-0731.md` (do not overwrite the
   existing frozen v0.4.2 decision artifacts — guardrail 5).

## What requires the owner's maintenance window (touches `/home/dsv4` or live service)

- Copying/linking the verified 0731 shards into `/home/dsv4`'s weights tree
  (or repointing `MODEL_PATH`/`DS4_HOME` env at the staging path, if the
  owner prefers not to duplicate ~90 GiB).
- Stopping the currently-installed production unit
  (`deepseek-v4-flash-llamacpp.service` per the current override) and
  installing/pointing it at the 0731 weights.
- API key rotation and re-keying of authorized clients (`41_install_service.sh`
  rotates by default; see cutover runbook).
- Anything using `sudo`, `systemctl`, or writing under `/home/dsv4`,
  `/etc/deepseek-v4-flash`, or `/run/dsv4`.
- Final `results/decision-0731.json` sign-off gating whether 0731 actually
  replaces v0.4.2, versus being parked like ds4 currently is.

## Preflight required before ANY 0731 engine launch (pre- or in-window)

Run in this order; abort and do not launch if any check fails:

```bash
# 1. Exclusivity — no other engine or heavy build process running
pgrep -fl 'ds4-server|llama-server|fio' && echo "ABORT: another engine/fio process is running" && exit 1

# 2. Memory floor — at least 10 GiB MemAvailable (this plan's floor; the
#    production wrapper itself additionally enforces its own 100+ GiB
#    start floor and 18 GiB kill floor once launched)
awk '/MemAvailable/{if ($2/1024/1024 < 10) { print "ABORT: MemAvailable below 10 GiB"; exit 1 }}' /proc/meminfo

# 3. Host preflight (disk/driver/kernel/ollama/swap)
scripts/00_preflight.sh --out /tmp/preflight-0731.json
```

Then launch only through the safe wrapper, e.g.:

```bash
sudo -u dsv4 bash results/glm52-gates/harness/glm_safe_run.sh --tag dsv4-0731-smoke -- \
  <engine-binary> --port 8021 -m /home/bmarti44/models/dsv4-flash-0731-ud-q2k-xl/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf ...
```

(If the wrapper's `sudo -u dsv4` path is unavailable to a non-owner operator,
use `GLM_SAFE_RUN_AS_CURRENT_USER=1` under the logged-in benchmark owner per
the wrapper's own header comment — never invoke the engine binary directly,
unwrapped.)

## DSpark drafter: flagged, unresolved

`configs/pins/unsloth-ud-q2_k_xl-0731.json`'s `dspark_drafter_note` records
the finding: v0.4.2's DSpark profile used a **third-party** drafter
(`bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`, Q2K-Q8 quant). The 0731 GGUF
repo ships its **own** first-party drafter files
(`dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, ~10.9 GiB, and
`dspark/dspark-DeepSeek-V4-Flash-0731-BF16.gguf`, ~11.3 GiB) directly under
`unsloth/DeepSeek-V4-Flash-0731-GGUF`. The old bleysg drafter file is
**not** applicable to 0731 — a drafter's proposal distribution must track the
target model's weights closely enough for a useful acceptance rate, and
0731 is a different base checkpoint (`deepseek-ai/DeepSeek-V4-Flash-0731`,
not the original `DeepSeek-V4-Flash`). Before any DSpark-profile speed run
against 0731:
1. Fetch + sha256-verify the first-party `Q8_0` drafter (not staged by this
   task — out of the UD-Q2_K_XL base-weights scope given).
2. Confirm the pinned `ds4` engine commit can load it (architecture tag,
   drafter-format compatibility).
3. Re-run the DSpark acceptance-rate sanity check before trusting any DSpark
   speed numbers — do not assume the v0.4.2 DSpark speed profile
   (`results/speed-ds4-dspark.json`) transfers.
4. If incompatible, ds4 requalification must fall back to `plain` or `mtp`
   profile only, and that must be stated explicitly in the 0731 decision doc.

## Exit criteria

This plan is "done" (0731 ready for a cutover decision) when
`results/decision-0731.json` exists with the same structure as
`results/decision.json`, both candidate stacks' `eligible: true` (or a
recorded, justified ineligibility), and a soak + agent-gate pass recorded
for whichever stack is proposed for production. Until then, 0731 remains
staged-only and the owner's maintenance-window steps in
`scripts/97_dsv4_0731_cutover.md` are not to be executed.
