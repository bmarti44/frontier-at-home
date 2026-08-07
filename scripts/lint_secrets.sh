#!/usr/bin/env bash
set -Eeuo pipefail

readonly SECRET_PATTERN='[0-9a-f]{64}|Bearer [A-Za-z0-9._-]{20,}|BEGIN( RSA| OPENSSH)? PRIVATE KEY|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|tskey-[A-Za-z0-9-]{20,}'
# Digest-bearing files get field/format-aware 64-hex validation. They still get
# every other secret pattern via SECRET_PATTERN_NOHEX.
readonly SECRET_PATTERN_NOHEX='Bearer [A-Za-z0-9._-]{20,}|BEGIN( RSA| OPENSSH)? PRIVATE KEY|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,}|tskey-[A-Za-z0-9-]{20,}'
readonly PUBLIC_DIGEST_ALLOWLIST='^scripts/65_glm52_w1_submit\.py:[0-9]+:P1_PYTHON_DEPENDENCY_SHA256 = "[0-9a-f]{64}"$|^scripts/66_install_glm52_w1_attestor\.sh:[0-9]+:readonly (SUBMITTER|PYTHON_DEPENDENCY)_SHA256=[0-9a-f]{64}$|^scripts/71_install_glm_benchmark_lock_acl\.sh:[0-9]+:readonly SOURCE_SHA256=[0-9a-f]{64}$|^scripts/73_run_glm_shared_router_probe\.py:[0-9]+:(MODEL|TOKENIZER)_SHA256 = "[0-9a-f]{64}"$|^scripts/76_run_glm_union_trace_smoke\.py:[0-9]+:QUALITY_FIXTURE_CONTENT_SHA256 = "[0-9a-f]{64}"$|^scripts/81_glm_union_baseline\.py:[0-9]+:FROZEN_(BINARY|MODEL|TOKENIZER|FIXTURE|ROOT_SUBMITTER)_SHA256 = "[0-9a-f]{64}"$|^scripts/81_glm_union_baseline\.py:[0-9]+:    (PROBE_PATH|CV_PATH|PRECISION_PATH|SAFE_RUN_PATH|MEMORY_GUARD_PATH): "[0-9a-f]{64}",$|^scripts/glm52_goal\.py:[0-9]+:_W7_SAFE_WRAPPER_SHA256 = "[0-9a-f]{64}"$|^scripts/tests/test_glm52_goal\.py:[0-9]+:                "wrapper_sha256": "[0-9a-f]{64}",$|^results/glm52-gates/harness/w3_direct_slot_probe_v[123]\.sh:[0-9]+:readonly (BINARY|MODEL)_SHA256=[0-9a-f]{64}$|^results/glm52-gates/RUNG-PLAN\.md:[0-9]+:  `[0-9a-f]{64}`[.;]$|^results/glm52-gates/RUNG-PLAN\.md:[0-9]+:SHA-256 `[0-9a-f]{64}`\.$|^scripts/tests/test_glm_rung0_slab_campaign\.py:[0-9]+:            "[0-9a-f]{64}",$|^scripts/tests/test_glm_mtp_proxy_seed_contract\.py:[0-9]+:(EXPECTED = |    "(manifest|drand)": )"[0-9a-f]{64}"[,]?$|^results/glm52-gates/harness/0012-test-glm-reject-compact-pack-identity-mutations\.patch:[0-9]+:\+                "[0-9a-f]{64}"\);$|^results/glm52-gates/harness/0015-test-require-exact-frozen-MTP-proxy-pack\.patch:[0-9]+:[+-](    parse_digest\((pack|model|inventory)_sha, |                )"[0-9a-f]{64}"\);$|^results/glm52-gates/harness/0016-test-require-canonical-crash-safe-proxy-publication\.patch:[0-9]+:\+        "\{\\"bytes\\":[0-9]+,\\"name\\":\\"[a-z0-9.-]+\\",\\"sha256\\":\\"[0-9a-f]{64}\\"\}[,]?"$|^results/glm52-gates/harness/0020-test-close-round-194-MTP-proxy-gaps\.patch:[0-9]+:[ +-](                 |        )("[0-9a-f]{64}"\);|"\{\\"bytes\\":[0-9]+,\\"name\\":\\"[a-z0-9.-]+\\",\\"sha256\\":\\"[0-9a-f]{64}\\"\}[,]?"?)$'
readonly W3_PUBLIC_DIGEST_ALLOWLIST='^results/glm52-gates/W3-slot-lifetime-probe-v11-pass/crash/(off|on)/cmd\.log:[0-9]+:ds4: expert-cache window tag=[^ ]+ lookup_bytes=[0-9]+ hit_bytes=[0-9]+ stream_sha256=[0-9a-f]{64}$|^results/glm52-gates/W3-slot-lifetime-probe-v11-pass/crash/(off|on)/main\.log:[0-9]+:[0-9T:+,.-]+ (candidate_src=[^ ]+ candidate_binary_sha256=[0-9a-f]{64} candidate_device_inode=[0-9:]+|executed_environment_allowlist=[A-Z0-9_,]+ executed_environment_sha256=[0-9a-f]{64}|executed_candidate_verified pid=[0-9]+ start_ticks=[0-9]+ path=[^ ]+ executed_binary_sha256=[0-9a-f]{64} device_inode=[0-9:]+|safety_artifact_verified name=(samples|kernel)\.log sha256=[0-9a-f]{64} size=[0-9]+)$'
readonly W7_PUBLIC_DIGEST_ALLOWLIST='^scripts/glm52_goal\.py:[0-9]+:_W7_(STEM_FILE|STEM_TEXT|POOL|TOKENIZER|TOKENIZER_INIT|TOKENIZER_NATIVE|SERVER_SOURCE|RENDER_ORACLE_SOURCE|RENDER_ORACLE_BINARY)_SHA256 = "[0-9a-f]{64}"$|^scripts/82_build_w7_fixture_pool\.py:[0-9]+:(TOKENIZER|RUNTIME_INIT|RUNTIME_NATIVE|SERVER_SOURCE|ORACLE_SOURCE|ORACLE_BINARY)_SHA256 = "[0-9a-f]{64}"$|^scripts/82_build_w7_render_oracle\.sh:[0-9]+:readonly (CC|SERVER|SOURCE|OUTPUT)_SHA256=[0-9a-f]{64}$|^scripts/83_score_w7_deployed_trace\.py:[0-9]+:TOKENIZER_(SHA256|INIT_SHA256|NATIVE_SHA256) = "[0-9a-f]{64}"$|^scripts/84_run_w7_frozen_candidate12\.sh:[0-9]+:(readonly HARNESS_SHA256=[0-9a-f]{64}|HARNESS_SHA256 = "[0-9a-f]{64}")$|^scripts/tests/test_glm52_goal\.py:[0-9]+:                "[0-9a-f]{64}",$|^scripts/tests/test_w7_resume_compiled_red\.py:[0-9]+:                "[0-9a-f]{64}",$|^results/glm52-gates/harness/w7_resume_compiled_red_v1\.sh:[0-9]+:readonly (POOL|BINARY|MODEL|RENDER_ORACLE|TRACE_SCORER|TOKENIZER|TOKENIZER_INIT|TOKENIZER_NATIVE)_SHA256=[0-9a-f]{64}$'

is_checksum_file() {
  case "$1" in
    verification/MANIFEST.sha256|configs/versions.lock|configs/glm52-profile.json|configs/dsv4-profile.json|configs/pins/*|configs/build-manifests/*|evalsets/pins.json|results/transcripts/*|results/acc-*.json|results/audit-*.json|results/speed-*.json|results/decision.json|results/holdout-ledger.json|results/glm52-gates/G6-rung0-io-sidecar-build.json|results/glm52-gates/G6-rung0-io-slab-calibration-no-results.json|results/glm52-gates/G6-rung0-io-accelerated-sha-falsifier.json|results/glm52-gates/R0-slab-canary-attempts-2026-08-02.json|results/glm52-gates/R0-e637-campaign-attempt-2026-08-02.json|results/glm52-gates/R0-e637-quality-timeout-attempt-2026-08-03.json|results/glm52-gates/R0-e637-slab-final-2026-08-03.json|results/glm52-gates/R0.2-prefetch-build-freeze-2026-08-03.json|results/glm52-gates/R0.2-prefetch-build-*.json|results/glm52-gates/R0.2-prefetch-freeze-*.json|results/glm52-gates/R0.2-prefetch-randomness-*.json|results/glm52-gates/R0.2-prefetch-probe-*-attempt-*.json|results/glm52-gates/R0.2-prefetch-probe-6885a45-final-2026-08-04.json|results/glm52-gates/R0[abc]-*.json|results/glm52-gates/W3-slot-gemv-*.json|results/glm52-gates/W3-slot-lifetime-*.json|results/glm52-gates/W3-performance-*.json|results/glm52-gates/NVME-characterization-attempt-*.json|results/glm52-gates/NVME-characterization-final-*.json|results/glm52-goal/evidence/roofline-*.json|results/glm52-goal/evidence/*-confirmation-*.json|results/glm52-goal/evidence/build-repro/*/*.json|results/glm52-goal/evidence/dsv4-decode-*/*.json|results/glm52-goal/evidence/glm-diagnostic-*/manifest.json|results/glm52-goal/evidence/glm-diagnostic-*/*/*.json|results/glm52-goal/evidence/glm-diagnostic-*/*/*.log|results/glm52-goal/evidence/glm-diagnostic-*/success/process.identity|results/glm52-goal/evidence/w1-affine-*/manifest.json|results/glm52-goal/evidence/w1-affine-*/raw.jsonl|results/glm52-goal/evidence/w1-affine-*/raw-inputs/randomness.json|results/glm52-goal/evidence/w1-telemetry-probe-*/manifest.json|results/glm52-goal/evidence/w1-telemetry-probe-*/raw.jsonl|results/glm52-goal/*/attempt-*/manifest.json|results/glm52-goal/*/attempt-*/raw.jsonl|weights/*/manifest.json|*.sha256) return 0 ;;
    results/glm52-gates/W3-performance-campaign-*/raw.jsonl|results/glm52-gates/R0.5-*.json|results/glm52-gates/W7-resume-correctness-plan-v*.json|results/glm52-gates/W7-resume-candidate*-red.json|results/glm52-gates/W7-resume-review-r*.json|results/glm52-gates/W7-resume-build-attempt-v*.json|results/glm52-gates/W7-resume-compiled-red-freeze-v*.json|results/glm52-gates/W7-resume-restored-frontier-freeze-v*.json|results/glm52-gates/W7-resume-compiled-red-attempt*.json|results/glm52-gates/W7-resume-smoke-randomness-v*.json|results/glm52-gates/harness/w7-production-fixture-pool-v1.json) return 0 ;;
    *) return 1 ;;
  esac
}

# Split a NUL-delimited file list (stdin) into the two scan tiers.
split_files() {
  # sets globals: files_full, files_nohex
  files_full=()
  files_nohex=()
  local f
  while IFS= read -r -d '' f; do
    if is_checksum_file "$f"; then
      files_nohex+=("$f")
    else
      files_full+=("$f")
    fi
  done
}

gitleaks_binary() {
  if command -v gitleaks >/dev/null 2>&1; then
    command -v gitleaks
  elif [[ -x bin/gitleaks ]]; then
    printf '%s\n' 'bin/gitleaks'
  fi
}

require_gitleaks() {
  local binary
  binary="$(gitleaks_binary || true)"
  if [[ -z "$binary" ]]; then
    printf '%s\n' \
      'gitleaks is required; install it from https://github.com/gitleaks/gitleaks#installing or place the pinned binary at bin/gitleaks' >&2
    return 1
  fi
  printf '%s\n' "$binary"
}

redact_matches() {
  sed -E \
    -e 's/([0-9a-f]{6})[0-9a-f]{58}/\1[REDACTED]/g' \
    -e 's/(Bearer)[ A-Za-z0-9._-]{21,}/\1[REDACTED]/g' \
    -e 's/(BEGIN )(RSA |OPENSSH )?PRIVATE KEY/\1[REDACTED]/g' \
    -e 's/(hf_[A-Za-z0-9]{3})[A-Za-z0-9]{27,}/\1[REDACTED]/g' \
    -e 's/(sk-[A-Za-z0-9]{3})[A-Za-z0-9]{17,}/\1[REDACTED]/g' \
    -e 's/(tskey-)[A-Za-z0-9-]{20,}/\1[REDACTED]/g'
}

scan_stream() {
  local matches
  matches="$(grep -E "$SECRET_PATTERN" | grep -Ev "$PUBLIC_DIGEST_ALLOWLIST|$W3_PUBLIC_DIGEST_ALLOWLIST|$W7_PUBLIC_DIGEST_ALLOWLIST" || true)"
  if [[ -n "$matches" ]]; then
    printf '%s\n' "$matches" | redact_matches >&2
    return 1
  fi
}

scan_digest_json() {
  local display_path="$1"
  python3 - "$display_path" 3<&0 <<'PY'
import json
import os
import re
import sys

display_path = sys.argv[1]
allowlist = {
    "sha256",
    "source_parquet_sha256",
    "oid",
    "git_oid_sha1",
    "rendered_prompt_sha256",
    "commit",
    "revision",
    "config_digest",
    "sha1",
    "server_binary_sha256",
    "model_manifest_sha256s",
    "model_manifest_sha256",
    "diagnostic_events_sha256",
    "runtime_start_sha256",
    "run_log_sha256",
    "canonical_sha256",
    "canonical_tree_sha256",
    "source_summary_sha256",
    "source_receipt_sha256",
    "source_transcript_sha256",
    "tokenizer_sha256",
    "tsa_certificate_sha256",
    "binary_sha256",
    "executed_binary_sha256",
    "build_manifest_sha256",
    "ca_certificate_sha256",
    "configuration_sha256",
    "source_commit_object_sha256",
    "production_binary_sha256",
    "candidate_binary_sha256",
    "quality_binary_sha256",
    "metric_scorer_sha256",
    "metric_scorer_tests_sha256",
    "tests_sha256",
    "randomness_sha256",
    "raw_jsonl_sha256",
    "raw_sha256",
    "reference_continuation_sha256",
    "reference_token_ids_sha256",
    "manifest_sha256",
    "stage_receipt_sha256",
    "all_arm_tsv_sha256",
    "arm_sha256",
    "off_arm_sha256",
    "on_arm_sha256",
    "off_containment_sha256",
    "on_containment_sha256",
    "off_containment_stdout_sha256",
    "off_containment_stderr_sha256",
    "off_result_sha256",
    "on_result_sha256",
    "off_result_1_sha256",
    "off_result_2_sha256",
    "on_result_1_sha256",
    "on_result_2_sha256",
    "off_ledger_sha256",
    "on_ledger_sha256",
    "off_responses_sha256",
    "on_responses_sha256",
    "manifest_sha256",
    "stdout_manifest_sha256",
    "splitter_sha256",
    "source_manifest_sha256",
    "source_output_sha256",
    "split_plan_sha256",
    "top_manifest_sha256",
    "quality_output_sha256",
    "long_output_sha256",
    "off_server_log_sha256",
    "on_server_log_sha256",
    "nll_sha256",
    "summary_sha256",
    "runtime_final_sha256",
    "runtime_init_sha256",
    "runtime_native_sha256",
    "checkpoint_chain_tail_sha256",
    "console_log_sha256",
    "generated_content_sha256",
    "generated_reasoning_sha256",
    "generated_sha256",
    "diff_sha256",
    "patch_sha256",
    "engine_source_sha256",
    "environment_sha256",
    "engine_test_sha256",
    "generator_sha256",
    "pool_sha256",
    "test_sha256",
    "production_cuda_source_sha256",
    "stdout_sha256",
    "stderr_sha256",
    "ds4_cuda_sha256",
    "state_header_sha256",
    "state_header_engine_copy_sha256",
    "acceptance_test_sha256",
    "build_1_binary_sha256",
    "build_2_binary_sha256",
    "fixture_sha256",
    "freeze_json_sha256",
    "freeze_sha256",
    "harness_sha256",
    "harness_tests_sha256",
    "fio_result_sha256",
    "model_sha256",
    "sidecar_sha256",
    "slab_on_configuration_sha256",
    "output_tokenizer_sha256",
    "profile_sha256",
    "public_randomness",
    "randomness",
    "signature",
    "request_sha256",
    "response_sha256",
    "result_sha256",
    "output_sha256",
    "scorer_sha256",
    "unchanged_scorer_sha256",
    "red_test_sha256",
    "compactor_sha256",
    "scorer_tests_sha256",
    "source_contract_tests_sha256",
    "server_sha256",
    "server_log_sha256",
    "containment_stdout_sha256",
    "nvme_inflight_sha256",
    "safety_samples_sha256",
    "partial_tsv_sha256",
    "main_log_sha256",
    "command_log_sha256",
    "engine_log_sha256",
    "access_stream_sha256",
    "raw_artifact_sha256",
    "frozen_target_manifest_sha256",
    "target_sha256_before",
    "target_sha256_after",
    "wrapper_log_sha256",
    "samples_log_sha256",
    "kernel_log_sha256",
    "installer_sha256",
    "submitter_sha256",
    "controller_sha256",
    "approval_sha256",
    "reservation_sha256",
    "marker_sha256",
    "failed_tree_manifest_sha256",
    "ledger_sha256",
    "attempt_sha256",
    "root_reservation_sha256",
    "cmd_log_sha256",
    "cmd_sha256",
    "kernel_sha256",
    "main_sha256",
    "samples_sha256",
    "seed_sha256",
    "staged_binary_sha256",
    "cuda_sass_sha256",
    "source_sha256",
    "source_raw_sha256",
    "preserved_archive_sha256",
    "drand_randomness",
    "drand_signature",
    "pin_sha256",
    "entry_sha256",
    "expected_sha256",
    "actual_sha256",
    "tarball_sha256",
    "weights_manifest_sha256",
    "GLM_SAFE_EXPECTED_BINARY_SHA256",
}
# Keys whose value is a MAP of repo-relative path -> sha256 (audit bindings).
# Every string leaf under them is an allowed digest.
map_allowlist = {
    "accuracy_result_sha256",
    "artifact_sha256",
    "artifacts",
    "crash_artifact_sha256",
    "engine_source_sha256",
    "environment_sha256",
    "external_artifacts",
    "evidence_archive",
    "evalset_sha256",
    "model_files",
    "row_sha256",
    "shared_libraries",
    "tests_sha256_by_path",
}
w3_campaign_raw = re.fullmatch(
    r"results/glm52-gates/W3-performance-campaign-[^/]+/raw\.jsonl",
    display_path,
) is not None
w3_campaign_manifest = re.fullmatch(
    r"results/glm52-gates/W3-performance-campaign-[^/]+/manifest\.json",
    display_path,
) is not None
w3_campaign_summary = re.fullmatch(
    r"results/glm52-gates/W3-performance-campaign-[^/]+/summary\.json",
    display_path,
) is not None
r05_proxy_accuracy_plan = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-router-accuracy-plan-v2.json"
)
r05_proxy_seed_domain = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-seed-domain-v2.json"
)
r05_proxy_pack_evidence = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-pack-v1.json"
)
r05_proxy_red_candidate_3 = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-behavioral-red-candidate-3.json"
)
r05_proxy_pack_evidence_v2 = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-pack-v2.json"
)
r05_proxy_red_candidate_4 = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-behavioral-red-candidate-4.json"
)
r05_proxy_review_r195 = display_path == (
    "results/glm52-gates/R0.5-mtp-proxy-router-review-r195.json"
)
w7_resume_plan = re.fullmatch(
    r"results/glm52-gates/W7-resume-correctness-plan-v(?:1|5|6|8|9|10)\.json",
    display_path,
) is not None
w7_fixture_pool = display_path == (
    "results/glm52-gates/harness/w7-production-fixture-pool-v1.json"
)
w3_campaign_allowlist = set()
if w3_campaign_manifest:
    w3_campaign_allowlist.update({"input_manifest_sha256", "input_summary_sha256"})
if w3_campaign_summary:
    w3_campaign_allowlist.add("pair_request_sha256")
r05_proxy_accuracy_allowlist = {
    "ds4_c_sha256",
    "ds4_cuda_cu_sha256",
    "request_id",
    "train_precision_manifest_sha256",
    "train_precision_records_sha256",
    "train_fit_manifest_sha256",
    "train_fit_records_sha256",
} if r05_proxy_accuracy_plan else set()
r05_proxy_seed_allowlist = {
    "freeze_manifest_sha256_hex",
    "drand_randomness_hex",
    "seed32_sha256",
} if r05_proxy_seed_domain else set()
r05_proxy_pack_allowlist = {
    "builder_sha256",
    "inventory_sha256",
    "model_sha256",
    "pack_sha256",
    "receipt_sha256",
    "test_sha256",
} if r05_proxy_pack_evidence else set()
r05_proxy_red_candidate_3_allowlist = {
    "inventory_sha256",
    "model_sha256",
    "pack_sha256",
} if r05_proxy_red_candidate_3 else set()
r05_proxy_pack_v2_allowlist = {
    "builder_sha256",
    "inventory_sha256",
    "model_sha256",
    "pack_sha256",
    "receipt_sha256",
    "test_sha256",
} if r05_proxy_pack_evidence_v2 else set()
r05_proxy_red_candidate_4_allowlist = {
    "pack_sha256",
} if r05_proxy_red_candidate_4 else set()
r05_proxy_review_r195_allowlist = {
    "pack_sha256",
} if r05_proxy_review_r195 else set()
w7_resume_plan_allowlist = {
    "ds4_c_sha256",
    "ds4_cuda_cu_sha256",
    "ds4_server_source_sha256",
    "oracle_source_sha256",
    "oracle_binary_sha256",
    "builder_sha256",
    "binary_sha256",
    "stem_file_sha256",
    "stem_text_sha256",
    "sha256",
} if w7_resume_plan else set()
w7_fixture_pool_allowlist = {
    "wire_sha256",
    "caller_wire_sha256",
    "rendered_wire_sha256",
    "ds4_server_source_sha256",
    "oracle_source_sha256",
    "oracle_binary_sha256",
} if w7_fixture_pool else set()
hex64 = re.compile(r"[0-9a-fA-F]{64}")
raw = os.fdopen(3, encoding="utf-8").read()
try:
    if display_path.endswith(".jsonl"):
        document = [
            json.loads(line)
            for line in raw.splitlines()
            if line.strip()
        ]
    else:
        document = json.loads(raw)
except (UnicodeDecodeError, json.JSONDecodeError) as error:
    print(f"{display_path}: invalid exempted JSON: {error}", file=sys.stderr)
    raise SystemExit(1)

findings = []


def child_path(path, key):
    if isinstance(key, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def walk(value, path, allowed_string=False):
    if isinstance(value, str):
        if hex64.fullmatch(value) and not allowed_string:
            findings.append((path, value))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            item_path = child_path(path, key)
            if (
                w3_campaign_raw and key == "executed_identity" and
                isinstance(item, list) and len(item) == 4 and
                all(isinstance(child, str) for child in item) and
                re.fullmatch(r"[0-9]+", item[0]) and
                re.fullmatch(r"[0-9]+", item[1]) and
                hex64.fullmatch(item[2]) and
                re.fullmatch(r"[0-9]+:[0-9]+", item[3])
            ):
                continue
            if (key in map_allowlist or (r05_proxy_red_candidate_4 and key == "test_sha256")) and isinstance(item, dict) and all(
                isinstance(child, str) for child in item.values()
            ):
                continue
            leaf_allowed = (
                key in allowlist or
                key in w3_campaign_allowlist or
                key in r05_proxy_accuracy_allowlist or
                key in r05_proxy_seed_allowlist or
                key in r05_proxy_pack_allowlist or
                key in r05_proxy_red_candidate_3_allowlist or
                key in r05_proxy_pack_v2_allowlist or
                key in r05_proxy_red_candidate_4_allowlist or
                key in r05_proxy_review_r195_allowlist or
                key in w7_resume_plan_allowlist
                or key in w7_fixture_pool_allowlist
            )
            if isinstance(item, list):
                for index, element in enumerate(item):
                    walk(
                        element,
                        f"{item_path}[{index}]",
                        leaf_allowed and isinstance(element, str),
                    )
            else:
                walk(item, item_path, leaf_allowed and isinstance(item, str))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            walk(item, f"{path}[{index}]")


walk(document, "$")
for path, value in findings:
    print(f"{display_path}:{path}: {value[:6]}[REDACTED]", file=sys.stderr)
raise SystemExit(bool(findings))
PY
}

scan_digest_manifest() {
  local display_path="$1"
  python3 - "$display_path" 3<&0 <<'PY'
import os
import re
import sys

display_path = sys.argv[1]
hex64 = re.compile(r"[0-9a-fA-F]{64}")
valid_line = re.compile(r"[0-9a-f]{64}  \S+")
failed = False
for line_number, line in enumerate(os.fdopen(3, encoding="utf-8"), 1):
    line = line.rstrip("\n")
    match = hex64.search(line)
    if match and not valid_line.fullmatch(line):
        value = match.group(0)
        print(
            f"{display_path}:{line_number}: {value[:6]}[REDACTED] invalid digest line",
            file=sys.stderr,
        )
        failed = True
raise SystemExit(failed)
PY
}

scan_public_evidence_log() {
  local display_path="$1"
  python3 - "$display_path" 3<&0 <<'PY'
import os
import re
import sys

display_path = sys.argv[1]
approved = {
    "5bc2c255f33fb8a282c4c67037579f26"
    + "be573ded680e4d5cdd1c0adbd0bad3f0",
}
failed = False
for line_number, line in enumerate(os.fdopen(3, encoding="utf-8"), 1):
    for value in re.findall(r"[0-9a-fA-F]{64}", line):
        if value.lower() not in approved:
            print(
                f"{display_path}:{line_number}: unapproved evidence digest",
                file=sys.stderr,
            )
            failed = True
raise SystemExit(failed)
PY
}

scan_digest_file() {
  local display_path="$1"
  case "$display_path" in
    MANIFEST|*/MANIFEST|*.sha256) scan_digest_manifest "$display_path" ;;
    results/glm52-goal/evidence/glm-diagnostic-*/*/*.log|results/glm52-goal/evidence/glm-diagnostic-*/success/process.identity) scan_public_evidence_log "$display_path" ;;
    *) scan_digest_json "$display_path" ;;
  esac
}

scan_staged() {
  local gitleaks
  gitleaks="$(require_gitleaks)" || return 1
  "$gitleaks" protect --staged

  # Diff base must exist even on an unborn branch (first commit), else zero
  # files are scanned and staged secrets pass silently.
  local diff_base
  diff_base="$(git rev-parse --verify --quiet HEAD || git hash-object -t tree /dev/null)"

  local -a files_full files_nohex
  split_files < <(git diff --cached --name-only --diff-filter=ACMR -z "$diff_base")
  if ((${#files_full[@]} == 0 && ${#files_nohex[@]} == 0)); then
    return 0
  fi

  local failed=0
  if ((${#files_full[@]} > 0)); then
    { git grep --cached -n -I -E "$SECRET_PATTERN" -- "${files_full[@]}" 2>/dev/null || true; } | scan_stream || failed=1
  fi
  if ((${#files_nohex[@]} > 0)); then
    { git grep --cached -n -I -E "$SECRET_PATTERN_NOHEX" -- "${files_nohex[@]}" 2>/dev/null || true; } | scan_stream || failed=1
    local f
    for f in "${files_nohex[@]}"; do
      git show ":$f" | scan_digest_file "$f" || failed=1
    done
  fi
  return "$failed"
}

scan_commit() {
  local commit="$1"
  local -a files_full files_nohex
  split_files < <(git diff-tree --root --no-commit-id --name-only --diff-filter=ACMR -r -z "$commit")

  if ((${#files_full[@]} == 0 && ${#files_nohex[@]} == 0)); then
    return 0
  fi

  local failed=0
  if ((${#files_full[@]} > 0)); then
    { git grep -n -I -E "$SECRET_PATTERN" "$commit" -- "${files_full[@]}" 2>/dev/null || true; } \
      | sed -E "s/^${commit}://" \
      | scan_stream || failed=1
  fi
  if ((${#files_nohex[@]} > 0)); then
    { git grep -n -I -E "$SECRET_PATTERN_NOHEX" "$commit" -- "${files_nohex[@]}" 2>/dev/null || true; } \
      | sed -E "s/^${commit}://" \
      | scan_stream || failed=1
    local f
    for f in "${files_nohex[@]}"; do
      git show "${commit}:$f" | scan_digest_file "$f" || failed=1
    done
  fi
  return "$failed"
}

scan_pushed_ref() {
  local local_sha="$1"
  local remote_sha="$2"
  local zero_sha='0000000000000000000000000000000000000000'
  local commit failed=0

  [[ "$local_sha" != "$zero_sha" ]] || return 0
  if [[ "$remote_sha" == "$zero_sha" ]]; then
    while IFS= read -r commit; do
      scan_commit "$commit" || failed=1
    done < <(git rev-list "$local_sha")
  else
    while IFS= read -r commit; do
      scan_commit "$commit" || failed=1
    done < <(git rev-list "$remote_sha..$local_sha")
  fi
  return "$failed"
}

scan_push() {
  local gitleaks
  gitleaks="$(require_gitleaks)" || return 1
  "$gitleaks" detect

  local local_ref local_sha remote_ref remote_sha
  local failed=0
  while read -r local_ref local_sha remote_ref remote_sha; do
    if ! scan_pushed_ref "$local_sha" "$remote_sha"; then
      failed=1
    fi
  done
  return "$failed"
}

self_test() {
  local fake_secret
  fake_secret="$(printf 'a%.0s' {1..64})"
  if printf 'self-test.txt:1:%s\n' "$fake_secret" | scan_stream >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: fake secret was not detected' >&2
    return 1
  fi
  if ! printf 'scripts/71_install_glm_benchmark_lock_acl.sh:15:readonly SOURCE_SHA256=%s\n' \
      "$fake_secret" | scan_stream >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: pinned public digest was rejected' >&2
    return 1
  fi
  if printf 'scripts/other.sh:15:readonly SOURCE_SHA256=%s\n' "$fake_secret" \
      | scan_stream >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: public digest allowlist was too broad' >&2
    return 1
  fi
  if printf '{"note":"%s"}\n' "$fake_secret" \
      | scan_digest_json 'self-test-note.json' >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: JSON note secret was not detected' >&2
    return 1
  fi
  if ! printf '{"sha256":"%s"}\n' "$fake_secret" \
      | scan_digest_json 'self-test-sha256.json' >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: allowed JSON sha256 was rejected' >&2
    return 1
  fi
  local w3_campaign_raw_path
  w3_campaign_raw_path='results/glm52-gates/W3-performance-campaign-self-test/raw.jsonl'
  if ! is_checksum_file "$w3_campaign_raw_path" ||
      ! printf '{"executed_identity":["1","2","%s","3:4"],"bindings":{"binary_sha256":"%s"}}\n' \
        "$fake_secret" "$fake_secret" \
        | scan_digest_json "$w3_campaign_raw_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: W3 campaign identity digests were rejected' >&2
    return 1
  fi
  if printf '{"unrelated":"%s"}\n' "$fake_secret" \
      | scan_digest_json "$w3_campaign_raw_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: W3 campaign digest allowlist was too broad' >&2
    return 1
  fi
  if printf '{"executed_identity":["%s","2","%s","3:4"]}\n' \
      "$fake_secret" "$fake_secret" \
      | scan_digest_json "$w3_campaign_raw_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: malformed W3 executed identity was accepted' >&2
    return 1
  fi
  if printf '{"executed_identity":["%s"]}\n' "$fake_secret" \
      | scan_digest_json 'results/decision.json' >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: W3 campaign fields were allowed outside their path' >&2
    return 1
  fi
  local r05_proxy_plan_path
  r05_proxy_plan_path='results/glm52-gates/R0.5-mtp-proxy-router-accuracy-plan-v2.json'
  if ! printf '{"request_id":"%s","train_fit_records_sha256":"%s"}\n' \
      "$fake_secret" "$fake_secret" \
      | scan_digest_json "$r05_proxy_plan_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: R0.5 proxy-plan public bindings were rejected' >&2
    return 1
  fi
  if printf '{"request_id":"%s"}\n' "$fake_secret" \
      | scan_digest_json 'results/decision.json' >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: R0.5 proxy-plan digest allowlist escaped its exact path' >&2
    return 1
  fi
  if printf '{"unrelated":"%s"}\n' "$fake_secret" \
      | scan_digest_json "$r05_proxy_plan_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: R0.5 proxy-plan digest allowlist was too broad' >&2
    return 1
  fi
  local r05_seed_domain_path
  r05_seed_domain_path='results/glm52-gates/R0.5-mtp-proxy-seed-domain-v2.json'
  if ! printf '{"seed32_sha256":"%s"}\n' "$fake_secret" \
      | scan_digest_json "$r05_seed_domain_path" >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: R0.5 seed-domain vector was rejected' >&2
    return 1
  fi
  if printf '{"seed32_sha256":"%s"}\n' "$fake_secret" \
      | scan_digest_json 'results/decision.json' >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: R0.5 seed-domain allowlist escaped its path' >&2
    return 1
  fi
  if ! printf 'scripts/tests/test_glm_mtp_proxy_seed_contract.py:8:EXPECTED = "%s"\n' \
      "$fake_secret" | scan_stream >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: R0.5 public seed test vector was rejected' >&2
    return 1
  fi
  if printf 'scripts/tests/other.py:8:EXPECTED = "%s"\n' "$fake_secret" \
      | scan_stream >/dev/null 2>&1; then
    printf '%s\n' 'self-test failed: R0.5 public seed code allowlist escaped its path' >&2
    return 1
  fi
  local decode_evidence_path
  decode_evidence_path='results/glm52-goal/evidence/dsv4-decode-no-result-test/failure.json'
  if ! is_checksum_file "$decode_evidence_path"; then
    printf '%s\n' \
      'self-test failed: structured decode evidence was not classified for field-aware scanning' >&2
    return 1
  fi
  if ! printf '{"harness_sha256":"%s","result_sha256":"%s"}\n' \
      "$fake_secret" "$fake_secret" \
      | scan_digest_json "$decode_evidence_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: declared decode evidence digests were rejected' >&2
    return 1
  fi
  if printf '{"note":"%s"}\n' "$fake_secret" \
      | scan_digest_json "$decode_evidence_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: undeclared decode evidence digest was accepted' >&2
    return 1
  fi
  local slab_no_result_path
  slab_no_result_path='results/glm52-gates/G6-rung0-io-slab-calibration-no-results.json'
  if ! is_checksum_file "$slab_no_result_path" ||
      ! printf '{"sidecar_sha256":"%s","server_log_sha256":"%s"}\n' \
        "$fake_secret" "$fake_secret" \
        | scan_digest_json "$slab_no_result_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: slab NO_RESULT evidence digests were rejected' >&2
    return 1
  fi
  local e637_attempt_path
  e637_attempt_path='results/glm52-gates/R0-e637-campaign-attempt-2026-08-02.json'
  if ! is_checksum_file "$e637_attempt_path" ||
      ! printf '{"binary_sha256":"%s","sha256":"%s"}\n' \
        "$fake_secret" "$fake_secret" \
        | scan_digest_json "$e637_attempt_path" >/dev/null 2>&1; then
    printf '%s\n' \
      'self-test failed: e637 campaign evidence digests were rejected' >&2
    return 1
  fi
  printf '%s\n' 'self-test passed'
}

case "${1:---staged}" in
  --staged)
    scan_staged
    ;;
  --pre-push)
    scan_push
    ;;
  --self-test)
    self_test
    ;;
  *)
    printf 'usage: %s [--staged|--pre-push|--self-test]\n' "$0" >&2
    exit 2
    ;;
esac
