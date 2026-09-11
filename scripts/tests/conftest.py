"""Test-suite policy shared by every module under scripts/tests.

Two lists live here so CI (ubuntu, no Spark, no dsv4 user, no root-owned
runtime trees) and the Spark itself run the same command and get an honest
result:

* SPARK_BOUND: tests that need the machine — the dsv4/bmarti44 accounts, the
  production checkout path, /run/lock/frontier-at-home, the sealed W1/W9
  runtime trees, the nvm node binary. Off-host they are SKIPPED with the
  reason. Set FRONTIER_HOST_BOUND_TESTS=1 to force them on, =0 to force off.
* KNOWN_STALE: tests that fail on main today for reasons unrelated to any
  open change (vendored DSV4 encoder deliberately not committed; wrapper
  contract strings that moved). They run everywhere and report XFAIL, so a
  fix flips them to XPASS instead of hiding.

Keep both lists short and dated; remove entries when the underlying reason
goes away.
"""
from __future__ import annotations

import os
from pathlib import Path
import pwd

import pytest

PRODUCTION_CHECKOUT = Path("/home/bmarti44/spark-deepseek-v4-flash")


def _on_spark() -> bool:
    forced = os.environ.get("FRONTIER_HOST_BOUND_TESTS")
    if forced in {"0", "1"}:
        return forced == "1"
    try:
        pwd.getpwnam("dsv4")
    except KeyError:
        return False
    return PRODUCTION_CHECKOUT.is_dir()


ON_SPARK = _on_spark()

# nodeid suffix (module::Class::test) -> what it needs from the host.
SPARK_BOUND = {
    "test_dsv4_context_graduation.py::ContextProbeTests::test_journal_artifact_witness_rejects_post_seal_rewrite":
        "journal witness bound to the Spark's trusted record",
    "test_dsv4_context_graduation.py::ContextProbeTests::test_journal_witness_is_process_linked_and_tamper_evident":
        "journal witness bound to the Spark's trusted record",
    "test_dsv_matched_signal_cleanup.py::DsvMatchedSignalCleanupTests::test_term_int_and_hup_stop_the_exact_transient_unit":
        "/run/lock/frontier-at-home/inference.lock",
    "test_foundation_user_runtime.py::FoundationRuntimeTests::test_execute_arm_writes_fixed_baseline_from_production_probe":
        "must run as bmarti44 on the Spark",
    "test_glm52_w1_root_attestor.py::RootAttestorContractTests::test_held_legacy_lock_fails_before_new_lock_creation":
        "dsv4 account and /run/dsv4 lock namespace",
    "test_glm52_w1_root_attestor.py::RootAttestorContractTests::test_inference_lock_is_left_usable_by_dsv4":
        "dsv4 account and /run/dsv4 lock namespace",
    "test_glm52_w1_root_attestor.py::RootAttestorContractTests::test_root_clone_scopes_safe_directory_to_exact_source_gitdir":
        "production checkout gitdir",
    "test_glm52_w1_root_attestor.py::RootAttestorContractTests::test_submitter_holds_legacy_and_current_locks_together":
        "dsv4 account and /run/dsv4 lock namespace",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_existing_journal_reservation_blocks_before_publication":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_existing_root_tombstone_fails_closed":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_journal_eviction_cannot_reopen_permanent_authority":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_root_authority_rejects_changed_executing_controller":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_root_authority_rejects_unapproved_executing_commit":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_root_managed_journal_reservation_is_visible_before_return":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_baseline.py::AtomicLifecycleTests::test_root_tombstone_helper_response_is_exactly_bound":
        "root-managed journal under /run/lock/frontier-at-home",
    "test_glm_union_trace_smoke.py::UnionTraceSmokeVerdictTests::test_committed_randomness_postdates_committed_freeze":
        "freeze commit timestamps of the Spark's sealed corpus",
    "test_glm_union_trace_smoke.py::UnionTraceSmokeVerdictTests::test_corpus_randomness_postdates_corpus_freeze":
        "freeze commit timestamps of the Spark's sealed corpus",
    "test_glm_union_trace_smoke.py::UnionTraceSmokeVerdictTests::test_quality_raw_output_accepts_exact_bytes_with_noncanonical_bpe":
        "vendored DSV4 tokenizer on the Spark",
    "test_glm_union_trace_smoke.py::UnionTraceSmokeVerdictTests::test_quality_raw_output_uses_first_reasoning_boundary":
        "vendored DSV4 tokenizer on the Spark",
    "test_runtime_lock_provisioner.py::RuntimeLockProvisionerTests::test_existing_legacy_lock_is_opened_without_create":
        "dsv4 group and /run/dsv4 lock namespace",
    "test_runtime_lock_provisioner.py::RuntimeLockProvisionerTests::test_legacy_holder_blocks_before_current_lock_is_published":
        "dsv4 group and /run/dsv4 lock namespace",
    "test_runtime_lock_provisioner.py::RuntimeLockProvisionerTests::test_preplanted_current_parent_symlink_is_not_followed":
        "dsv4 group and /run/dsv4 lock namespace",
    "test_runtime_lock_provisioner.py::RuntimeLockProvisionerTests::test_success_publishes_both_locks_only_while_bridge_is_held":
        "dsv4 group and /run/dsv4 lock namespace",
    "test_w7_drand_verifier.py::W7DrandVerifierTest::test_pinned_default_chain_beacon_verifies":
        "pinned nvm node binary under /home/bmarti44",
    "test_w7_equivalence_launcher.py::W7EquivalenceLauncherTest::test_all_runtime_programs_are_kernel_sealed":
        "kernel-sealed runtime programs under /usr/local/libexec",
    "test_w7_resume_production_launcher.py::W7ProductionLauncherTest::test_all_runtime_programs_are_kernel_sealed":
        "kernel-sealed runtime programs under /usr/local/libexec",
    "test_w7_resume_production_launcher.py::W7ProductionLauncherTest::test_public_randomness_rejects_stale_forged_and_disagreement":
        "pinned nvm node binary under /home/bmarti44",
    "test_w8_exact_smoke_contract.py::W8ExactSmokeContractTests::test_safe_wrapper_candidate_source_contains_frozen_binary":
        "frozen candidate binary under /usr/local/libexec",
    "test_w9_e2m1_fidelity_runner.py::W9E2M1FidelityRunnerTests::test_launcher_is_frozen_base_plus_only_the_e2m1_flag":
        "sealed W9 launcher tree on the Spark",
}

# nodeid suffix -> why it fails on main (2026-09-11).
KNOWN_STALE = {
    "test_encoder_registration.py::EncoderRegistrationTests::test_every_registered_encoder_exists":
        "vendor/official-encoding/encoding/encoding_dsv4.py is gitignored on purpose",
    "test_encoding_qwen38.py::Qwen38EncodingTests::test_encoder_default_preserves_dsv4_behavior":
        "vendor/official-encoding/encoding/encoding_dsv4.py is gitignored on purpose",
    "test_matched_memory_safety.py::MatchedHarnessContractTests::test_glm_wrapper_hashes_the_selected_candidate_binary":
        "asserts wrapper strings that glm_safe_run.sh no longer carries",
    "test_matched_memory_safety.py::MatchedHarnessContractTests::test_glm_wrapper_rejects_unsafe_safety_overrides":
        "asserts wrapper strings that glm_safe_run.sh no longer carries",
}


def _suffix(item: pytest.Item) -> str:
    nodeid = item.nodeid.replace("\\", "/")
    return nodeid.split("scripts/tests/", 1)[-1]


def pytest_collection_modifyitems(config, items):  # noqa: ARG001 - pytest hook
    for item in items:
        key = _suffix(item)
        if key in KNOWN_STALE:
            item.add_marker(pytest.mark.xfail(reason=f"known stale: {KNOWN_STALE[key]}", strict=False))
        elif key in SPARK_BOUND and not ON_SPARK:
            item.add_marker(pytest.mark.skip(reason=f"Spark-bound: {SPARK_BOUND[key]}"))


def pytest_report_header(config):  # noqa: ARG001 - pytest hook
    return f"frontier host-bound tests: {'on' if ON_SPARK else 'off (skipped)'}"
