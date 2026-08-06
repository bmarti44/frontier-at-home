#!/usr/bin/env python3
"""Production-path acceptance tests for scripts/glm52_goal.py."""

from __future__ import annotations

import importlib.util
import base64
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import hashlib
import struct
import zlib
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "glm52_goal.py"


def load_goal_module():
    spec = importlib.util.spec_from_file_location("glm52_goal", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def replacement_attempt_validator(_attempt):
    raise ValueError("mutated validator")


def replacement_finite_number(_value, _label, *, minimum=0.0):
    return minimum + 1.0


def replacement_utc_timestamp(_value, _label):
    return None


def replacement_w11_fixture(_candidate_hash, _seed_sha256):
    return {}


def replacement_dsv4_profile(_candidate_hash):
    return {
        "schema_version": 1,
        "profile": "dsv4",
        "binary_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
    }


def _compressed_f32(count, peak_index=3, peak_value=1.0):
    raw = bytearray(count * 4)
    struct.pack_into("<f", raw, peak_index * 4, peak_value)
    return base64.b64encode(zlib.compress(bytes(raw), 9)).decode("ascii")


def _compressed_i32(values):
    raw = struct.pack(f"<{len(values)}i", *values)
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def _i32_sha256(values):
    return hashlib.sha256(struct.pack(f"<{len(values)}i", *values)).hexdigest()


def _w7_payload_bytes(token_count):
    return (
        13 * 4 + token_count * 4 + 154880 * 4 + 78 * 4 * 2 +
        token_count * (78 * (512 + 64) * 4 + 21 * 128 * 4)
    )


def w7_resume_record(execution_nonce_sha256="c" * 64):
    logits = _compressed_f32(154880, 13, 2.0)
    confirmation_seed = "d" * 64
    binary_sha256 = "a" * 64
    configuration_sha256 = "b" * 64
    pool_path = ROOT / "results/glm52-gates/harness/w7-production-fixture-pool-v1.json"
    pool_raw = pool_path.read_bytes()
    pool = json.loads(pool_raw)
    pool_sha256 = hashlib.sha256(pool_raw).hexdigest()
    pool_by_variant = {item["variant"]: item for item in pool["variants"]}
    live_raw = zlib.decompress(base64.b64decode(pool["live"]["token_ids_zlib_b64"]))
    frozen_live = list(struct.unpack(f"<{pool['live']['token_count']}i", live_raw))
    attempt_id = hashlib.sha256(
        bytes.fromhex(binary_sha256) + bytes.fromhex(configuration_sha256) +
        bytes.fromhex(confirmation_seed) + bytes.fromhex(pool_sha256) +
        bytes.fromhex(execution_nonce_sha256)
    ).hexdigest()
    contracts = {
        "cold-primary": ("glm", "unset", "primary-cold"),
        "strict-primary": ("glm", "unset", "primary-extension"),
        "candidate-primary": ("glm", "1", "primary-extension"),
        "exact-off": ("glm", "unset", "exact-replay"),
        "exact-on": ("glm", "1", "exact-replay"),
        "divergence-off": ("glm", "unset", "divergence"),
        "divergence-on": ("glm", "1", "divergence"),
        "shorten-off": ("glm", "unset", "shorten"),
        "shorten-on": ("glm", "1", "shorten"),
        "malformed-off": ("glm", "unset", "malformed-checkpoint"),
        "malformed-on": ("glm", "1", "malformed-checkpoint"),
        "wrong-lineage-off": ("glm", "unset", "wrong-lineage"),
        "wrong-lineage-on": ("glm", "1", "wrong-lineage"),
        "non-glm-off": ("non-glm", "unset", "primary-extension"),
        "non-glm-on": ("non-glm", "1", "primary-extension"),
        "flag-zero": ("glm", "0", "primary-extension"),
        "flag-empty": ("glm", "", "primary-extension"),
        "flag-garbage": ("glm", "garbage", "primary-extension"),
    }

    def regime(regime_id, live, common, selected, prompt):
        selection_seed = hashlib.sha256(
            bytes.fromhex(confirmation_seed) + regime_id.encode("ascii")
        ).hexdigest()
        if regime_id == "primary":
            variant = "primary-fixed"
        else:
            base = int(selection_seed[:4], 16) % 8
            variant = base if regime_id == "confirmation-1" else 8 + base
        specification = pool_by_variant[variant]
        prompt = specification["prompt_tokens"]
        common = specification["common_tokens"]
        live = specification["live_tokens"]
        selected = specification["selected_tokens"]
        token_raw = zlib.decompress(
            base64.b64decode(specification["canonical_token_ids_zlib_b64"])
        )
        tokens = list(struct.unpack(f"<{prompt}i", token_raw))
        wire_bytes = base64.b64decode(specification["rendered_wire_utf8_b64"])
        offset_raw = zlib.decompress(
            base64.b64decode(specification["wire_token_end_offsets_zlib_b64"])
        )
        wire_prefix_bytes = [0, *struct.unpack(f"<{prompt}i", offset_raw)]
        payload = hashlib.sha256(f"payload:{regime_id}:{selected}".encode()).hexdigest()

        def inventory_item(record_id, token_length, inode, *, valid=True,
                           wire_match=True, lineage_match=True, payload_bytes=None):
            if payload_bytes is None:
                payload_bytes = _w7_payload_bytes(token_length)
            prefix_hash = _i32_sha256(tokens[:token_length])
            return {
                "record_id": record_id,
                "path_sha256": hashlib.sha256(f"{regime_id}:{record_id}".encode()).hexdigest(),
                "device": 259, "inode": inode, "size": payload_bytes + 256,
                "mtime_ns": inode * 10, "payload_bytes": payload_bytes,
                "token_length": token_length,
                "rendered_prefix_bytes": wire_prefix_bytes[token_length],
                "rendered_prefix_sha256": (
                    hashlib.sha256(wire_bytes[:wire_prefix_bytes[token_length]]).hexdigest()
                    if wire_match else "7" * 64
                ),
                "token_sha256": prefix_hash,
                "lineage_sha256": prefix_hash if lineage_match else "9" * 64,
                "payload_sha256": (
                    payload if record_id == "selected" else
                    hashlib.sha256(f"{regime_id}:{record_id}:payload".encode()).hexdigest()
                ),
                "structurally_valid": valid,
                "matches_wire_prefix": wire_match,
                "matches_lineage": lineage_match,
            }

        inventory = [
            inventory_item("older", selected - 4, 1001),
            inventory_item("selected", selected, 1002),
            inventory_item("wrong-lineage", selected + 1, 1003, lineage_match=False),
            inventory_item("malformed", selected + 2, 1004, valid=False, payload_bytes=0),
        ]

        def event(invocation_id, seq, kind, *, record_id=None,
                  checkpoint_tokens=None, payload_sha256=None, start=None,
                  end=None, byte_count=0, status="ok"):
            return {
                "seq": seq, "invocation_id": invocation_id, "kind": kind,
                "record_id": record_id, "checkpoint_tokens": checkpoint_tokens,
                "payload_sha256": payload_sha256, "start": start, "end": end,
                "byte_count": byte_count, "status": status,
            }

        indexer_layers = [0, 1, 2] + list(range(6, 78, 4))

        def state_manifest():
            config = {
                "normal_layers": 78, "vocab": 154880, "kv_lora": 512,
                "rope": 64, "indexer_dim": 128, "full_live": 0,
                "compact_live": prompt, "indexer_layers": indexer_layers,
            }
            members = [
                ("header", "u32", [13]),
                ("checkpoint_tokens", "u32", [prompt]),
                ("logits", "f32", [154880]),
                ("compact_live_rows", "u32", [78]),
                ("index_live_rows", "u32", [78]),
            ]
            for layer in range(78):
                members.extend([
                    (f"layer.{layer:02d}.kv_lora", "f32", [prompt, 512]),
                    (f"layer.{layer:02d}.k_rope", "f32", [prompt, 64]),
                ])
                if layer in indexer_layers:
                    members.append(
                        (f"layer.{layer:02d}.indexer_key", "f32", [prompt, 128])
                    )
            sections = []
            offset = 0
            for name, dtype, shape in members:
                count = math.prod(shape)
                size = count * 4
                sections.append({
                    "name": name, "dtype": dtype, "shape": shape,
                    "offset": offset, "byte_count": size,
                    "content_sha256": hashlib.sha256(
                        f"{regime_id}:state:{name}".encode()
                    ).hexdigest(),
                })
                offset += size
            root = hashlib.sha256(
                "".join(item["content_sha256"] for item in sections).encode()
            ).hexdigest()
            return {
                "format": "ds4-glm-session-canonical-v2", "config": config,
                "total_bytes": offset, "content_sha256": root,
                "token_lineage_sha256": _i32_sha256(tokens),
                "logical_frontiers": [prompt] * 78, "sections": sections,
            }

        def safety(case_id):
            return {
                "wrapper_path": "results/glm52-gates/harness/glm_safe_run.sh",
                "wrapper_sha256": "6e4d382bc5e5818787af8c17aae7a0750ca3ab7b36471f21355789d194b2e801",
                "lock_path": "/run/dsv4/inference.lock", "lock_owner_uid": 0,
                "lock_mode_octal": "0660", "lock_acquired": True,
                "cgroup_path": f"/sys/fs/cgroup/system.slice/glm52-w7-{regime_id}-{case_id}.scope",
                "cgroup_identity_sha256": "8" * 64,
                "start_available_gib": 112.0, "minimum_available_gib": 12.0,
                "bare_engine_detected": False, "terminal_rc": 0,
                "oom_events_before": 0, "oom_events_after": 0, "xid_count": 0,
                "swap_used_before_bytes": 0, "swap_used_after_bytes": 0,
                "survivor_pids": [],
            }

        live_base = frozen_live

        def case(case_id):
            family, flag, mutation = contracts[case_id]
            invocation_id = f"invoke:{attempt_id}:{regime_id}:{case_id}"
            live_ids = [] if case_id == "cold-primary" else (
                tokens if case_id in {"exact-off", "exact-on"} else live_base
            )
            row = {
                "case_id": case_id, "invocation_id": invocation_id,
                "model_family": family, "flag_value": flag,
                "fixture_mutation": mutation,
                "wire_text_utf8_b64": base64.b64encode(wire_bytes).decode("ascii"),
                "wire_token_end_offsets_zlib_b64": _compressed_i32(wire_prefix_bytes[1:]),
                "canonical_token_ids_zlib_b64": _compressed_i32(tokens),
                "live_token_count": len(live_ids),
                "live_token_ids_zlib_b64": _compressed_i32(live_ids) if live_ids else None,
                "restored_checkpoint_token_ids_zlib_b64": None,
                "sync_tokens_sha256": _i32_sha256(tokens),
                "final_lineage_sha256": _i32_sha256(tokens),
                "prompt_tokens": prompt, "checkpoint_inventory": [], "events": [],
                "output_token_ids": None, "logits_f32_zlib_b64": None,
                "state_manifest": None, "safety": safety(case_id),
            }
            if case_id in {
                "cold-primary", "strict-primary", "candidate-primary",
                "exact-off", "exact-on",
            }:
                row["output_token_ids"] = list(range(64))
                row["logits_f32_zlib_b64"] = logits
                row["state_manifest"] = state_manifest()
            if case_id == "cold-primary":
                row["events"] = [
                    event(invocation_id, 0, "reset"),
                    event(invocation_id, 1, "evaluate", start=0, end=prompt - 2),
                    event(invocation_id, 2, "evaluate", start=prompt - 2, end=prompt),
                ]
            elif case_id in {
                "strict-primary", "candidate-primary", "flag-zero",
                "flag-empty", "flag-garbage",
            }:
                row["checkpoint_inventory"] = copy.deepcopy(inventory)
                row["restored_checkpoint_token_ids_zlib_b64"] = _compressed_i32(tokens[:selected])
                events = [
                    event(invocation_id, index, "matcher_candidate",
                          record_id=item["record_id"],
                          checkpoint_tokens=item["token_length"],
                          payload_sha256=item["payload_sha256"], status="observed")
                    for index, item in enumerate(inventory)
                ]
                events.extend([
                    event(invocation_id, 4, "matcher_selected", record_id="selected",
                          checkpoint_tokens=selected, payload_sha256=payload),
                    event(invocation_id, 5, "payload_read", record_id="selected",
                          checkpoint_tokens=selected, payload_sha256=payload,
                          byte_count=inventory[1]["payload_bytes"]),
                ])
                if case_id != "candidate-primary":
                    events.extend([
                        event(invocation_id, 6, "reset"),
                        event(invocation_id, 7, "evaluate", start=0, end=prompt - 2),
                        event(invocation_id, 8, "evaluate", start=prompt - 2, end=prompt),
                    ])
                else:
                    events.extend([
                        event(invocation_id, 6, "evaluate", start=selected, end=prompt - 1),
                        event(invocation_id, 7, "evaluate", start=prompt - 1, end=prompt),
                    ])
                row["events"] = events
            elif case_id.startswith("malformed-"):
                row["checkpoint_inventory"] = [copy.deepcopy(inventory[3])]
            elif case_id.startswith("wrong-lineage-"):
                row["checkpoint_inventory"] = [copy.deepcopy(inventory[2])]
            elif case_id.startswith("divergence-"):
                item = inventory_item("selected", selected, 1002, wire_match=False)
                row["checkpoint_inventory"] = [item]
            elif case_id.startswith("shorten-"):
                row["checkpoint_inventory"] = [
                    inventory_item("selected", common + 1, 1002)
                ]
            if row["checkpoint_inventory"] and not row["events"]:
                item = row["checkpoint_inventory"][0]
                row["events"] = [
                    event(invocation_id, 0, "matcher_candidate",
                          record_id=item["record_id"],
                          checkpoint_tokens=item["token_length"],
                          payload_sha256=item["payload_sha256"], status="observed")
                ]
            return row

        cases = [case(case_id) for case_id in contracts]
        fixture_material = {
            "fixture_variant": variant,
            "cases": [
                {
                    key: row[key] for key in (
                        "case_id", "model_family", "flag_value", "fixture_mutation",
                        "wire_text_utf8_b64", "wire_token_end_offsets_zlib_b64",
                        "canonical_token_ids_zlib_b64", "live_token_count",
                        "live_token_ids_zlib_b64", "prompt_tokens",
                        "checkpoint_inventory",
                    )
                }
                for row in cases
            ],
        }
        fixture_sha = hashlib.sha256(
            json.dumps(fixture_material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            "regime_id": regime_id, "fixture_variant": variant,
            "fixture_sha256": fixture_sha,
            "selection_seed_sha256": selection_seed, "cases": cases,
        }

    regimes = [
        regime("primary", 5063, 5045, 5044, 5066),
        regime("confirmation-1", 137, 111, 109, 143),
        regime("confirmation-2", 271, 240, 238, 284),
    ]
    return {
        "record_type": "w7_resume_observation",
        "gate": "W7", "attempt_id": attempt_id,
        "execution_nonce_sha256": execution_nonce_sha256,
        "binary_sha256": binary_sha256,
        "configuration_sha256": configuration_sha256,
        "fixture_sha256": hashlib.sha256(
            "".join(item["fixture_sha256"] for item in regimes).encode()
        ).hexdigest(),
        "guard_off_present": False,
        "confirmation_seed_sha256": confirmation_seed,
        "regimes": regimes,
        "failures": [],
    }


def _rebind_w7_fixture_hashes(record):
    fixture_keys = (
        "case_id", "model_family", "flag_value", "fixture_mutation",
        "wire_text_utf8_b64", "wire_token_end_offsets_zlib_b64",
        "canonical_token_ids_zlib_b64", "live_token_count",
        "live_token_ids_zlib_b64", "prompt_tokens", "checkpoint_inventory",
    )
    for regime in record["regimes"]:
        material = {
            "fixture_variant": regime["fixture_variant"],
            "cases": [
                {key: row[key] for key in fixture_keys}
                for row in regime["cases"]
            ],
        }
        regime["fixture_sha256"] = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    record["fixture_sha256"] = hashlib.sha256(
        "".join(item["fixture_sha256"] for item in record["regimes"]).encode()
    ).hexdigest()


def w11_record(hashes=None):
    identities = hashes or {
        "binary_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "model_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "fixture_sha256": "e" * 64,
    }
    stages = []
    for stage_index, context_cap in enumerate((1_048_576,)):
        start = stage_index * 2.0
        stages.append(
            {
                "context_cap": context_cap,
                "processed_tokens": context_cap,
                "started_at_seconds": start,
                "finished_at_seconds": start + 1.0,
                "completed_output_tokens": 8,
                "token_timestamps": [
                    start + 0.3 + index / 10 for index in range(8)
                ],
                "output_sha256": "8" * 64,
                "finish_reason": "stop",
                "truncated": False,
            }
        )
    retrieval = [
        {
            "case_id": f"needle-{index}",
            "position": position,
            "expected_sha256": str(index + 1) * 64,
            "observed_sha256": str(index + 1) * 64,
        }
        for index, position in enumerate((16_384, 524_288, 983_040))
    ]
    return {
        "record_type": "context_observation",
        **identities,
        "stages": stages,
        "retrieval_results": retrieval,
        "negative_control_results": [
            {
                "case_id": "absent-0",
                "expected_sha256": "4" * 64,
                "observed_sha256": "4" * 64,
            }
        ],
        "memory_samples": [
            {
                "timestamp_seconds": index * 0.25,
                "available_gib": 10.0 + index / 10,
                "swap_current_bytes": 0,
            }
            for index in range(29)
        ],
        "failure_events": [],
        "oom_events": [],
        "xid_events": [],
    }


def workstream_record(gate):
    workflow = {
        "test_committed": True,
        "red_confirmed": True,
        "implementation_default_off": True,
        "candidate_frozen": True,
        "post_freeze_randomness": True,
        "clean_build": True,
        "blinded_ab": True,
        "diff_scan_clean": True,
        "mutation_rejected": True,
    }
    metrics = {
        "W1": {
            "f16_tested": True,
            "block_e4m3_tested": True,
            "f32_rope": True,
            "fidelity_pass": True,
            "retrieval_pass": True,
            "available_memory_gib": 12.0,
        },
        "W2": {
            "byte_identical": True,
            "baseline_hit_rate": 0.50,
            "candidate_hit_rate": 0.54,
        },
        "W3": {
            "byte_identical": True,
            "event_safe": True,
            "baseline_seconds": [10.0] * 5,
            "candidate_seconds": [9.0] * 5,
        },
        "W4": {
            "ids_identical": True,
            "logits_identical": True,
            "baseline_topk_seconds": [10.0] * 5,
            "candidate_topk_seconds": [4.0] * 5,
            "baseline_prefill_seconds": [10.0] * 5,
            "candidate_prefill_seconds": [9.0] * 5,
        },
        "W5": {
            "scores_identical": True,
            "ids_identical": True,
            "logits_identical": True,
            "baseline_allocation_bytes": 200,
            "candidate_allocation_bytes": 100,
        },
        "W6": {
            "outputs_identical": True,
            "width2_measured": True,
            "width4_measured": True,
            "selected_width": 4,
            "width2_seconds": 9.0,
            "width4_seconds": 8.0,
            "baseline_load_bytes": 1000,
            "selected_load_bytes": 700,
        },
        "W7": {
            "complete_dumps_equal": True,
            "max_abs_logit_delta": 0.009,
            "argmax_identical": True,
            "checkpoint_correct": True,
            "global_guard_preserved": True,
        },
        "W8": {
            "checksums_verified": True,
            "corruption_failed_closed": True,
            "selected_rows_exact": True,
            "selected_block_cache": True,
            "context_1m_pass": True,
            "retrieval_pass": True,
            "available_memory_gib": 12.0,
        },
        "W9": {
            "real_capture": True,
            "capture_width": 512,
            "query_weighted_error": 0.005,
            "maximum_allowed_error": 0.01,
        },
        "W10": {
            "data_frozen": True,
            "splits_frozen": True,
            "seeds_frozen": True,
            "storage_ratio": 0.5,
            "maximum_storage_ratio": 0.6,
            "runtime_ratio": 1.1,
            "maximum_runtime_ratio": 1.2,
            "fidelity_pass": True,
        },
        "switch": {
            "serialized": True,
            "idempotent": True,
            "hashes_verified": True,
            "environment_allowlisted": True,
            "identity_safe_stop": True,
            "authenticated_completion": True,
            "unauthenticated_rejected": True,
            "memwatch_pass": True,
            "semantic_output_pass": True,
            "rollback_pass": True,
            "reboot_restore_pass": True,
            "fault_matrix_pass": True,
            "transition_cycle_pass": True,
        },
    }[gate]
    return {
        "record_type": "workstream_observation",
        "gate": gate,
        "binary_sha256": "a" * 64,
        "configuration_sha256": "b" * 64,
        "fixture_sha256": "c" * 64,
        "workflow": workflow,
        "metrics": metrics,
        "failures": [],
    }


class FormulaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.goal = load_goal_module()

    def test_decode_formula_uses_first_and_last_token(self):
        timestamps = [1.0 + index / 10.0 for index in range(128)]
        self.assertAlmostEqual(
            self.goal.decode_tokens_per_second(timestamps), 10.0, places=12
        )

    def test_decode_formula_rejects_short_or_nonfinite_input(self):
        for timestamps in (
            [1.0] * 127,
            [1.0] * 128,
            [1.0] * 127 + [math.nan],
            [1.0] * 127 + [math.inf],
        ):
            with self.subTest(timestamps=timestamps[-1]):
                with self.assertRaises(ValueError):
                    self.goal.decode_tokens_per_second(timestamps)

    def test_ratio_confidence_bounds_are_directional(self):
        candidate = [8.0, 8.1, 8.2, 8.3, 8.4]
        reference = [10.0, 10.1, 10.2, 10.3, 10.4]
        lower = self.goal.paired_ratio_bound(candidate, reference, side="lower")
        upper = self.goal.paired_ratio_bound(candidate, reference, side="upper")
        self.assertLessEqual(lower, sum(candidate) / sum(reference))
        self.assertGreaterEqual(upper, sum(candidate) / sum(reference))

    def test_ratio_rejects_bad_or_unpaired_samples(self):
        bad = (
            ([], []),
            ([1.0], [1.0, 2.0]),
            ([1.0, math.nan], [1.0, 2.0]),
            ([1.0, 0.0], [1.0, 2.0]),
            ([1.0, 2.0], [1.0, math.inf]),
        )
        for candidate, reference in bad:
            with self.subTest(candidate=candidate, reference=reference):
                with self.assertRaises(ValueError):
                    self.goal.paired_ratio_bound(candidate, reference, side="lower")

    def test_performance_acceptance_formula(self):
        passing = {
            "decode_glm": [8.4] * 5,
            "decode_dsv4": [10.0] * 5,
            "prefill_glm": [84.0] * 5,
            "prefill_dsv4": [100.0] * 5,
            "prefill_time_glm": [12.0] * 5,
            "prefill_time_dsv4": [10.0] * 5,
            "warm_ttft_glm": [1.15] * 5,
            "warm_ttft_dsv4": [1.0] * 5,
            "cold_ttft_glm": [11.5] * 5,
            "cold_ttft_dsv4": [10.0] * 5,
        }
        verdict = self.goal.performance_verdict(passing)
        self.assertEqual(verdict["verdict"], "PASS")
        mutated = dict(passing)
        mutated["decode_glm"] = [7.0] * 5
        self.assertEqual(self.goal.performance_verdict(mutated)["verdict"], "FAIL")

    def test_long_context_requires_real_processing_and_safety(self):
        passing = {
            "context_cap": 1_048_576,
            "processed_tokens": 1_000_000,
            "retrieval_pass": True,
            "negative_control_pass": True,
            "completed_generation": True,
            "truncated": False,
            "oom": False,
            "xid": False,
            "available_memory_gib": 10.0,
        }
        self.assertEqual(self.goal.context_verdict(passing)["verdict"], "PASS")
        for key, value in (
            ("processed_tokens", 999_999),
            ("retrieval_pass", False),
            ("negative_control_pass", False),
            ("truncated", True),
            ("oom", True),
            ("xid", True),
            ("available_memory_gib", 9.99),
        ):
            mutated = dict(passing)
            mutated[key] = value
            with self.subTest(key=key):
                self.assertEqual(
                    self.goal.context_verdict(mutated)["verdict"], "FAIL"
                )
        for key, value in (
            ("context_cap", "1048576"),
            ("processed_tokens", 1_000_000.9),
            ("processed_tokens", True),
        ):
            malformed = dict(passing)
            malformed[key] = value
            with self.subTest(malformed=key):
                with self.assertRaises(ValueError):
                    self.goal.context_verdict(malformed)

    def test_lossy_quality_formula_and_mutations(self):
        passing = [
            {
                "tokens": 100,
                "baseline_nll_sum": 200.0,
                "candidate_nll_sum": 200.5,
                "baseline_top1_correct": 70,
                "candidate_top1_correct": 70,
            }
            for _ in range(100)
        ]
        verdict = self.goal.quality_verdict(passing)
        self.assertEqual(verdict["verdict"], "PASS")
        high_nll = [dict(case) for case in passing]
        high_nll[0]["candidate_nll_sum"] = 400.0
        self.assertEqual(self.goal.quality_verdict(high_nll)["verdict"], "FAIL")
        top1_loss = [dict(case) for case in passing]
        for case in top1_loss:
            case["candidate_top1_correct"] = 69
        self.assertEqual(self.goal.quality_verdict(top1_loss)["verdict"], "FAIL")
        for mutation in (
            passing[:99],
            passing + [dict(passing[0])],
            [{**passing[0], "tokens": 0}] * 100,
            [{**passing[0], "candidate_nll_sum": math.nan}] * 100,
        ):
            with self.assertRaises(ValueError):
                self.goal.quality_verdict(mutation)

    def _w1_affine_campaign(self):
        seed = "00" * 32
        fixture = "1" * 64
        binary = "2" * 64
        configuration = "3" * 64
        baseline_environment = "4" * 64
        candidate_environment = "5" * 64
        attempts = []
        schedules = ("ABBA", "BAAB", "ABBA", "BAAB", "ABBA")
        for block, schedule in enumerate(schedules):
            baseline_cases = [
                {
                    "case_id": f"case-{block:02d}-{index:02d}",
                    "tokens": 100,
                    "nll_sum": 200.0,
                    "top1_correct": 70,
                }
                for index in range(20)
            ]
            candidate_cases = [
                {
                    **case,
                    "nll_sum": 200.5,
                }
                for case in baseline_cases
            ]
            for sequence, arm in enumerate(schedule):
                candidate = arm == "A"
                attempts.append(
                    {
                        "block": block,
                        "sequence": sequence,
                        "arm": arm,
                        "server_instance_id": (
                            f"server-{block}-{sequence}-{arm}"
                        ),
                        "binary_sha256": binary,
                        "configuration_sha256": configuration,
                        "fixture_sha256_before": fixture,
                        "fixture_sha256_after": fixture,
                        "environment_sha256": (
                            candidate_environment
                            if candidate
                            else baseline_environment
                        ),
                        "resolved_mode": 2 if candidate else 0,
                        "affine_store_count": 2000 if candidate else 0,
                        "completed": True,
                        "available_memory_gib": 88.0,
                        "swap_bytes": 0,
                        "oom": False,
                        "xid": False,
                        "failures": [],
                        "cases": (
                            copy.deepcopy(
                                candidate_cases if candidate else baseline_cases
                            )
                        ),
                    }
                )
        return {
            "record_type": "w1_affine_campaign",
            "engine_candidate_hash": "a" * 40,
            "seed_sha256": seed,
            "binary_sha256": binary,
            "configuration_sha256": configuration,
            "fixture_sha256": fixture,
            "baseline_environment_sha256": baseline_environment,
            "candidate_environment_sha256": candidate_environment,
            "candidate_arm": "A",
            "attempts": attempts,
        }

    def test_legacy_w1_affine_scorer_is_not_registered(self):
        campaign = self._w1_affine_campaign()
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W1", "w1.affine-quality.v1", [campaign]
            )

    def test_legacy_w1_affine_scorer_cannot_be_reenabled_by_mutation(self):
        def rejected(mutator):
            campaign = self._w1_affine_campaign()
            mutator(campaign)
            with self.assertRaises(ValueError):
                self.goal.score_registered_gate(
                    "W1", "w1.affine-quality.v1", [campaign]
                )

        rejected(lambda value: value["attempts"].pop())
        rejected(
            lambda value: value["attempts"][0].__setitem__("sequence", 1)
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "server_instance_id",
                value["attempts"][1]["server_instance_id"],
            )
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "fixture_sha256_after", "f" * 64
            )
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "environment_sha256",
                value["baseline_environment_sha256"],
            )
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "resolved_mode", 0
            )
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "affine_store_count", 0
            )
        )
        rejected(
            lambda value: value["attempts"][0].__setitem__(
                "completed", False
            )
        )
        rejected(
            lambda value: value["attempts"][0]["failures"].append(
                "injected failure"
            )
        )
        rejected(
            lambda value: value["attempts"][0]["cases"][0].__setitem__(
                "case_id",
                value["attempts"][0]["cases"][1]["case_id"],
            )
        )
        rejected(
            lambda value: value["attempts"][2]["cases"][0].__setitem__(
                "nll_sum", 999.0
            )
        )
        rejected(
            lambda value: value.__setitem__(
                "candidate_environment_sha256",
                value["baseline_environment_sha256"],
            )
        )

    def test_raw_arm_validation_fails_closed(self):
        valid = {
            "arm": "A",
            "fixture_sha256": "a" * 64,
            "binary_sha256": "b" * 64,
            "token_timestamps": [float(i) for i in range(128)],
            "evaluated_tokens": 1000,
            "prefill_seconds": 10.0,
            "failures": [],
        }
        self.goal.validate_raw_record(valid)
        mutations = (
            {"arm": ""},
            {"fixture_sha256": "bad"},
            {"binary_sha256": "bad"},
            {"token_timestamps": [float(i) for i in range(127)]},
            {"token_timestamps": [0.0] * 128},
            {"evaluated_tokens": 0},
            {"prefill_seconds": math.nan},
            {"failures": ["timeout"]},
        )
        for mutation in mutations:
            record = dict(valid)
            record.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.validate_raw_record(record)

    def test_five_fresh_server_blocks_require_abba_baab_and_distinct_arms(self):
        records = []
        for block in range(5):
            order = "ABBA" if block % 2 == 0 else "BAAB"
            for sequence, arm in enumerate(order):
                records.append(
                    {
                        "block": block,
                        "sequence": sequence,
                        "arm": arm,
                        "server_boot_id": f"boot-{block}-{sequence}",
                        "fixture_sha256": "a" * 64,
                        "binary_sha256": ("b" if arm == "A" else "c") * 64,
                        "configuration_sha256": ("d" if arm == "A" else "e") * 64,
                    }
                )
        self.goal.validate_ab_blocks(records)
        flipped = [dict(item) for item in records]
        for item in flipped:
            item["arm"] = "B" if item["arm"] == "A" else "A"
            item["binary_sha256"] = (
                "b" if item["arm"] == "A" else "c"
            ) * 64
            item["configuration_sha256"] = (
                "d" if item["arm"] == "A" else "e"
            ) * 64
        self.goal.validate_ab_blocks(flipped, flip=True)
        with self.assertRaises(ValueError):
            self.goal.validate_ab_blocks(flipped, flip=False)
        rotating = [dict(item) for item in records]
        for item in rotating:
            item["binary_sha256"] = (
                f"{item['block'] + (0 if item['arm'] == 'A' else 8):x}" * 64
            )[:64]
        with self.assertRaises(ValueError):
            self.goal.validate_ab_blocks(rotating)
        for mutation in ("same_binary", "same_boot", "wrong_order", "unequal_fixture"):
            broken = [dict(item) for item in records]
            if mutation == "same_binary":
                for item in broken:
                    item["binary_sha256"] = "b" * 64
                    item["configuration_sha256"] = "d" * 64
            elif mutation == "same_boot":
                broken[-1]["server_boot_id"] = broken[-2]["server_boot_id"]
            elif mutation == "wrong_order":
                broken[0]["arm"] = "B"
            else:
                broken[-1]["fixture_sha256"] = "f" * 64
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.validate_ab_blocks(broken)

    def test_registered_w11_scorer_is_derived_from_exact_raw_record(self):
        passing = w11_record()
        result = self.goal.score_registered_gate(
            "W11", "w11.context.v1", [passing]
        )
        self.assertEqual(result["verdict"], "PASS")
        failed = json.loads(json.dumps(passing))
        failed["stages"][-1]["processed_tokens"] = 999_999
        self.assertEqual(
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [failed]
            )["verdict"],
            "FAIL",
        )
        fail_mutations = []
        retrieval_failure = json.loads(json.dumps(passing))
        retrieval_failure["retrieval_results"][0]["observed_sha256"] = "f" * 64
        fail_mutations.append(retrieval_failure)
        negative_failure = json.loads(json.dumps(passing))
        negative_failure["negative_control_results"][0][
            "observed_sha256"
        ] = "f" * 64
        fail_mutations.append(negative_failure)
        oom_failure = json.loads(json.dumps(passing))
        oom_failure["oom_events"] = [{"event": "allocation failure"}]
        fail_mutations.append(oom_failure)
        memory_failure = json.loads(json.dumps(passing))
        memory_failure["memory_samples"][-1]["available_gib"] = 9.99
        fail_mutations.append(memory_failure)
        truncated_failure = json.loads(json.dumps(passing))
        truncated_failure["stages"][-1]["truncated"] = True
        fail_mutations.append(truncated_failure)
        length_failure = json.loads(json.dumps(passing))
        length_failure["stages"][-1]["finish_reason"] = "length"
        with self.assertRaisesRegex(ValueError, "non-truncated stop"):
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [length_failure]
            )
        swap_failure = json.loads(json.dumps(passing))
        swap_failure["memory_samples"][-1]["swap_current_bytes"] = 4096
        fail_mutations.append(swap_failure)
        for index, mutation in enumerate(fail_mutations):
            with self.subTest(fail_mutation=index):
                self.assertEqual(
                    self.goal.score_registered_gate(
                        "W11", "w11.context.v1", [mutation]
                    )["verdict"],
                    "FAIL",
                )
        malformed_timestamp = json.loads(json.dumps(passing))
        malformed_timestamp["stages"][0]["token_timestamps"][0] = "0.0"
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [malformed_timestamp]
            )
        for gate, scorer, records in (
            ("W10", "w11.context.v1", [passing]),
            ("W11", "unknown", [passing]),
            ("W11", "w11.context.v1", [passing, passing]),
            ("W11", "w11.context.v1", [{**passing, "unexpected": True}]),
            (
                "W11",
                "w11.context.v1",
                [
                    {
                        **passing,
                        "memory_samples": [
                            *passing["memory_samples"][:-1],
                            {
                                "timestamp_seconds": 7.0,
                                "available_gib": "10.0",
                                "swap_current_bytes": 0,
                            },
                        ],
                    }
                ],
            ),
            (
                "W11",
                "w11.context.v1",
                [
                    {
                        **passing,
                        "memory_samples": [
                            *passing["memory_samples"][:-1],
                            {
                                "timestamp_seconds": 7.0,
                                "available_gib": 10**10000,
                                "swap_current_bytes": 0,
                            },
                        ],
                    }
                ],
            ),
        ):
            with self.subTest(gate=gate, scorer=scorer, records=len(records)):
                with self.assertRaises(ValueError):
                    self.goal.score_registered_gate(gate, scorer, records)

    def test_w11_requires_one_direct_one_million_stage(self):
        direct = w11_record()
        direct["stages"] = [direct["stages"][-1]]
        self.assertEqual(
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [direct]
            )["verdict"],
            "PASS",
        )
        ladder = w11_record()
        lower_stage = copy.deepcopy(ladder["stages"][0])
        lower_stage["context_cap"] = 524_288
        lower_stage["processed_tokens"] = 524_288
        ladder["stages"].insert(0, lower_stage)
        with self.assertRaisesRegex(ValueError, "one direct 1M stage"):
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [ladder]
            )

    def test_registered_workstream_scorer_is_fail_closed_for_w1_w10_and_switch(self):
        for gate in [f"W{index}" for index in range(1, 11)] + ["switch"]:
            with self.subTest(gate=gate):
                passing = workstream_record(gate)
                result = self.goal.score_registered_gate(
                    gate, "workstream.terminal.v1", [passing]
                )
                self.assertEqual(result["verdict"], "FAIL")
                self.assertFalse(result["checks"]["raw_evidence_authority"])

                broken_workflow = json.loads(json.dumps(passing))
                broken_workflow["workflow"]["red_confirmed"] = False
                self.assertEqual(
                    self.goal.score_registered_gate(
                        gate, "workstream.terminal.v1", [broken_workflow]
                    )["verdict"],
                    "FAIL",
                )

                broken_metric = json.loads(json.dumps(passing))
                first_metric = next(iter(broken_metric["metrics"]))
                value = broken_metric["metrics"][first_metric]
                broken_metric["metrics"][first_metric] = (
                    False if isinstance(value, bool) else 0
                )
                self.assertEqual(
                    self.goal.score_registered_gate(
                        gate, "workstream.terminal.v1", [broken_metric]
                    )["verdict"],
                    "FAIL",
                )

        malformed = workstream_record("W2")
        malformed["unexpected"] = True
        with self.assertRaises(ValueError):
            self.goal.score_registered_gate(
                "W2", "workstream.terminal.v1", [malformed]
            )

    def test_workstream_self_attestation_cannot_create_pass(self):
        fabricated = workstream_record("W1")
        self.assertEqual(
            self.goal.score_registered_gate(
                "W1", "workstream.terminal.v1", [fabricated]
            )["verdict"],
            "FAIL",
        )

    def test_w7_resume_scorer_derives_resume_and_numerical_checks(self):
        record = w7_resume_record()
        result = self.goal.score_registered_gate(
            "W7", "w7.resume.v1", [record]
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["derived_metrics"]["max_abs_logit_delta"], 0.0)
        self.assertEqual(result["derived_metrics"]["qualified_regimes"], 3)

    def test_w7_resume_scorer_rejects_operation_receipt_false_pass(self):
        record = w7_resume_record()
        candidate = next(
            case for case in record["regimes"][0]["cases"]
            if case["case_id"] == "candidate-primary"
        )
        # A summary-only scorer cannot see this real cold reset/evaluation.
        invocation_id = candidate["invocation_id"]
        sequence = len(candidate["events"])
        candidate["events"].extend([
            {
                "seq": sequence, "invocation_id": invocation_id, "kind": "reset",
                "record_id": None, "checkpoint_tokens": None,
                "payload_sha256": None, "start": None, "end": None,
                "byte_count": 0, "status": "ok",
            },
            {
                "seq": sequence + 1, "invocation_id": invocation_id, "kind": "evaluate",
                "record_id": None, "checkpoint_tokens": None,
                "payload_sha256": None, "start": 0, "end": 5066,
                "byte_count": 0, "status": "ok",
            },
        ])
        with self.assertRaisesRegex(ValueError, "operation-derived mode"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [record])

    def test_w7_resume_scorer_rejects_guard_contract_and_inventory_mutations(self):
        for name, mutate in (
            (
                "swapped-flags",
                lambda row: (
                    next(case for case in row["regimes"][0]["cases"]
                         if case["case_id"] == "divergence-off").update(flag_value="1"),
                    next(case for case in row["regimes"][0]["cases"]
                         if case["case_id"] == "divergence-on").update(flag_value="unset"),
                ),
            ),
            (
                "missing-decoy",
                lambda row: next(
                    case for case in row["regimes"][0]["cases"]
                    if case["case_id"] == "candidate-primary"
                )["checkpoint_inventory"].pop(),
            ),
            (
                "wrong-selected-record",
                lambda row: next(
                    event for case in row["regimes"][0]["cases"]
                    if case["case_id"] == "candidate-primary"
                    for event in case["events"] if event["kind"] == "matcher_selected"
                ).update(record_id="older"),
            ),
            (
                "legacy-guard-off",
                lambda row: row.update(guard_off_present=True),
            ),
        ):
            with self.subTest(name=name):
                record = w7_resume_record()
                mutate(record)
                _rebind_w7_fixture_hashes(record)
                try:
                    result = self.goal.score_registered_gate(
                        "W7", "w7.resume.v1", [record]
                    )
                except ValueError:
                    continue
                self.assertEqual(result["verdict"], "FAIL")

    def test_w7_resume_scorer_rejects_lineage_output_state_and_safety_mutations(self):
        mutations = []
        lineage = w7_resume_record()
        next(case for case in lineage["regimes"][0]["cases"]
             if case["case_id"] == "cold-primary")["sync_tokens_sha256"] = "9" * 64
        mutations.append(lineage)
        output = w7_resume_record()
        next(case for case in output["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["output_token_ids"][-1] = 154880
        mutations.append(output)
        state = w7_resume_record()
        next(case for case in state["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["state_manifest"]["sections"].pop()
        mutations.append(state)
        unsafe = w7_resume_record()
        next(case for case in unsafe["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["safety"]["start_available_gib"] = 109.9
        mutations.append(unsafe)
        for mutation in mutations:
            try:
                result = self.goal.score_registered_gate(
                    "W7", "w7.resume.v1", [mutation]
                )
            except ValueError:
                continue
            self.assertEqual(result["verdict"], "FAIL")

    def test_w7_resume_scorer_rejects_receipt_order_complete_state_and_confirmation_mutations(self):
        mutations = []
        reordered = w7_resume_record()
        candidate = next(case for case in reordered["regimes"][0]["cases"]
                         if case["case_id"] == "candidate-primary")
        candidate["events"][4], candidate["events"][5] = candidate["events"][5], candidate["events"][4]
        for index, event in enumerate(candidate["events"]):
            event["seq"] = index
        mutations.append(reordered)

        unprobed_state = w7_resume_record()
        manifest = next(case for case in unprobed_state["regimes"][0]["cases"]
                        if case["case_id"] == "candidate-primary")["state_manifest"]
        manifest["sections"][40]["content_sha256"] = "6" * 64
        manifest["content_sha256"] = hashlib.sha256(
            "".join(item["content_sha256"] for item in manifest["sections"]).encode()
        ).hexdigest()
        mutations.append(unprobed_state)

        truncated_member = w7_resume_record()
        manifest = next(case for case in truncated_member["regimes"][0]["cases"]
                        if case["case_id"] == "candidate-primary")["state_manifest"]
        section_index = next(
            index for index, section in enumerate(manifest["sections"])
            if section["name"] == "layer.00.kv_lora"
        )
        section = manifest["sections"][section_index]
        removed = section["byte_count"] - 4
        section["shape"] = [1]
        section["byte_count"] = 4
        for later in manifest["sections"][section_index + 1:]:
            later["offset"] -= removed
        manifest["total_bytes"] -= removed
        mutations.append(truncated_member)

        duplicate_confirmation = w7_resume_record()
        duplicate_confirmation["regimes"][2]["fixture_sha256"] = duplicate_confirmation["regimes"][1]["fixture_sha256"]
        mutations.append(duplicate_confirmation)

        bad_wrapper = w7_resume_record()
        next(case for case in bad_wrapper["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["safety"]["wrapper_sha256"] = "5" * 64
        mutations.append(bad_wrapper)

        flag_empty = w7_resume_record()
        next(case for case in flag_empty["regimes"][0]["cases"]
             if case["case_id"] == "flag-empty")["flag_value"] = "unset"
        mutations.append(flag_empty)

        for mutation in mutations:
            try:
                result = self.goal.score_registered_gate("W7", "w7.resume.v1", [mutation])
            except ValueError:
                continue
            self.assertEqual(result["verdict"], "FAIL")

    def test_w7_resume_scorer_rejects_partial_reads_ambiguous_selection_and_live_lineage(self):
        mutations = []
        expected_payload = _w7_payload_bytes(5044)
        for byte_count in (1, expected_payload - 1, expected_payload + 1):
            record = w7_resume_record()
            candidate = next(case for case in record["regimes"][0]["cases"]
                             if case["case_id"] == "candidate-primary")
            next(event for event in candidate["events"]
                 if event["kind"] == "payload_read")["byte_count"] = byte_count
            mutations.append(record)

        ambiguous = w7_resume_record()
        candidate = next(case for case in ambiguous["regimes"][0]["cases"]
                         if case["case_id"] == "candidate-primary")
        older, selected = candidate["checkpoint_inventory"][:2]
        for field in (
            "token_length", "rendered_prefix_bytes", "rendered_prefix_sha256",
            "token_sha256", "lineage_sha256", "matches_wire_prefix",
            "matches_lineage",
        ):
            older[field] = selected[field]
        candidate["events"][0]["checkpoint_tokens"] = selected["token_length"]
        _rebind_w7_fixture_hashes(ambiguous)
        mutations.append(ambiguous)

        wrong_live = w7_resume_record()
        candidate = next(case for case in wrong_live["regimes"][0]["cases"]
                         if case["case_id"] == "candidate-primary")
        count = candidate["live_token_count"]
        raw = zlib.decompress(base64.b64decode(candidate["live_token_ids_zlib_b64"]))
        values = list(struct.unpack(f"<{count}i", raw))
        values[100] = (values[100] + 1) % 154880
        candidate["live_token_ids_zlib_b64"] = _compressed_i32(values)
        _rebind_w7_fixture_hashes(wrong_live)
        mutations.append(wrong_live)

        multiple_reads = w7_resume_record()
        candidate = next(case for case in multiple_reads["regimes"][0]["cases"]
                         if case["case_id"] == "candidate-primary")
        duplicate = copy.deepcopy(next(event for event in candidate["events"]
                                       if event["kind"] == "payload_read"))
        candidate["events"].insert(6, duplicate)
        for index, event in enumerate(candidate["events"]):
            event["seq"] = index
        mutations.append(multiple_reads)

        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    self.goal.score_registered_gate("W7", "w7.resume.v1", [mutation])

    def test_w7_resume_scorer_rejects_eval_gaps_cross_case_reuse_and_replay_drift(self):
        mutations = []
        for edit in ("gap", "overlap", "hidden-prefix"):
            record = w7_resume_record()
            candidate = next(case for case in record["regimes"][0]["cases"]
                             if case["case_id"] == "candidate-primary")
            evaluations = [event for event in candidate["events"]
                           if event["kind"] == "evaluate"]
            if edit == "gap":
                evaluations[1]["start"] += 1
            elif edit == "overlap":
                evaluations[1]["start"] -= 1
            else:
                evaluations[0]["start"] = 0
            mutations.append(record)

        reused = w7_resume_record()
        exact_off = next(case for case in reused["regimes"][0]["cases"]
                         if case["case_id"] == "exact-off")
        exact_on = next(case for case in reused["regimes"][0]["cases"]
                        if case["case_id"] == "exact-on")
        exact_on["invocation_id"] = exact_off["invocation_id"]
        mutations.append(reused)

        replay_drift = w7_resume_record()
        exact_on = next(case for case in replay_drift["regimes"][0]["cases"]
                        if case["case_id"] == "exact-on")
        exact_on["output_token_ids"][-1] += 1
        mutations.append(replay_drift)

        cloned_confirmation = w7_resume_record()
        source = cloned_confirmation["regimes"][1]
        target = cloned_confirmation["regimes"][2]
        fixture_keys = (
            "wire_text_utf8_b64", "wire_token_end_offsets_zlib_b64",
            "canonical_token_ids_zlib_b64", "live_token_count",
            "live_token_ids_zlib_b64", "prompt_tokens", "checkpoint_inventory",
        )
        for source_case, target_case in zip(source["cases"], target["cases"]):
            for key in fixture_keys:
                target_case[key] = copy.deepcopy(source_case[key])
        _rebind_w7_fixture_hashes(cloned_confirmation)
        mutations.append(cloned_confirmation)

        for mutation in mutations:
            try:
                result = self.goal.score_registered_gate("W7", "w7.resume.v1", [mutation])
            except ValueError:
                continue
            self.assertEqual(result["verdict"], "FAIL")

    def test_w7_resume_scorer_rejects_confirmation_that_removes_divergent_append(self):
        record = w7_resume_record()
        for regime in record["regimes"][1:]:
            for case_id in ("strict-primary", "candidate-primary"):
                case = next(
                    item for item in regime["cases"] if item["case_id"] == case_id
                )
                raw = zlib.decompress(
                    base64.b64decode(case["canonical_token_ids_zlib_b64"])
                )
                canonical = list(
                    struct.unpack(f"<{case['prompt_tokens']}i", raw)
                )
                case["live_token_count"] = len(canonical)
                case["live_token_ids_zlib_b64"] = _compressed_i32(canonical)
        _rebind_w7_fixture_hashes(record)
        with self.assertRaisesRegex(ValueError, "frozen production live lineage"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [record])

    def test_w7_resume_scorer_rejects_post_seed_fixture_and_cross_attempt_receipts(self):
        post_seed = w7_resume_record()
        case = post_seed["regimes"][1]["cases"][0]
        wire = base64.b64decode(case["wire_text_utf8_b64"]) + b" post-seed"
        case["wire_text_utf8_b64"] = base64.b64encode(wire).decode("ascii")
        offsets_raw = zlib.decompress(
            base64.b64decode(case["wire_token_end_offsets_zlib_b64"])
        )
        offsets = list(struct.unpack(f"<{case['prompt_tokens']}i", offsets_raw))
        offsets[-1] = len(wire)
        case["wire_token_end_offsets_zlib_b64"] = _compressed_i32(offsets)
        _rebind_w7_fixture_hashes(post_seed)
        with self.assertRaisesRegex(ValueError, "not frozen and seed-selected"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [post_seed])

        stale_attempt = w7_resume_record()
        stale_attempt["attempt_id"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "attempt identity"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [stale_attempt])

    def test_w7_resume_scorer_requires_production_tokens_and_derived_attempt_identity(self):
        record = w7_resume_record()
        rerun = w7_resume_record("d" * 64)
        self.assertNotEqual(record["attempt_id"], rerun["attempt_id"])
        first_invocations = {
            case["invocation_id"] for regime in record["regimes"] for case in regime["cases"]
        }
        rerun_invocations = {
            case["invocation_id"] for regime in rerun["regimes"] for case in regime["cases"]
        }
        self.assertTrue(first_invocations.isdisjoint(rerun_invocations))
        self.assertEqual(
            self.goal.score_registered_gate("W7", "w7.resume.v1", [rerun])["verdict"],
            "PASS",
        )
        expected = {
            "confirmation-1": (
                5064,
                "ffadfb3cdb935fac755a5dfd1d24f20712ca9ede93639e4d054e54c55f35f440",
                "b0d3d893e1a4a0fb52f55233bffa158065b456bded386e855f1d66dd2a396585",
            ),
            "confirmation-2": (
                5066,
                "a5a8a6f5f171fbbf503eac01fda2e60e8eeccfade88218bcaab823db237f60d4",
                "b9a235f59d267fb6d85528a093df642f9352bd3de5594bcf7cdf1e79d285b4b8",
            ),
        }
        for regime in record["regimes"][1:]:
            case = regime["cases"][0]
            count, token_sha, offset_sha = expected[regime["regime_id"]]
            self.assertEqual(case["prompt_tokens"], count)
            token_raw = zlib.decompress(
                base64.b64decode(case["canonical_token_ids_zlib_b64"])
            )
            offset_raw = zlib.decompress(
                base64.b64decode(case["wire_token_end_offsets_zlib_b64"])
            )
            self.assertEqual(hashlib.sha256(token_raw).hexdigest(), token_sha)
            self.assertEqual(hashlib.sha256(offset_raw).hexdigest(), offset_sha)

        renamed = w7_resume_record()
        old_attempt = renamed["attempt_id"]
        renamed["attempt_id"] = "f" * 64
        for regime in renamed["regimes"]:
            for case in regime["cases"]:
                case["invocation_id"] = case["invocation_id"].replace(
                    old_attempt, renamed["attempt_id"], 1
                )
                for event in case["events"]:
                    event["invocation_id"] = case["invocation_id"]
        with self.assertRaisesRegex(ValueError, "attempt identity"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [renamed])

    def test_w7_resume_scorer_rejects_malformed_or_nonfinite_artifacts(self):
        malformed = w7_resume_record()
        next(case for case in malformed["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["logits_f32_zlib_b64"] = "not-base64"
        with self.assertRaisesRegex(ValueError, "compressed F32"):
            self.goal.score_registered_gate(
                "W7", "w7.resume.v1", [malformed]
            )
        nonfinite = w7_resume_record()
        raw = bytearray(154880 * 4)
        struct.pack_into("<f", raw, 0, float("nan"))
        next(case for case in nonfinite["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["logits_f32_zlib_b64"] = base64.b64encode(
            zlib.compress(bytes(raw), 9)
        ).decode("ascii")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.goal.score_registered_gate(
                "W7", "w7.resume.v1", [nonfinite]
            )

        bomb = w7_resume_record()
        next(case for case in bomb["regimes"][0]["cases"]
             if case["case_id"] == "candidate-primary")["logits_f32_zlib_b64"] = base64.b64encode(
            zlib.compress(b"x" * (154880 * 16), 9)
        ).decode("ascii")
        with self.assertRaisesRegex(ValueError, "compressed F32|wrong F32 length"):
            self.goal.score_registered_gate("W7", "w7.resume.v1", [bomb])

    def test_w11_requires_timestamped_memory_coverage(self):
        record = w11_record()
        self.assertEqual(
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [record]
            )["verdict"],
            "PASS",
        )
        missing_coverage = json.loads(json.dumps(record))
        missing_coverage["memory_samples"] = missing_coverage[
            "memory_samples"
        ][::4]
        with self.assertRaisesRegex(ValueError, "cover execution at 4 Hz"):
            self.goal.score_registered_gate(
                "W11", "w11.context.v1", [missing_coverage]
            )

    def test_registered_parity_scorer_recomputes_five_block_bounds(self):
        rows = []
        for block in range(5):
            order = "ABBA" if block % 2 == 0 else "BAAB"
            for sequence, arm in enumerate(order):
                glm = arm == "A"
                decode = 8.4 if glm else 10.0
                rows.append(
                    {
                        "record_type": "matched_arm",
                        "block": block,
                        "sequence": sequence,
                        "arm": arm,
                        "profile": "glm52" if glm else "dsv4",
                        "server_boot_id": f"boot-{block}-{sequence}",
                        "fixture_sha256": "a" * 64,
                        "binary_sha256": ("b" if glm else "c") * 64,
                        "configuration_sha256": ("d" if glm else "e") * 64,
                        "token_timestamps": [
                            index / decode for index in range(128)
                        ],
                        "evaluated_tokens": 1000,
                        "prefill_seconds": 11.9 if glm else 10.0,
                        "warm_ttft_seconds": 1.15 if glm else 1.0,
                        "cold_ttft_seconds": 11.5 if glm else 10.0,
                        "available_memory_gib": 20.0,
                        "truncated": False,
                        "oom": False,
                        "xid": False,
                        "failures": [],
                    }
                )
        result = self.goal.score_registered_gate(
            "parity", "parity.performance.v1", rows
        )
        self.assertEqual(result["verdict"], "PASS")
        broken = [dict(row) for row in rows]
        broken[0]["token_timestamps"] = [
            index / 7.0 for index in range(128)
        ]
        self.assertEqual(
            self.goal.score_registered_gate(
                "parity", "parity.performance.v1", broken
            )["verdict"],
            "FAIL",
        )
        malformed = [dict(row) for row in rows]
        malformed[0]["token_timestamps"] = [
            str(value) for value in malformed[0]["token_timestamps"]
        ]
        with self.assertRaisesRegex(ValueError, "exact numeric"):
            self.goal.score_registered_gate(
                "parity", "parity.performance.v1", malformed
            )
        for mutation in ("short", "duplicate_boot", "wrong_profile", "oom"):
            malformed = [dict(row) for row in rows]
            if mutation == "short":
                malformed[0]["token_timestamps"] = [0.0] * 127
            elif mutation == "duplicate_boot":
                malformed[-1]["server_boot_id"] = malformed[-2]["server_boot_id"]
            elif mutation == "wrong_profile":
                malformed[0]["profile"] = "dsv4"
            else:
                malformed[0]["oom"] = True
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.score_registered_gate(
                        "parity", "parity.performance.v1", malformed
                    )

    def test_reviewed_no_go_requires_decisive_matched_failure_and_clean_reviews(self):
        candidate = "a" * 40
        arms = []
        for block in range(5):
            order = "ABBA" if block % 2 == 0 else "BAAB"
            for sequence, arm in enumerate(order):
                glm = arm == "A"
                decode = 1.0 if glm else 10.0
                arms.append(
                    {
                        "record_type": "matched_arm",
                        "block": block,
                        "sequence": sequence,
                        "arm": arm,
                        "profile": "glm52" if glm else "dsv4",
                        "server_boot_id": f"boot-{block}-{sequence}",
                        "fixture_sha256": "a" * 64,
                        "binary_sha256": ("b" if glm else "c") * 64,
                        "configuration_sha256": ("d" if glm else "e") * 64,
                        "token_timestamps": [
                            index / decode for index in range(128)
                        ],
                        "evaluated_tokens": 1000,
                        "prefill_seconds": 20.0 if glm else 10.0,
                        "warm_ttft_seconds": 2.0 if glm else 1.0,
                        "cold_ttft_seconds": 20.0 if glm else 10.0,
                        "available_memory_gib": 20.0,
                        "truncated": False,
                        "oom": False,
                        "xid": False,
                        "failures": [],
                    }
                )
        measurement_digest = self.goal.reviewed_measurements_digest(arms)
        reviews = [
            {
                "record_type": "no_go_review",
                "reviewer": reviewer,
                "candidate_hash": candidate,
                "review_round": 1,
                "reviewed_measurements_sha256": measurement_digest,
                "claimed_score": score,
                "critical": [],
                "high": [],
                "medium": [],
                "low": [],
                "prior_issue_status": [],
                "verdict": "REJECT",
            }
            for reviewer, score in (
                ("gap_reviewer", 0),
                ("adversarial_reviewer", 1),
            )
        ]
        result = self.goal.score_registered_gate(
            "parity", "parity.reviewed-no-go.v1", arms + reviews
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["decision"], "NO_GO")
        self.assertTrue(result["checks"]["decisive_matched_failure"])
        self.assertNotIn("scores_at_least_90", result["checks"])

        high = json.loads(json.dumps(arms + reviews))
        high[-1]["high"] = [
            {
                "id": "H-001",
                "evidence": "unresolved high issue",
                "affected_gate": "parity",
                "reproduction_instructions": "reproduce the high issue",
                "proposed_acceptance_test": "prove the issue is fixed",
            }
        ]
        high[-1]["claimed_score"] = 100
        self.assertEqual(
            self.goal.score_registered_gate(
                "parity", "parity.reviewed-no-go.v1", high
            )["verdict"],
            "FAIL",
        )

        stale = json.loads(json.dumps(arms + reviews))
        stale[0]["prefill_seconds"] += 1.0
        with self.assertRaisesRegex(ValueError, "reviewed measurements"):
            self.goal.score_registered_gate(
                "parity", "parity.reviewed-no-go.v1", stale
            )

        straddling = json.loads(json.dumps(arms))
        for arm in straddling:
            if arm["profile"] == "glm52":
                ratio = 0.6 + 0.1 * arm["block"]
                arm["token_timestamps"] = [
                    index / (10.0 * ratio) for index in range(128)
                ]
                arm["prefill_seconds"] = 10.0
                arm["warm_ttft_seconds"] = 1.0
                arm["cold_ttft_seconds"] = 10.0
        straddling_digest = self.goal.reviewed_measurements_digest(straddling)
        straddling_reviews = json.loads(json.dumps(reviews))
        for review in straddling_reviews:
            review["reviewed_measurements_sha256"] = straddling_digest
        inconclusive = self.goal.score_registered_gate(
            "parity",
            "parity.reviewed-no-go.v1",
            straddling + straddling_reviews,
        )
        self.assertEqual(inconclusive["parity"]["verdict"], "FAIL")
        self.assertFalse(inconclusive["checks"]["decisive_matched_failure"])
        self.assertEqual(inconclusive["verdict"], "FAIL")

    def test_registered_foundation_scorer_requires_clean_safe_baselines(self):
        def baseline(profile, hash_char, spacing):
            return {
                "profile": profile,
                "server_instance_id": f"{profile}-fresh-1",
                "fixture_sha256": "a" * 64,
                "binary_sha256": hash_char * 64,
                "configuration_sha256": chr(ord(hash_char) + 2) * 64,
                "token_timestamps": [
                    index * spacing for index in range(128)
                ],
                "evaluated_tokens": 1000,
                "prefill_seconds": 10.0,
                "warm_ttft_seconds": 1.0,
                "cold_ttft_seconds": 10.0,
                "available_memory_gib": 20.0,
                "truncated": False,
                "oom": False,
                "xid": False,
                "failures": [],
            }

        passing = {
            "record_type": "foundation_observation",
            "upstream_commit": "b" * 40,
            "source_clean": True,
            "clean_build": True,
            "model_artifacts_verified": True,
            "tokenizer_artifacts_verified": True,
            "bandwidth_gb_s": [105.0, 106.0, 107.0, 108.0, 109.0],
            "glm_baseline": baseline("glm52", "b", 0.12),
            "dsv4_baseline": baseline("dsv4", "c", 0.10),
        }
        result = self.goal.score_registered_gate(
            "foundation", "foundation.v1", [passing]
        )
        self.assertEqual(result["verdict"], "PASS")
        for mutation in (
            "dirty",
            "short_bandwidth",
            "same_identity",
            "short_decode",
            "oom",
            "unequal_fixture",
        ):
            broken = json.loads(json.dumps(passing))
            if mutation == "dirty":
                broken["source_clean"] = False
            elif mutation == "short_bandwidth":
                broken["bandwidth_gb_s"] = broken["bandwidth_gb_s"][:4]
            elif mutation == "same_identity":
                broken["dsv4_baseline"]["binary_sha256"] = (
                    broken["glm_baseline"]["binary_sha256"]
                )
                broken["dsv4_baseline"]["configuration_sha256"] = (
                    broken["glm_baseline"]["configuration_sha256"]
                )
            elif mutation == "short_decode":
                broken["glm_baseline"]["token_timestamps"] = [0.0] * 127
            elif mutation == "oom":
                broken["glm_baseline"]["oom"] = True
            else:
                broken["glm_baseline"]["fixture_sha256"] = "f" * 64
            with self.subTest(mutation=mutation):
                if mutation == "dirty":
                    self.assertEqual(
                        self.goal.score_registered_gate(
                            "foundation", "foundation.v1", [broken]
                        )["verdict"],
                        "FAIL",
                    )
                else:
                    with self.assertRaises(ValueError):
                        self.goal.score_registered_gate(
                            "foundation", "foundation.v1", [broken]
                        )

    def test_registered_review_scorer_accepts_reviewer_assigned_scores(self):
        candidate = "a" * 40

        def issue(issue_id):
            return {
                "id": issue_id,
                "evidence": f"evidence for {issue_id}",
                "affected_gate": "W11",
                "reproduction_instructions": f"reproduce {issue_id}",
                "proposed_acceptance_test": f"test {issue_id}",
            }

        def review(reviewer):
            return {
                "record_type": "review",
                "reviewer": reviewer,
                "candidate_hash": candidate,
                "review_round": 7,
                # Reviewers score holistically; issue counts do not determine it.
                "claimed_score": 93,
                "critical": [],
                "high": [],
                "medium": [issue("M-001")],
                "low": [issue("L-001")],
                "prior_issue_status": [
                    {"id": "OLD-001", "status": "FIXED"},
                    {"id": "OLD-002", "status": "FALSIFIED"},
                ],
                "verdict": "ACCEPT",
            }

        rows = [review("gap_reviewer"), review("adversarial_reviewer")]
        result = self.goal.score_registered_gate(
            "review", "review.final.v1", rows
        )
        self.assertEqual(result["verdict"], "PASS")
        low_score = json.loads(json.dumps(rows))
        low_score[0]["claimed_score"] = 12
        low_score[0]["verdict"] = "REJECT"
        low_score[1]["claimed_score"] = 47
        low_score[1]["verdict"] = "REJECT"
        low_score_result = self.goal.score_registered_gate(
            "review", "review.final.v1", low_score
        )
        self.assertEqual(low_score_result["verdict"], "PASS")
        self.assertEqual(
            low_score_result["scores"],
            {"gap_reviewer": 12, "adversarial_reviewer": 47},
        )
        self.assertNotIn(
            "both_scores_at_least_90", low_score_result["checks"]
        )
        for mutation in (
            "one_reviewer",
            "wrong_name",
            "candidate_mismatch",
            "score_out_of_range",
            "score_boolean",
            "critical",
            "duplicate_prior",
        ):
            broken = json.loads(json.dumps(rows))
            if mutation == "one_reviewer":
                broken = broken[:1]
            elif mutation == "wrong_name":
                broken[0]["reviewer"] = "fresh_reviewer"
            elif mutation == "candidate_mismatch":
                broken[0]["candidate_hash"] = "b" * 40
            elif mutation == "score_out_of_range":
                broken[0]["claimed_score"] = 101
            elif mutation == "score_boolean":
                broken[0]["claimed_score"] = True
            elif mutation == "critical":
                broken[0]["critical"] = [issue("C-001")]
                broken[0]["claimed_score"] = 99
                broken[0]["verdict"] = "REJECT"
            else:
                broken[0]["prior_issue_status"].append(
                    {"id": "OLD-001", "status": "FIXED"}
                )
            with self.subTest(mutation=mutation):
                if mutation == "critical":
                    self.assertEqual(
                        self.goal.score_registered_gate(
                            "review", "review.final.v1", broken
                        )["verdict"],
                        "FAIL",
                    )
                else:
                    with self.assertRaises(ValueError):
                        self.goal.score_registered_gate(
                            "review", "review.final.v1", broken
                        )

    def test_review_scorer_summary_version_is_accepted_end_to_end(self):
        candidate = "a" * 40
        records = [
            {
                "record_type": "review",
                "reviewer": reviewer,
                "candidate_hash": candidate,
                "review_round": 1,
                "claimed_score": score,
                "critical": [],
                "high": [],
                "medium": [],
                "low": [],
                "prior_issue_status": [],
                "verdict": "ACCEPT",
            }
            for reviewer, score in (
                ("gap_reviewer", 37),
                ("adversarial_reviewer", 82),
            )
        ]
        scorer_id = "review.final.v1"
        summary = self.goal.score_registered_gate(
            "review", scorer_id, records
        )
        self.assertEqual(summary["formula_version"], 3)
        implementation_digest = self.goal.registered_scorer_digest(scorer_id)
        descriptor = {
            "schema_version": 1,
            "scorer_id": scorer_id,
            "implementation_sha256": implementation_digest,
        }
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            contents = {
                name: (
                    json.dumps(descriptor, sort_keys=True).encode()
                    if name == "scorer"
                    else f"{name}-artifact".encode()
                )
                for name in (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            }
            manifest = {
                "gate": "review",
                "candidate_hash": candidate,
                "lineage": {},
                "artifacts": {},
            }
            for name, content in contents.items():
                path = attempt / f"{name}.artifact"
                path.write_bytes(content)
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(content).hexdigest()
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records)
            )
            (attempt / "summary.json").write_text(json.dumps(summary))
            candidate_ok = subprocess.CompletedProcess([], 0)
            with (
                mock.patch.object(
                    self.goal.subprocess, "run", return_value=candidate_ok
                ),
                mock.patch.object(self.goal, "validate_manifest_lineage"),
                mock.patch.object(self.goal, "validate_source_provenance"),
                mock.patch.object(
                    self.goal, "validate_profile_artifact_bindings"
                ),
                mock.patch.object(
                    self.goal, "validate_record_artifact_bindings"
                ),
                mock.patch.object(
                    self.goal,
                    "registered_scorer_digest",
                    return_value=implementation_digest,
                ),
            ):
                self.goal.validate_attempt(attempt)

    def test_registered_scorer_identity_is_function_scoped(self):
        digests = {
            scorer: self.goal.registered_scorer_digest(scorer)
            for scorer in (
                "foundation.v1",
                "w11.context.v1",
                "parity.performance.v1",
                "parity.reviewed-no-go.v1",
                "review.final.v1",
                "workstream.terminal.v1",
            )
        }
        self.assertEqual(len(set(digests.values())), 6)
        for digest in digests.values():
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
        with self.assertRaises(ValueError):
            self.goal.registered_scorer_digest("unknown")

    def test_exact_frozen_w1_scorer_digest_survives_unrelated_updates(self):
        frozen_candidate = "3879eb01a2a427be76373b847d832738f1f86552"
        frozen_digest = (
            "b322d78612d51eb714039c38fe79d512"
            "1428681610815d7453dbb6e69ad5a1e6"
        )
        self.assertTrue(
            self.goal.scorer_descriptor_matches(
                "W1",
                frozen_candidate,
                "w1.affine-quality.v2",
                frozen_digest,
            )
        )
        self.assertFalse(
            self.goal.scorer_descriptor_matches(
                "W1",
                "0" * 40,
                "w1.affine-quality.v2",
                frozen_digest,
            )
        )
        self.assertFalse(
            self.goal.scorer_descriptor_matches(
                "W1",
                frozen_candidate,
                "w1.affine-quality.v2",
                "0" * 64,
            )
        )

    def test_scorer_identity_changes_with_attempt_validation(self):
        before = self.goal.registered_scorer_digest("w11.context.v1")
        original = self.goal.validate_attempt
        try:
            self.goal.validate_attempt = replacement_attempt_validator
            after = self.goal.registered_scorer_digest("w11.context.v1")
        finally:
            self.goal.validate_attempt = original
        self.assertNotEqual(before, after)

    def test_scorer_identity_covers_numeric_and_lineage_dependencies(self):
        before = self.goal.registered_scorer_digest("w11.context.v1")
        originals = (
            self.goal._finite_number,
            self.goal._utc_timestamp,
            self.goal.generate_w11_fixture,
            self.goal._load_approved_dsv4_profile,
        )
        try:
            self.goal._finite_number = replacement_finite_number
            finite_digest = self.goal.registered_scorer_digest(
                "w11.context.v1"
            )
            self.goal._finite_number = originals[0]
            self.goal._utc_timestamp = replacement_utc_timestamp
            utc_digest = self.goal.registered_scorer_digest("w11.context.v1")
            self.goal._utc_timestamp = originals[1]
            self.goal.generate_w11_fixture = replacement_w11_fixture
            fixture_digest = self.goal.registered_scorer_digest(
                "w11.context.v1"
            )
            self.goal.generate_w11_fixture = originals[2]
            self.goal._load_approved_dsv4_profile = replacement_dsv4_profile
            profile_digest = self.goal.registered_scorer_digest(
                "parity.performance.v1"
            )
        finally:
            (
                self.goal._finite_number,
                self.goal._utc_timestamp,
                self.goal.generate_w11_fixture,
                self.goal._load_approved_dsv4_profile,
            ) = originals
        self.assertNotEqual(before, finite_digest)
        self.assertNotEqual(before, utc_digest)
        self.assertNotEqual(before, fixture_digest)
        self.assertNotEqual(
            self.goal.registered_scorer_digest("parity.performance.v1"),
            profile_digest,
        )

    def test_manifest_lineage_requires_post_freeze_verifiable_randomness(self):
        signature = (
            "952376f4137b3dcb0798721d8a76ffe3"
            "35949d115e02cd6b6fcf97b0f748a66d"
            "e3258668f148ea29172d8438b94185e00"
            "48c3061497303a027fabec04eb0bb27e"
            "bf01e86e7b97ee4232ccea760fc867af"
            "8bfe18b0e1106915148438bb3c235f6"
        )
        candidate = "a" * 40
        randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        seed = hashlib.sha256(
            f"{candidate}:{randomness}:W11".encode()
        ).hexdigest()
        lineage = {
            "freeze": {
                "candidate_hash": candidate,
                "frozen_at": "2026-07-27T03:59:00+00:00",
            },
            "randomness": {
                "source": "drand-default",
                "round": 6_323_125,
                "randomness": randomness,
                "signature": signature,
                "obtained_at": "2026-07-27T04:00:00+00:00",
                "seed_sha256": seed,
            },
        }
        self.goal.validate_manifest_lineage(lineage, "W11", candidate)
        for mutation in (
            "pre_freeze",
            "wrong_candidate",
            "wrong_randomness",
            "wrong_seed",
            "bad_round",
            "extra",
        ):
            broken = json.loads(json.dumps(lineage))
            if mutation == "pre_freeze":
                broken["randomness"]["obtained_at"] = (
                    "2026-07-26T23:59:59+00:00"
                )
            elif mutation == "wrong_candidate":
                broken["freeze"]["candidate_hash"] = "b" * 40
            elif mutation == "wrong_randomness":
                broken["randomness"]["randomness"] = "0" * 64
            elif mutation == "wrong_seed":
                broken["randomness"]["seed_sha256"] = "0" * 64
            elif mutation == "bad_round":
                broken["randomness"]["round"] = True
            else:
                broken["freeze"]["extra"] = True
            with self.subTest(mutation=mutation):
                with self.assertRaises(ValueError):
                    self.goal.validate_manifest_lineage(
                        broken, "W11", candidate
                    )

    def test_manifest_lineage_rejects_signature_not_published_by_relays(self):
        candidate = "a" * 40
        signature = "00" * 96
        randomness = hashlib.sha256(bytes.fromhex(signature)).hexdigest()
        lineage = {
            "freeze": {
                "candidate_hash": candidate,
                "frozen_at": "2026-07-27T03:59:00+00:00",
            },
            "randomness": {
                "source": "drand-default",
                "round": 6_323_125,
                "randomness": randomness,
                "signature": signature,
                "obtained_at": "2026-07-27T04:00:00+00:00",
                "seed_sha256": hashlib.sha256(
                    f"{candidate}:{randomness}:W11".encode()
                ).hexdigest(),
            },
        }
        published = {
            "round": 6_323_125,
            "randomness": "1" * 64,
            "signature": "2" * 192,
        }
        with self.assertRaisesRegex(ValueError, "public drand relays"):
            self.goal.validate_manifest_lineage(
                lineage,
                "W11",
                candidate,
                relay_fetcher=lambda _host, _round: published,
            )

    def test_drand_fetch_disables_curl_default_configuration(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=b'{"round":1,"randomness":"a","signature":"b"}',
            stderr=b"",
        )
        with mock.patch.object(
            self.goal.subprocess, "run", return_value=completed
        ) as run:
            self.goal._fetch_public_drand("api.drand.sh", 1)
        argv = run.call_args.args[0]
        self.assertEqual(argv[:2], ["/usr/bin/curl", "--disable"])

    def test_manifest_freeze_time_is_derived_from_commit(self):
        confirmation = json.loads(
            (
                ROOT
                / "results/glm52-goal/evidence/"
                "lineage-confirmation-6996608.json"
            ).read_text()
        )
        candidate = confirmation["candidate_hash"]
        item = next(
            value
            for value in confirmation["gate_seeds"]
            if value["gate"] == "W11"
        )
        beacon = confirmation["confirmation"]
        lineage = {
            "freeze": {
                "candidate_hash": candidate,
                "frozen_at": "2020-01-01T00:00:00+00:00",
            },
            "randomness": {
                "source": beacon["source"],
                "round": beacon["round"],
                "randomness": beacon["drand_randomness"],
                "signature": beacon["drand_signature"],
                "obtained_at": beacon["obtained_at"],
                "seed_sha256": item["sha256"],
            },
        }
        published = {
            "round": beacon["round"],
            "randomness": beacon["drand_randomness"],
            "signature": beacon["drand_signature"],
        }
        with self.assertRaisesRegex(ValueError, "commit timestamp"):
            self.goal.validate_manifest_lineage(
                lineage,
                "W11",
                candidate,
                relay_fetcher=lambda _host, _round: published,
                commit_time_fetcher=lambda _candidate: (
                    "2026-07-27T04:24:33+00:00"
                ),
            )

    def test_attempt_requires_manifest_raw_and_fixed_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            artifacts = {}
            manifest = {"gate": "foundation", "candidate_hash": "a" * 40}
            for index, field in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                artifact = attempt / f"{field}.artifact"
                artifact.write_bytes(f"artifact-{index}".encode())
                artifacts[field] = artifact.name
                manifest[f"{field}_sha256"] = hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest()
            manifest["artifacts"] = artifacts
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(
                json.dumps({"event": "diagnostic", "failures": []}) + "\n"
            )
            (attempt / "summary.json").write_text(
                json.dumps(
                    {
                        "formula_version": 1,
                        "verdict": "NO_RESULT",
                        "reason": "diagnostic-only",
                    }
                )
            )
            with self.assertRaisesRegex(
                ValueError,
                "repository commit|manifest lineage|terminal scorer",
            ):
                self.goal.validate_attempt(attempt)
            (attempt / "raw.jsonl").write_text("")
            with self.assertRaises(ValueError):
                self.goal.validate_attempt(attempt)

    def test_attempt_rejects_fabricated_pass_and_artifact_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            manifest = {
                f"{name}_sha256": str(index) * 64
                for index, name in enumerate(
                    (
                        "source",
                        "diff",
                        "binary",
                        "scorer",
                        "model",
                        "tokenizer",
                        "fixture",
                        "configuration",
                    )
                )
            }
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text('{"event":"fabricated","failures":[]}\n')
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"PASS"}'
            )
            with self.assertRaises(ValueError):
                self.goal.validate_attempt(attempt)

    def test_attempt_discovery_orders_numeric_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            gate_dir = state_dir / "foundation"
            (gate_dir / "attempt-9").mkdir(parents=True)
            (gate_dir / "attempt-100").mkdir()
            state = self.goal._initial_state()
            self.goal._ingest_attempts(state_dir, state)
            self.assertEqual(
                state["gates"]["foundation"]["attempts"],
                ["foundation/attempt-9", "foundation/attempt-100"],
            )
            self.assertIn(
                "attempt-100", state["gates"]["foundation"]["reason"]
            )

    def test_attempt_validation_rejects_directory_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "outside"
            target.mkdir()
            link = root / "attempt-001"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                self.goal.validate_attempt(link)

    def test_attempt_rejects_self_consistent_but_unbound_source_artifact(self):
        candidate = "77782656208f120a59f0650699d877fd286304b3"
        confirmation = json.loads(
            (
                ROOT
                / "results/glm52-goal/evidence/"
                "lineage-confirmation-7778265.json"
            ).read_text()
        )
        with tempfile.TemporaryDirectory() as tmp:
            attempt = Path(tmp)
            scorer_id = "w11.context.v1"
            descriptor = {
                "schema_version": 1,
                "scorer_id": scorer_id,
                "implementation_sha256": self.goal.registered_scorer_digest(
                    scorer_id
                ),
            }
            contents = {
                "source": b"self-authored arbitrary source provenance",
                "diff": b"self-authored arbitrary diff",
                "binary": b"candidate binary",
                "scorer": json.dumps(descriptor, sort_keys=True).encode(),
                "model": b"candidate model",
                "tokenizer": b"candidate tokenizer",
                "fixture": b"candidate fixture",
                "configuration": b"candidate configuration",
            }
            manifest = {
                "gate": "W11",
                "candidate_hash": candidate,
                "artifacts": {},
            }
            for name, content in contents.items():
                path = attempt / f"{name}.artifact"
                path.write_bytes(content)
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(content).hexdigest()
            seed = next(
                item["sha256"]
                for item in confirmation["gate_seeds"]
                if item["gate"] == "W11"
            )
            manifest["lineage"] = {
                "freeze": {
                    "candidate_hash": candidate,
                    "frozen_at": confirmation["frozen_at"],
                },
                "randomness": {
                    "source": confirmation["confirmation"]["source"],
                    "round": confirmation["confirmation"]["round"],
                    "randomness": confirmation["confirmation"][
                        "drand_randomness"
                    ],
                    "signature": confirmation["confirmation"][
                        "drand_signature"
                    ],
                    "obtained_at": confirmation["confirmation"]["obtained_at"],
                    "seed_sha256": seed,
                },
            }
            observation = w11_record(
                {
                    name: manifest[name]
                    for name in (
                        "binary_sha256",
                        "configuration_sha256",
                        "model_sha256",
                        "tokenizer_sha256",
                        "fixture_sha256",
                    )
                }
            )
            summary = self.goal.score_registered_gate(
                "W11", scorer_id, [observation]
            )
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(json.dumps(observation) + "\n")
            (attempt / "summary.json").write_text(json.dumps(summary))
            published = {
                "round": confirmation["confirmation"]["round"],
                "randomness": confirmation["confirmation"]["drand_randomness"],
                "signature": confirmation["confirmation"]["drand_signature"],
            }
            with mock.patch.object(
                self.goal, "_fetch_public_drand", return_value=published
            ):
                with self.assertRaisesRegex(ValueError, "source provenance"):
                    self.goal.validate_attempt(attempt)

    def test_w11_raw_identities_must_match_manifest_artifacts(self):
        manifest = {
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "fixture_sha256": "e" * 64,
        }
        record = {
            "binary_sha256": "f" * 64,
            "configuration_sha256": "b" * 64,
            "model_sha256": "c" * 64,
            "tokenizer_sha256": "d" * 64,
            "fixture_sha256": "e" * 64,
        }
        with self.assertRaisesRegex(ValueError, "raw binary identity"):
            self.goal.validate_record_artifact_bindings(
                "W11", manifest, [record]
            )

    def test_foundation_and_parity_candidate_identities_match_manifest(self):
        manifest = {
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        foundation = {
            "glm_baseline": {
                "binary_sha256": "f" * 64,
                "configuration_sha256": "b" * 64,
                "fixture_sha256": "c" * 64,
            }
        }
        parity = {
            "profile": "glm52",
            "binary_sha256": "f" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        for gate, records in (
            ("foundation", [foundation]),
            ("parity", [parity]),
        ):
            with self.subTest(gate=gate):
                with self.assertRaisesRegex(ValueError, "raw binary identity"):
                    self.goal.validate_record_artifact_bindings(
                        gate, manifest, records
                    )

    def test_workstream_candidate_identities_match_manifest(self):
        record = workstream_record("W2")
        manifest = {
            name: record[name]
            for name in (
                "binary_sha256",
                "configuration_sha256",
                "fixture_sha256",
            )
        }
        self.goal.validate_record_artifact_bindings("W2", manifest, [record])
        broken = json.loads(json.dumps(record))
        broken["binary_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "raw binary identity"):
            self.goal.validate_record_artifact_bindings(
                "W2", manifest, [broken]
            )

    def test_foundation_and_parity_reject_unapproved_dsv4_reference(self):
        candidate = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
        manifest = {
            "candidate_hash": candidate,
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        glm = {
            "profile": "glm52",
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        dsv4 = {
            "profile": "dsv4",
            "binary_sha256": "d" * 64,
            "configuration_sha256": "e" * 64,
            "fixture_sha256": "c" * 64,
        }
        records_by_gate = {
            "foundation": [
                {"glm_baseline": glm, "dsv4_baseline": dsv4}
            ],
            "parity": [glm, dsv4],
        }
        for gate, records in records_by_gate.items():
            with self.subTest(gate=gate):
                with self.assertRaisesRegex(
                    ValueError, "approved DeepSeek profile"
                ):
                    self.goal.validate_record_artifact_bindings(
                        gate, manifest, records
                    )

    def test_foundation_and_parity_accept_frozen_dsv4_reference(self):
        manifest = {
            "candidate_hash": "f" * 40,
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        glm = {
            "profile": "glm52",
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "fixture_sha256": "c" * 64,
        }
        dsv4 = {
            "profile": "dsv4",
            "binary_sha256": "d" * 64,
            "configuration_sha256": "e" * 64,
            "fixture_sha256": "c" * 64,
        }
        records_by_gate = {
            "foundation": [
                {"glm_baseline": glm, "dsv4_baseline": dsv4}
            ],
            "parity": [glm, dsv4],
        }
        approved = {
            "schema_version": 1,
            "profile": "dsv4",
            "binary_sha256": "d" * 64,
            "configuration_sha256": "e" * 64,
        }
        with mock.patch.object(
            self.goal,
            "_load_approved_dsv4_profile",
            return_value=approved,
        ):
            for gate, records in records_by_gate.items():
                with self.subTest(gate=gate):
                    self.goal.validate_record_artifact_bindings(
                        gate, manifest, records
                    )

    def test_approved_dsv4_profile_parser_fails_closed(self):
        valid = {
            "schema_version": 2,
            "profile": "dsv4",
            "binary_sha256": "a" * 64,
            "configuration_sha256": "b" * 64,
            "build_manifest_sha256": "c" * 64,
            "weights_manifest_sha256": "d" * 64,
            "shared_libraries": {"libllama.so": "e" * 64},
            "model_files": {"base": "f" * 64},
        }
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(valid).encode(),
            stderr=b"",
        )
        with mock.patch.object(
            self.goal.subprocess, "run", return_value=completed
        ):
            self.assertEqual(
                self.goal._load_approved_dsv4_profile("f" * 40), valid
            )
        for mutation in (
            {**valid, "profile": "glm52"},
            {**valid, "binary_sha256": "bad"},
            {**valid, "shared_libraries": {}},
            {**valid, "model_files": {"base": "bad"}},
            {**valid, "extra": True},
        ):
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=json.dumps(mutation).encode(),
                stderr=b"",
            )
            with self.subTest(mutation=mutation):
                with mock.patch.object(
                    self.goal.subprocess, "run", return_value=completed
                ):
                    with self.assertRaisesRegex(
                        ValueError, "approved DeepSeek profile"
                    ):
                        self.goal._load_approved_dsv4_profile("f" * 40)

    def test_controller_git_subprocesses_are_absolute_and_isolated(self):
        source = SCRIPT.read_text()
        self.assertNotIn('["git",', source)
        self.assertIn('["/usr/bin/git",', source)
        self.assertIn('"GIT_CONFIG_NOSYSTEM": "1"', source)
        self.assertIn('"GIT_CONFIG_GLOBAL": "/dev/null"', source)

    def test_candidate_binary_model_and_profile_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in ("binary", "model", "configuration"):
                path = root / name
                path.write_bytes(f"arbitrary {name}".encode())
                paths[name] = path
            manifest = {
                "gate": "W11",
                "candidate_hash": subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    check=True,
                ).stdout.strip(),
                "binary_sha256": hashlib.sha256(
                    paths["binary"].read_bytes()
                ).hexdigest(),
                "model_sha256": hashlib.sha256(
                    paths["model"].read_bytes()
                ).hexdigest(),
                "configuration_sha256": hashlib.sha256(
                    paths["configuration"].read_bytes()
                ).hexdigest(),
            }
            with self.assertRaisesRegex(ValueError, "approved GLM profile"):
                self.goal.validate_profile_artifact_bindings(manifest, paths)

    def test_complete_glm_runtime_profile_is_accepted_and_hash_bound(self):
        artifact_paths = {
            "scripts/11_build_glm52_repro.sh",
            "results/glm52-goal/harness/decisive_matched.sh",
            "results/glm52-goal/harness/glm_decisive_arm.sh",
            "results/glm52-gates/harness/glm_safe_run.sh",
            "results/glm52-gates/harness/glm_cgroup_run.sh",
            "results/glm52-gates/harness/glm_evidence_export.py",
            "scripts/30_bench_speed.py",
        }
        artifact_digest = hashlib.sha256(b"frozen artifact").hexdigest()
        profile = {
            "schema_version": 2,
            "profile": "glm52",
            "binary_sha256": hashlib.sha256(b"binary").hexdigest(),
            "model_sha256": hashlib.sha256(b"model").hexdigest(),
            "tokenizer_sha256": "a" * 64,
            "context_cap": 1_048_576,
            "build_manifest_sha256": artifact_digest,
            "runtime": {
                "engine_environment": {
                    "DS4_CUDA_EXPERT_CACHE_GB": "0",
                    "DS4_CUDA_EXPERT_CACHE_PIN": "1",
                    "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
                    "DS4_CUDA_FETCH_THREADS": "6",
                    "DS4_CUDA_IQ2_DOWN_REFERENCE": "1",
                    "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
                    "DS4_TOKEN_TIMING_LOG": "1",
                },
                "launch_arguments": [
                    "--cuda", "-m", "{model}", "-c", "8192",
                    "--host", "127.0.0.1", "--port", "{port}",
                    "--ssd-streaming",
                    "--ssd-streaming-cache-experts", "40GB",
                ],
                "benchmark": {
                    "fixture_context_tokens": 0,
                    "max_completion_tokens": 160,
                    "minimum_completion_tokens": 128,
                    "raw_token_timing_required": True,
                },
                "safety": {
                    "kill_floor_gib": 40,
                    "minimum_start_gib": 110,
                    "sample_hz": 4,
                    "swap_max_bytes": 0,
                    "timeout_seconds": 2400,
                    "virtual_memory_limit_kib": 419_430_400,
                },
            },
            "artifact_sha256": {
                path: artifact_digest for path in artifact_paths
            },
        }
        profile_bytes = json.dumps(profile).encode()

        def git_show(args, **kwargs):
            requested = args[-1].split(":", 1)[1]
            data = (
                profile_bytes
                if requested == "configs/glm52-profile.json"
                else b"frozen artifact"
            )
            return subprocess.CompletedProcess(args, 0, data, b"")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name, data in (
                ("binary", b"binary"),
                ("model", b"model"),
                ("configuration", profile_bytes),
            ):
                paths[name] = root / name
                paths[name].write_bytes(data)
            manifest = {
                "gate": "W11",
                "candidate_hash": "f" * 40,
                "binary_sha256": profile["binary_sha256"],
                "model_sha256": profile["model_sha256"],
                "configuration_sha256": hashlib.sha256(profile_bytes).hexdigest(),
            }
            with mock.patch.object(
                self.goal.subprocess, "run", side_effect=git_show
            ):
                self.goal.validate_profile_artifact_bindings(manifest, paths)

    def test_w11_rejects_self_authored_fixture_with_valid_seed_label(self):
        record = w11_record()
        candidate = "a" * 40
        seed = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture = {
                "schema_version": 1,
                "candidate_hash": candidate,
                "seed_sha256": seed,
                "generator_version": "w11-fixture.v2",
                "context_cap": 1_048_576,
                "stage_context_caps": [1_048_576],
                "retrieval_cases": [
                    {
                        "case_id": item["case_id"],
                        "position": item["position"],
                        "expected_sha256": item["expected_sha256"],
                    }
                    for item in record["retrieval_results"]
                ],
                "negative_control_cases": [
                    {
                        "case_id": item["case_id"],
                        "expected_sha256": item["expected_sha256"],
                    }
                    for item in record["negative_control_results"]
                ],
            }
            fixture_path.write_text(json.dumps(fixture))
            manifest = {
                field: record[field]
                for field in (
                    "binary_sha256",
                    "configuration_sha256",
                    "model_sha256",
                    "tokenizer_sha256",
                )
            }
            manifest.update(
                {
                    "candidate_hash": candidate,
                    "fixture_sha256": hashlib.sha256(
                        fixture_path.read_bytes()
                    ).hexdigest(),
                    "lineage": {
                        "randomness": {"seed_sha256": seed}
                    },
                }
            )
            record["fixture_sha256"] = manifest["fixture_sha256"]
            with self.assertRaisesRegex(ValueError, "deterministic"):
                self.goal.validate_record_artifact_bindings(
                    "W11",
                    manifest,
                    [record],
                    {"fixture": fixture_path},
                )

    def test_w11_accepts_exact_deterministic_fixture(self):
        record = w11_record()
        candidate = "a" * 40
        seed = "b" * 64
        fixture = self.goal.generate_w11_fixture(candidate, seed)
        self.assertEqual(fixture["stage_context_caps"], [1_048_576])
        record["retrieval_results"] = [
            {**case, "observed_sha256": case["expected_sha256"]}
            for case in fixture["retrieval_cases"]
        ]
        record["negative_control_results"] = [
            {**case, "observed_sha256": case["expected_sha256"]}
            for case in fixture["negative_control_cases"]
        ]
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture_path.write_text(json.dumps(fixture))
            manifest = {
                field: record[field]
                for field in (
                    "binary_sha256",
                    "configuration_sha256",
                    "model_sha256",
                    "tokenizer_sha256",
                )
            }
            manifest.update(
                {
                    "candidate_hash": candidate,
                    "fixture_sha256": hashlib.sha256(
                        fixture_path.read_bytes()
                    ).hexdigest(),
                    "lineage": {
                        "randomness": {"seed_sha256": seed}
                    },
                }
            )
            record["fixture_sha256"] = manifest["fixture_sha256"]
            self.goal.validate_record_artifact_bindings(
                "W11",
                manifest,
                [record],
                {"fixture": fixture_path},
            )

    def test_w11_retrieval_expectations_are_bound_to_fixture(self):
        record = w11_record()
        candidate = "a" * 40
        seed = "b" * 64
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture = self.goal.generate_w11_fixture(candidate, seed)
            record["retrieval_results"] = [
                {
                    **case,
                    "observed_sha256": case["expected_sha256"],
                }
                for case in fixture["retrieval_cases"]
            ]
            record["negative_control_results"] = [
                {
                    **case,
                    "observed_sha256": case["expected_sha256"],
                }
                for case in fixture["negative_control_cases"]
            ]
            record["retrieval_results"][0]["expected_sha256"] = "f" * 64
            record["retrieval_results"][0]["observed_sha256"] = "f" * 64
            fixture_path.write_text(json.dumps(fixture))
            manifest = {
                field: record[field]
                for field in (
                    "binary_sha256",
                    "configuration_sha256",
                    "model_sha256",
                    "tokenizer_sha256",
                )
            }
            manifest["fixture_sha256"] = hashlib.sha256(
                fixture_path.read_bytes()
            ).hexdigest()
            manifest["candidate_hash"] = candidate
            manifest["lineage"] = {"randomness": {"seed_sha256": seed}}
            record["fixture_sha256"] = manifest["fixture_sha256"]
            with self.assertRaisesRegex(ValueError, "fixture retrieval"):
                self.goal.validate_record_artifact_bindings(
                    "W11",
                    manifest,
                    [record],
                    {"fixture": fixture_path},
                )

    def test_w11_fixture_is_bound_to_candidate_seed_and_generator(self):
        record = w11_record()
        with tempfile.TemporaryDirectory() as tmp:
            fixture_path = Path(tmp) / "fixture.json"
            fixture = {
                "schema_version": 1,
                "context_cap": 1_048_576,
                "stage_context_caps": [131_072, 262_144, 524_288, 1_048_576],
                "retrieval_cases": [
                    {
                        "case_id": item["case_id"],
                        "position": item["position"],
                        "expected_sha256": item["expected_sha256"],
                    }
                    for item in record["retrieval_results"]
                ],
                "negative_control_cases": [
                    {
                        "case_id": item["case_id"],
                        "expected_sha256": item["expected_sha256"],
                    }
                    for item in record["negative_control_results"]
                ],
            }
            fixture_path.write_text(json.dumps(fixture))
            manifest = {
                field: record[field]
                for field in (
                    "binary_sha256",
                    "configuration_sha256",
                    "model_sha256",
                    "tokenizer_sha256",
                )
            }
            manifest["fixture_sha256"] = hashlib.sha256(
                fixture_path.read_bytes()
            ).hexdigest()
            record["fixture_sha256"] = manifest["fixture_sha256"]
            with self.assertRaisesRegex(ValueError, "seed|generator|candidate"):
                self.goal.validate_record_artifact_bindings(
                    "W11",
                    manifest,
                    [record],
                    {"fixture": fixture_path},
                )

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                '{"formula_version":1,"verdict":"FAIL","verdict":"PASS"}'
            )
            with self.assertRaises(ValueError):
                self.goal._read_strict_json(path)


class ControllerTests(unittest.TestCase):
    def run_cli(self, state_dir: Path, *args: str):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--state-dir", str(state_dir), *args],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_status_initializes_all_required_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_cli(Path(tmp), "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            expected = {"foundation", "switch", "parity", "review"} | {
                f"W{i}" for i in range(1, 12)
            }
            self.assertEqual(set(state["gates"]), expected)
            self.assertTrue(
                all(
                    gate["status"] == "PENDING"
                    for name, gate in state["gates"].items()
                    if name != "W1"
                )
            )
            # W1 is deliberately global and root-authoritative, even when a
            # disposable controller state directory is used. Its initial
            # status therefore reflects preserved machine evidence.
            self.assertIn(
                state["gates"]["W1"]["status"],
                {"PENDING", "RED_CONFIRMED", "PASS", "FAIL", "NO_RESULT"},
            )

    def test_resume_selects_highest_value_unfinished_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            first = self.run_cli(state_dir, "resume")
            self.assertEqual(first.returncode, 0, first.stderr)
            event = json.loads(first.stdout)
            self.assertEqual(event["selected_gate"], "foundation")

    def test_w1_affine_diagnostic_pass_is_only_red_confirmed(self):
        goal = load_goal_module()
        status, reason = goal._gate_status_from_summary(
            "W1",
            {
                "scorer_id": "w1.affine-quality.v2",
                "verdict": "PASS",
            },
        )
        self.assertEqual(status, "RED_CONFIRMED")
        self.assertIn("real packed storage", reason)
        self.assertIn("retrieval", reason)

    def test_w1_real_packed_fidelity_pass_still_requires_retrieval(self):
        goal = load_goal_module()
        status, reason = goal._gate_status_from_summary(
            "W1",
            {
                "scorer_id": "w1.affine-quality.v2",
                "verdict": "PASS",
            },
            candidate_format="affine-int8-block16",
        )
        self.assertEqual(status, "RED_CONFIRMED")
        self.assertIn("storage, memory, and fidelity passed", reason)
        self.assertIn("retrieval", reason)

    def test_w1_retrieval_dependency_routes_to_w8_then_w11(self):
        goal = load_goal_module()
        state = goal._initial_state()
        state["gates"]["foundation"]["status"] = "PASS"
        state["gates"]["W1"].update(
            status="RED_CONFIRMED",
            reason=(
                "real packed storage, memory, and fidelity passed; "
                "deterministic retrieval qualification remains unfinished"
            ),
        )

        self.assertEqual(goal._selected_gate(state), "W8")
        state["gates"]["W8"]["status"] = "PASS"
        self.assertEqual(goal._selected_gate(state), "W11")

        state["gates"]["W11"]["status"] = "PASS"
        self.assertEqual(goal._selected_gate(state), "W1")
        self.assertIn("W1", goal._release_verdict(Path("/tmp"), state)["failed_requirements"])

    def test_unregistered_default_runner_is_never_executed(self):
        goal = load_goal_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "scripts" / "glm52-runners" / "W1"
            runner.parent.mkdir(parents=True)
            runner.write_text("#!/bin/sh\nexit 99\n")
            runner.chmod(0o700)
            self.assertIsNone(
                goal._registered_default_runner(
                    "W1", root=root, registry={}
                )
            )

    def test_registered_runner_executes_verified_bytes_after_path_swap(self):
        goal = load_goal_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "scripts" / "glm52-runners" / "W8"
            runner.parent.mkdir(parents=True)
            approved = b"#!/bin/sh\nprintf 'approved\\n'\n"
            runner.write_bytes(approved)
            runner.chmod(0o700)
            descriptor = goal._registered_default_runner(
                "W8",
                root=root,
                registry={"W8": hashlib.sha256(approved).hexdigest()},
            )
            self.assertIsInstance(descriptor, int)
            runner.write_text("#!/bin/sh\nprintf 'replacement\\n'\n")
            try:
                completed = subprocess.run(
                    [f"/proc/self/fd/{descriptor}"],
                    pass_fds=(descriptor,),
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            finally:
                os.close(descriptor)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "approved\n")

    def test_registered_runner_copy_handles_short_memfd_writes(self):
        goal = load_goal_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "scripts" / "glm52-runners" / "W8"
            runner.parent.mkdir(parents=True)
            approved = b"#!/bin/sh\nprintf 'complete-approved-runner\\n'\n"
            runner.write_bytes(approved)
            runner.chmod(0o700)
            real_write = os.write

            def short_write(descriptor, value):
                return real_write(descriptor, value[: max(1, len(value) // 2)])

            with mock.patch.object(goal.os, "write", side_effect=short_write):
                descriptor = goal._registered_default_runner(
                    "W8",
                    root=root,
                    registry={"W8": hashlib.sha256(approved).hexdigest()},
                )
            self.assertIsInstance(descriptor, int)
            try:
                self.assertEqual(os.pread(descriptor, len(approved) + 1, 0), approved)
            finally:
                os.close(descriptor)

    def test_state_rejects_unknown_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            state["gates"]["W1"]["status"] = "CLAIMED_PASS"
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "status", "--json")
            self.assertNotEqual(result.returncode, 0)

    def test_state_rejects_terminal_status_without_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            for gate in state["gates"].values():
                gate["status"] = "PASS"
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "resume")
            self.assertNotEqual(result.returncode, 0)

    def test_state_rejects_terminal_status_when_attempt_disappears(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state_path = state_dir / "state.json"
            state = json.loads(state_path.read_text())
            state["gates"]["foundation"] = {
                "status": "PASS",
                "attempts": ["foundation/ghost-attempt"],
                "reason": None,
            }
            state_path.write_text(json.dumps(state))
            result = self.run_cli(state_dir, "status", "--json")
            self.assertNotEqual(result.returncode, 0)

    def test_attempt_manifest_gate_must_match_directory_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "W2" / "attempt-001"
            attempt.mkdir(parents=True)
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = attempt / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text('{"event":"diagnostic"}\n')
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"NO_RESULT","reason":"fake"}'
            )
            result = self.run_cli(state_dir, "status", "--json")
            self.assertEqual(result.returncode, 0, result.stderr)
            state = json.loads(result.stdout)
            self.assertEqual(state["gates"]["W2"]["status"], "FAIL")
            self.assertIn("does not match", state["gates"]["W2"]["reason"])

    def test_release_check_fails_closed_on_incomplete_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            result = self.run_cli(state_dir, "release-check", "--json")
            self.assertNotEqual(result.returncode, 0)
            value = json.loads(result.stdout)
            self.assertFalse(value["release_qualified"])
            self.assertIn("W11", value["failed_requirements"])

    def test_release_accepts_only_registered_recomputed_no_go(self):
        goal = load_goal_module()
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "parity" / "attempt-002"
            attempt.mkdir(parents=True)
            summary = {
                "scorer_id": "parity.reviewed-no-go.v1",
                "formula_version": 1,
                "decision": "NO_GO",
                "verdict": "PASS",
            }
            (attempt / "summary.json").write_text(json.dumps(summary))
            state = goal._initial_state()
            for name, gate in state["gates"].items():
                gate["attempts"] = [f"{name}/attempt-001"]
                if name in {"foundation", "W11", "switch", "parity", "review"}:
                    gate["status"] = "PASS"
                else:
                    gate["status"] = "FAIL"
            state["gates"]["parity"]["attempts"] = ["parity/attempt-002"]
            with mock.patch.object(goal, "validate_attempt"):
                result = goal._release_verdict(state_dir, state)
            self.assertTrue(result["release_qualified"])
            self.assertEqual(result["parity_decision"], "NO_GO")

            summary["scorer_id"] = "parity.performance.v1"
            (attempt / "summary.json").write_text(json.dumps(summary))
            with mock.patch.object(goal, "validate_attempt"):
                rejected = goal._release_verdict(state_dir, state)
            self.assertFalse(rejected["release_qualified"])
            self.assertEqual(rejected["parity_decision"], "UNPROVEN")

    def test_resume_ingests_valid_attempt_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "foundation" / "attempt-001"
            attempt.mkdir(parents=True)
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = attempt / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (attempt / "manifest.json").write_text(json.dumps(manifest))
            (attempt / "raw.jsonl").write_text(
                json.dumps({"event": "bounded falsifier", "failures": []}) + "\n"
            )
            (attempt / "summary.json").write_text(
                json.dumps(
                    {
                        "formula_version": 1,
                        "verdict": "NO_RESULT",
                        "reason": "bounded attempt exhausted",
                    }
                )
            )
            result = self.run_cli(state_dir, "resume")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "foundation")
            state = json.loads((state_dir / "state.json").read_text())
            self.assertEqual(state["gates"]["foundation"]["status"], "FAIL")
            self.assertEqual(
                state["gates"]["foundation"]["attempts"], ["foundation/attempt-001"]
            )

    def test_malformed_attempt_fails_gate_and_does_not_get_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            attempt = state_dir / "foundation" / "attempt-001"
            attempt.mkdir(parents=True)
            (attempt / "manifest.json").write_text("{}")
            (attempt / "raw.jsonl").write_text("{}\n")
            (attempt / "summary.json").write_text(
                '{"formula_version":1,"verdict":"PASS"}'
            )
            result = self.run_cli(state_dir, "resume")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "foundation")
            state = json.loads((state_dir / "state.json").read_text())
            self.assertEqual(state["gates"]["foundation"]["status"], "FAIL")
            self.assertIn("invalid", state["gates"]["foundation"]["reason"])

    def test_run_executes_registered_runner_and_advances(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            seed = state_dir / "seed-attempt"
            seed.mkdir()
            manifest = {
                "gate": "foundation",
                "candidate_hash": "a" * 40,
                "artifacts": {},
            }
            for index, name in enumerate(
                (
                    "source",
                    "diff",
                    "binary",
                    "scorer",
                    "model",
                    "tokenizer",
                    "fixture",
                    "configuration",
                )
            ):
                path = seed / f"{name}.artifact"
                path.write_text(f"artifact-{index}")
                manifest["artifacts"][name] = path.name
                manifest[f"{name}_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
            (seed / "manifest.json").write_text(json.dumps(manifest))
            (seed / "raw.jsonl").write_text(
                '{"event":"bounded_runner","failures":[]}\n'
            )
            (seed / "summary.json").write_text(
                '{"formula_version":1,"verdict":"NO_RESULT","reason":"bounded"}'
            )
            runners = state_dir / "runners"
            runners.mkdir()
            runner = runners / "foundation"
            runner.write_text(
                "#!/bin/sh\n"
                "set -eu\n"
                "mkdir -p \"$1/foundation\"\n"
                "cp -R \"$1/seed-attempt\" \"$1/foundation/attempt-001\"\n"
                "echo ran > \"$1/runner.marker\"\n"
            )
            runner.chmod(0o700)
            result = self.run_cli(state_dir, "run")
            self.assertEqual(result.returncode, 0, result.stderr)
            event = json.loads(result.stdout)
            self.assertEqual(event["selected_gate"], "foundation")
            self.assertEqual(
                event["action"], "runner_produced_no_terminal_evidence"
            )
            self.assertEqual((state_dir / "runner.marker").read_text(), "ran\n")


if __name__ == "__main__":
    unittest.main()
