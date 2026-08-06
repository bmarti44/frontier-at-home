#!/usr/bin/env python3
"""No-model integration mutations for candidate provenance shutdown."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SAFE = ROOT / "results/glm52-gates/harness/glm_safe_run.sh"
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
W3_PROBE_V3 = ROOT / "results/glm52-gates/harness/w3_direct_slot_probe_v3.sh"
W3_TOKEN_SCORER = ROOT / "scripts/84_count_glm_output_tokens.py"
FIXTURE = ROOT / "scripts/tests/fixtures/candidate_lifecycle.c"


class CandidateLifecycleSourceTests(unittest.TestCase):
    def test_w3_probe_closes_reviewed_false_pass_routes(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        self.assertIn('-x $CGROUP && -r $SAFE', probe)
        self.assertNotIn('-x $CGROUP && -x $SAFE', probe)
        for contract in (
            'OBSERVED_MODEL_SHA256=$(sha256sum -- "$MODEL"',
            "/usr/bin/env -i",
            'direct expert-slot dispatch layer=',
            'measured_dispatch_delta',
            'warm_generated_output_byte_identical',
            'freeze manifest differs from HEAD',
            'unique W3 evidence tag was already consumed',
            'drand relay disagreement',
            'wait "$runner_pid"',
            'changed != [str(relative)]',
            'independent_exact_output_tokens',
            '/usr/bin/python3 -I',
            '84_count_glm_output_tokens.py',
        ):
            self.assertIn(contract, probe)

    def test_w3_probe_isolates_every_python_authority(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        self.assertIn("isolated_python()", probe)
        self.assertIn("/usr/bin/env -i", probe)
        self.assertIn("/usr/bin/python3 -I -B", probe)
        self.assertNotRegex(probe, r"(?m)^\s*python3(?:\s|$)")
        self.assertNotRegex(probe, r"(?m)^\s*/usr/bin/python3(?:\s|$)")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "ambient-python-loaded"
            (root / "sitecustomize.py").write_text(
                f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(root)
            result = subprocess.run([
                "/usr/bin/env", "-i", "HOME=/nonexistent", "PATH=/usr/bin:/bin",
                "LANG=C.UTF-8", "LC_ALL=C.UTF-8", "/usr/bin/python3", "-I", "-B",
                "-c", "print('isolated')",
            ], env=environment, text=True, stdout=subprocess.PIPE,
               stderr=subprocess.PIPE, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "isolated")
            self.assertFalse(marker.exists())

    def test_w3_engine_environment_hash_uses_launch_assignments(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        self.assertIn('local -a engine_environment=(', probe)
        self.assertIn(
            'env_sha=$(environment_sha256 "${engine_environment[@]}")', probe
        )
        self.assertIn('"${engine_environment[@]}" \\', probe)
        self.assertIn('"$CGROUP" --tag', probe)
        self.assertNotIn("export DS4_", probe)

        names = [
            "DS4_CUDA_EXPERT_CACHE_GB",
            "DS4_CUDA_EXPERT_CACHE_PIN",
            "DS4_CUDA_EXPERT_CACHE_SLRU",
            "DS4_CUDA_FETCH_THREADS",
            "DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS",
            "DS4_CUDA_MOE_NO_ATOMIC_DOWN",
            "DS4_GLM_TP_DEBUG",
            "DS4_LOCK_EXPECTED_DEV_INO",
            "DS4_LOCK_FILE",
            "DS4_TOKEN_TIMING_LOG",
        ]
        off = {
            "DS4_CUDA_EXPERT_CACHE_GB": "68",
            "DS4_CUDA_EXPERT_CACHE_PIN": "1",
            "DS4_CUDA_EXPERT_CACHE_SLRU": "1",
            "DS4_CUDA_FETCH_THREADS": "6",
            "DS4_CUDA_MOE_NO_ATOMIC_DOWN": "1",
            "DS4_GLM_TP_DEBUG": "1",
            "DS4_LOCK_EXPECTED_DEV_INO": "31:12345",
            "DS4_LOCK_FILE": "/run/user/1000/ds4-engine.lock",
            "DS4_TOKEN_TIMING_LOG": "1",
        }
        on = dict(off, DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS="1")

        def digest(values):
            canonical = b"".join(
                f"{name}={values.get(name, '<UNSET>')}\n".encode("ascii")
                for name in names
            )
            return hashlib.sha256(canonical).hexdigest()

        self.assertNotEqual(digest(off), digest(on))
        mutated = dict(on)
        del mutated["DS4_CUDA_FETCH_THREADS"]
        self.assertNotEqual(digest(on), digest(mutated))

    def test_w3_run_arm_derives_locals_after_positional_assignment(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        self.assertIn(
            "local arm=$1 port=$2 direct=$3\n"
            "  local tag=\"w3s${DRAND_ROUND}-${arm}\" arm_dir=\"$OUT/$arm\"",
            probe,
        )
        self.assertNotIn(
            "local arm=$1 port=$2 direct=$3 tag=\"w3s${DRAND_ROUND}-${arm}\"",
            probe,
        )

    def test_w3_uses_validated_private_engine_lock_and_runner_identity(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        for contract in (
            "readonly ENGINE_LOCK=/run/user/1000/ds4-engine.lock",
            "readonly OUTER_LOCK=/run/lock/frontier-at-home/inference.lock",
            '[[ $ENGINE_LOCK != "$OUTER_LOCK" ]]',
            'DS4_LOCK_FILE=$ENGINE_LOCK',
            ',DS4_LOCK_FILE',
            "validate_engine_lock",
            'flock -n -E 75 -- "$ENGINE_LOCK"',
            "'%U:%G:%a:%h'",
            "DS4_LOCK_EXPECTED_DEV_INO",
            "revalidate_engine_lock",
            "runner_start_ticks",
            "runner_is_exact",
            '[[ $runner_state != Z && $runner_state != X ]]',
            'runner exited before the frozen W3 engine appeared',
        ):
            self.assertIn(contract, probe)
        self.assertIn("runner_is_exact || return 1", probe)

        shell = r'''
set -u
runner_pid=
runner_start_ticks=
runner_is_exact() {
  local runner_state observed_start_ticks
  [[ -n ${runner_pid:-} && -n ${runner_start_ticks:-} &&
     -r /proc/$runner_pid/stat ]] || return 1
  read -r runner_state observed_start_ticks < <(
    awk '{print $3, $22}' "/proc/$runner_pid/stat" 2>/dev/null
  ) || return 1
  [[ $runner_state != Z && $runner_state != X ]] || return 1
  [[ $observed_start_ticks == "$runner_start_ticks" ]]
}
zombie_file=$(mktemp)
/usr/bin/python3 - "$zombie_file" <<'PY' &
import os
from pathlib import Path
import sys
import time
pid = os.fork()
if pid == 0:
    os._exit(7)
Path(sys.argv[1]).write_text(str(pid))
time.sleep(10)
PY
zombie_parent=$!
trap 'kill "$zombie_parent" 2>/dev/null || true; rm -f "$zombie_file"' EXIT
for _ in $(seq 1 100); do
  [[ -s $zombie_file ]] && break
  sleep 0.01
done
runner_pid=$(<"$zombie_file")
runner_start_ticks=$(awk '{print $22}' "/proc/$runner_pid/stat")
runner_is_exact && exit 99
kill "$zombie_parent" 2>/dev/null || true
wait "$zombie_parent" 2>/dev/null || true
rm -f "$zombie_file"
trap - EXIT
exit 0
'''
        result = subprocess.run(
            ["/usr/bin/bash", "-c", shell], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("sort -n | tail -1", probe)
        self.assertNotIn('a["completion_tokens"] >= 64', probe)
        self.assertIn('a["independent_completion_tokens"] == required_tokens', probe)
        self.assertIn('a["independent_warm_completion_tokens"] == required_tokens', probe)
        scorer = W3_TOKEN_SCORER.read_text(encoding="utf-8")
        for contract in (
            "sys.flags.isolated != 1",
            "os.environ != ALLOWED_ENVIRONMENT",
            "native tokenizer did not load through its bound descriptor",
            "native tokenizer changed during import",
            "tokenizer dependency changed during scoring",
            "tokenizer.encode(content, add_special_tokens=False).ids",
            "native.Tokenizer.from_str",
        ):
            self.assertIn(contract, scorer)
        for contract in (
            'token_record["response_sha256"]',
            'token_record["label"]',
            'token_record["runtime_native_sha256"]',
            'token_record["runtime_native_loaded_path"]',
        ):
            self.assertIn(contract, probe)

    def test_w3_supports_bound_129_token_balanced_campaign_arms(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        for contract in (
            'readonly TOKENS=${3:-64}',
            'readonly ARM_ORDER=${4:-off-on}',
            '[[ $TOKENS == 64 || $TOKENS == 129 ]]',
            '[[ $ARM_ORDER == off-on || $ARM_ORDER == on-off ]]',
            'DS4_TOKEN_TIMING_LOG=1',
            'readonly CAMPAIGN_SCORER=$REPO/scripts/85_score_w3_performance_campaign.py',
            '"campaign_scorer": Path(campaign_scorer_raw).resolve()',
            '"block_schedule": ["ABBA", "BAAB", "ABBA", "BAAB", "ABBA"]',
            '"timing_receipt": timing_receipt',
            '"cmd_log_identity"',
            '"token_timestamps_ns"',
            '"environment_receipt"',
            '"DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS": "1" if direct == "1" else "<UNSET>"',
            '"arm_order": arm_order',
            '"required_completion_tokens": required_tokens',
            'if [[ $ARM_ORDER == off-on ]]',
            'run_checked_arm on 18164 1',
            'run_checked_arm off 18163 0',
        ):
            self.assertIn(contract, probe)

    def test_w3_memory_envelope_fits_post_arm_recovery(self):
        probe = W3_PROBE_V3.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_MEMORY_HIGH_GIB=94", probe)
        self.assertIn("GLM_SAFE_KILL_FLOOR_GIB=18", probe)
        self.assertIn("GLM_SAFE_MIN_START_GIB=110", probe)

    def test_evidence_timeout_ceiling_is_consistent_across_containment(self):
        safe_environment = {
            "PATH": "/usr/bin:/bin",
            "GLM_SAFE_TIMEOUT_S": "9000",
            "GLM_SAFE_MIN_START_GIB": "109",
        }
        safe = subprocess.run(
            ["env", *[f"{key}={value}" for key, value in safe_environment.items()],
             "bash", str(SAFE), "--tag", "timeout-contract", "--", "/bin/true"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(safe.returncode, 2)
        self.assertIn("invalid GLM_SAFE_MIN_START_GIB", safe.stderr)
        self.assertNotIn("invalid GLM_SAFE_TIMEOUT_S", safe.stderr)

        launcher_environment = {
            "PATH": "/usr/bin:/bin",
            "GLM_SAFE_TIMEOUT_S": "9000",
            "GLM_SAFE_RUN_AS_CURRENT_USER": "2",
        }
        launcher = subprocess.run(
            ["env", *[f"{key}={value}" for key, value in launcher_environment.items()],
             "bash", str(CGROUP), "--tag", "timeout-contract", "--", "/bin/true"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(launcher.returncode, 2)
        self.assertIn("invalid GLM_SAFE_RUN_AS_CURRENT_USER", launcher.stderr)

        for script in (SAFE, CGROUP):
            source = script.read_text(encoding="utf-8")
            self.assertIn("TIMEOUT_S > 9000", source)

    def test_benchmark_lock_acl_has_a_narrow_installer(self):
        references = []
        for script in (ROOT / "scripts").glob("*.sh"):
            text = script.read_text(encoding="utf-8")
            if "frontier-at-home-glm-benchmark.conf" in text:
                references.append((script, text))
        self.assertEqual(len(references), 1)
        _, installer = references[0]
        self.assertIn("systemd-tmpfiles --create", installer)
        self.assertIn("/etc/tmpfiles.d/frontier-at-home-glm-benchmark.conf", installer)
        expected = re.search(r"SOURCE_SHA256=([0-9a-f]{64})", installer)
        self.assertIsNotNone(expected)
        self.assertEqual(
            expected.group(1),
            hashlib.sha256(
                (ROOT / "configs/tmpfiles/frontier-at-home-glm-benchmark.conf").read_bytes()
            ).hexdigest(),
        )
        self.assertIn('sha256sum -- "$TARGET"', installer)
        self.assertLess(
            installer.index('sha256sum -- "$TARGET"'),
            installer.index("systemd-tmpfiles --create"),
        )
        self.assertNotIn("sudoers", installer)
        self.assertNotIn("systemctl", installer)

    def test_disappearing_proc_counters_are_emitted_as_numeric_zero(self):
        source = SAFE.read_text(encoding="utf-8")
        start = source.index("  RSS=$(awk ")
        end = source.index('  echo "$(date -u --iso-8601=ns)', start)
        sampler = source[start:end]
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "status").write_text("", encoding="utf-8")
            (proc / "io").write_text("", encoding="utf-8")
            sampler = sampler.replace(
                '"/proc/$SPID2/status"', f'"{proc / "status"}"'
            ).replace(
                '"/proc/$SPID2/io"', f'"{proc / "io"}"'
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    "set -u\nSPID2=missing\nREQUIRE_CGROUP=0\n"
                    + sampler
                    + 'printf "%s|%s\\n" "$RSS" "$RB"\n',
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "0|0\n")

    def test_exit_state_is_distinct_from_live_identity_failure(self):
        source = SAFE.read_text(encoding="utf-8")
        shutdown = source.index("elif [[ -z $CURRENT_STATE")
        identity = source.index(
            'plog "FATAL executed candidate identity changed pid=$EXECUTED_PID',
            shutdown,
        )
        guarded = source[shutdown:identity]
        self.assertIn("EXECUTED_CANDIDATE_EXIT_PENDING=1", guarded)
        self.assertNotIn("-z $CURRENT_HASH", guarded)
        self.assertNotIn("-z $CURRENT_DEVICE_INODE", guarded)

    def test_missing_start_ticks_make_identity_sample_incomplete(self):
        source = SAFE.read_text(encoding="utf-8")
        start = source.index("      IDENTITY_INCOMPLETE=0")
        end = source.index(
            "      if [[ -n $CURRENT_START_TICKS", start
        )
        self.assertIn("-z $CURRENT_START_TICKS", source[start:end])

    def test_verified_candidate_environment_is_hash_bound_from_proc(self):
        source = SAFE.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_PROVENANCE_ENV_ALLOWLIST", source)
        self.assertIn("GLM_SAFE_EXPECTED_ENV_SHA256", source)
        self.assertIn('"/proc/$SPID2/environ"', source)
        self.assertIn("executed_environment_sha256=", source)
        self.assertIn("executed candidate environment mismatch", source)
        self.assertIn("DS4_GLM_CKV_RUN_NONCE", source)
        self.assertIn("run_nonce=", source)

    def test_default_off_current_user_mode_retains_containment_and_provenance(self):
        safe = SAFE.read_text(encoding="utf-8")
        launcher = CGROUP.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER", safe)
        self.assertIn("GLM_SAFE_RUN_AS_CURRENT_USER", launcher)
        self.assertIn("/home/bmarti44/.local/state/glm52-crashlog", safe)
        self.assertIn("/home/bmarti44/.cache/glm52-", safe)
        self.assertIn("MemorySwapMax=0", launcher)
        self.assertIn("OOMPolicy=kill", launcher)
        self.assertIn("GLM_SAFE_PROVENANCE_ENV_ALLOWLIST", launcher)
        self.assertIn("GLM_SAFE_EXPECTED_ENV_SHA256", launcher)
        self.assertIn(
            "/run/lock/frontier-at-home/inference.lock", launcher
        )
        self.assertNotIn("/run/user/$UID/glm52-inference.lock", launcher)
        self.assertIn("/usr/bin/sudo -n -u dsv4", launcher)
        self.assertIn('RUN_CWD=$(pwd -P)', launcher)
        self.assertIn('--working-directory="$RUN_CWD"', launcher)
        self.assertIn("GLM_SAFE_WITNESS_NONCE", safe)
        self.assertIn("glm52-w1-witness", safe)
        self.assertIn("GLM_SAFE_WITNESS_NONCE", launcher)
        self.assertIn("GLM_SAFE_WITNESS_ARTIFACT", safe)
        self.assertIn("artifact_sha256=", safe)
        self.assertIn("GLM_SAFE_WITNESS_ARTIFACT", launcher)
        self.assertIn("date -u --iso-8601=ns", safe)
        for name in (
            "DS4_GLM_CKV_NVME",
            "DS4_GLM_CKV_DIR",
            "DS4_GLM_CKV_MODEL_SHA256",
            "DS4_GLM_CKV_RUN_NONCE",
            "DS4_GLM_CKV_MAX_GIB",
            "DS4_GLM_CKV_TRACE_PATH",
            "DS4_GLM_CKV_TRACE_SAMPLE_POSITIONS",
            "DS4_GLM_CKV_TRACE_MAX_RECORDS",
        ):
            self.assertIn(name, launcher)

    def test_cgroup_launcher_forwards_expert_slab_arm_identity(self):
        launcher = CGROUP.read_text(encoding="utf-8")
        for name in (
            "DS4_CUDA_EXPERT_SLAB_PATH",
            "DS4_CUDA_EXPERT_SLAB_SHA256",
            "DS4_CUDA_EXPERT_SLAB_MODEL_SHA256",
            "DS4_CUDA_EXPERT_SLAB_TRACE",
            "DS4_CUDA_EXPERT_SLAB_AUTH_TRACE",
            "DS4_CUDA_EXPERT_SLAB_PREFETCH_SHA",
            "DS4_CUDA_LOAD_PROFILE",
            "DS4_TOKEN_TIMING_LOG",
            "DS4_GLM_TP_DEBUG",
            "DS4_GLM_PREFETCH",
            "DS4_GLM_PREFETCH_THREADS",
            "DS4_CUDA_MOE_DIRECT_EXPERT_SLOTS",
            "DS4_LOCK_EXPECTED_DEV_INO",
            "DS4_LOCK_FILE",
        ):
            self.assertIn(name, launcher)

    def test_cgroup_launcher_accepts_only_a_measured_profile_envelope(self):
        launcher = CGROUP.read_text(encoding="utf-8")
        self.assertIn("GLM_SAFE_MEMORY_HIGH_GIB", launcher)
        self.assertIn("profile memory envelope exceeds safe host budget", launcher)
        self.assertIn("max_mib=$((high_mib + 2048))", launcher)
        self.assertIn(
            "max_mib + KILL_FLOOR_GIB * 1024 <= available_mib", launcher
        )

    def test_memory_guard_is_resolved_from_frozen_harness(self):
        safe = SAFE.read_text(encoding="utf-8")
        self.assertNotIn(
            "MEMORY_GUARD=/home/bmarti44/spark-deepseek-v4-flash", safe
        )
        self.assertIn('HARNESS_ROOT=$(dirname -- "$(dirname -- "$(dirname -- "$(dirname -- "$(readlink -f -- "$0")")")")")', safe)
        self.assertIn('MEMORY_GUARD=$HARNESS_ROOT/scripts/03_memory_guard.py', safe)

    def test_authoritative_timestamps_are_emitted_in_utc(self):
        source = SAFE.read_text(encoding="utf-8")
        self.assertIn(
            'plog() { echo "$(date -u --iso-8601=ns) $*" >> "$MAIN"',
            source,
        )
        self.assertIn(
            'echo "$(date -u --iso-8601=ns) mem_avail_kb=$MA',
            source,
        )
        self.assertNotIn("date -u -Is", source)
        self.assertNotIn('echo "$(date -Is)', source)
        self.assertNotIn('echo "$(date --iso-8601=ns)', source)

    def test_cgroup_and_xid_safety_telemetry_is_preserved(self):
        source = SAFE.read_text(encoding="utf-8")
        for control in (
            "memory.current",
            "memory.peak",
            "memory.swap.current",
            "memory.events.local",
        ):
            self.assertIn(control, source)
        self.assertIn("cgroup_current_bytes=", source)
        self.assertIn("cgroup_peak_bytes=", source)
        self.assertIn("cgroup_swap_current_bytes=", source)
        self.assertIn('KERNEL_LOG="$DIR/kernel.log"', source)
        for marker in ("NVRM.*Xid", "NV_ERR_NO_MEMORY", "oom-kill"):
            self.assertIn(marker, source)
        self.assertIn("FATAL kernel GPU/OOM evidence appeared during run", source)


class CandidateLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is unavailable")
        probe = subprocess.run(
            ["sudo", "-n", "-u", "dsv4", "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode:
            raise unittest.SkipTest("passwordless dsv4 test account is unavailable")
        available_gib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        ) / 1048576
        if available_gib < 110:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

        cls.local_tmp = Path(tempfile.mkdtemp(prefix="glm-lifecycle-"))
        os.chmod(cls.local_tmp, 0o755)
        cls.runner = cls.local_tmp / "runner"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-o", cls.runner, FIXTURE],
            check=True,
        )
        os.chmod(cls.runner, 0o755)
        created = subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "mktemp", "-d",
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle.XXXXXX",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        cls.candidate_src = Path(created.stdout.strip())
        subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "cp", "--",
                "/usr/bin/sleep", cls.candidate_src / "ds4-server",
            ],
            check=True,
        )
        cls.digest = hashlib.sha256(
            Path("/usr/bin/sleep").read_bytes()
        ).hexdigest()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "candidate_src"):
            target = str(cls.candidate_src)
            if target.startswith(
                "/home/dsv4/ds4-project/src/glm-safe-lifecycle."
            ):
                subprocess.run(
                    ["sudo", "-n", "-u", "dsv4", "rm", "-rf", "--", target],
                    check=False,
                )
        if hasattr(cls, "local_tmp"):
            shutil.rmtree(cls.local_tmp)

    def run_mutation(self, mode: str) -> subprocess.CompletedProcess[str]:
        environment = {
            "HOME": "/home/dsv4",
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "GLM_CANDIDATE_SRC": str(self.candidate_src),
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "30",
        }
        return subprocess.run(
            [
                "sudo", "-n", "-u", "dsv4", "env",
                *[f"{key}={value}" for key, value in environment.items()],
                "bash", str(SAFE), "--tag", f"lifecycle-{mode}", "--",
                str(self.runner), mode, str(self.candidate_src / "ds4-server"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )

    def test_clean_candidate_zombie_during_wrapper_exit_is_accepted(self):
        result = self.run_mutation("clean")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_clean_reaped_candidate_between_sampler_ticks_is_attested(self):
        result = self.run_mutation("reaped")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate was verified alive at least once; no identity "
            "contradiction observed by the periodic sampler; actual cadence is "
            "recorded in samples.log; wrapper and descendant checks clean",
            main,
        )

    def test_replacement_candidate_during_exit_is_rejected(self):
        result = self.run_mutation("replace")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonzero_wrapper_after_candidate_exit_is_rejected(self):
        result = self.run_mutation("fail")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_controller_postprocessing_after_candidate_exit_is_accepted(self):
        result = self.run_mutation("linger")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class CurrentUserTimestampTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("gcc") is None:
            raise unittest.SkipTest("gcc is unavailable")
        available_gib = int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            )
        ) / 1048576
        if available_gib < 110:
            raise unittest.SkipTest("110 GiB safe-run precondition is unavailable")

        cls.local_tmp = Path(tempfile.mkdtemp(prefix="glm-utc-lifecycle-"))
        cls.runner = cls.local_tmp / "runner"
        subprocess.run(
            ["gcc", "-O2", "-Wall", "-Wextra", "-o", cls.runner, FIXTURE],
            check=True,
        )
        cache = Path("/home/bmarti44/.cache")
        cache.mkdir(parents=True, exist_ok=True)
        cls.candidate_src = Path(
            tempfile.mkdtemp(prefix="glm52-utc-lifecycle-", dir=cache)
        )
        shutil.copy2("/usr/bin/sleep", cls.candidate_src / "ds4-server")
        cls.digest = hashlib.sha256(
            (cls.candidate_src / "ds4-server").read_bytes()
        ).hexdigest()
        cls.shim_dir = cls.local_tmp / "shim"
        cls.shim_dir.mkdir()
        cls.readlink_count = cls.local_tmp / "readlink-count"
        cls.readlink_count.write_text("0\n", encoding="ascii")
        os.chmod(cls.readlink_count, 0o666)
        readlink_shim = cls.shim_dir / "readlink"
        readlink_shim.write_text(
            "#!/bin/bash\n"
            "target=${!#}\n"
            "if [[ $target =~ ^/proc/([0-9]+)/exe$ ]]; then\n"
            "  count=$(<\"$GLM_TEST_READLINK_COUNT\")\n"
            "  count=$((count + 1))\n"
            "  printf '%s\\n' \"$count\" >\"$GLM_TEST_READLINK_COUNT\"\n"
            "  if (( count == 2 )); then\n"
            "    if [[ $GLM_TEST_READLINK_ACTION == exit ]]; then\n"
            "      /bin/kill -TERM \"${BASH_REMATCH[1]}\" 2>/dev/null || true\n"
            "      for _ in $(/usr/bin/seq 1 100); do\n"
            "        [[ ! -e $target ]] && break\n"
            "        /bin/sleep 0.005\n"
            "      done\n"
            "    fi\n"
            "    exit 1\n"
            "  fi\n"
            "fi\n"
            "exec /usr/bin/readlink \"$@\"\n",
            encoding="utf-8",
        )
        os.chmod(readlink_shim, 0o755)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "candidate_src"):
            shutil.rmtree(cls.candidate_src)
        if hasattr(cls, "local_tmp"):
            shutil.rmtree(cls.local_tmp)

    def run_current_user_mutation(
        self, mode: str, extra_environment: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "HOME": "/home/bmarti44",
            "PATH": (
                "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                "/usr/bin:/sbin:/bin"
            ),
            "GLM_CANDIDATE_SRC": str(self.candidate_src),
            "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
            "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
            "GLM_SAFE_KILL_FLOOR_GIB": "40",
            "GLM_SAFE_MIN_START_GIB": "110",
            "GLM_SAFE_TIMEOUT_S": "30",
            "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
        }
        if extra_environment:
            environment.update(extra_environment)
        return subprocess.run(
            [
                "env",
                *[
                    f"{key}={value}"
                    for key, value in environment.items()
                ],
                "bash",
                str(SAFE),
                "--tag",
                f"current-user-lifecycle-{mode}",
                "--",
                str(self.runner),
                mode,
                str(self.candidate_src / "ds4-server"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            check=False,
        )

    def test_exit_between_live_stat_and_executable_identity_is_attested(self):
        self.readlink_count.write_text("0\n", encoding="ascii")
        result = self.run_current_user_mutation(
            "identity-race",
            {
                "PATH": f"{self.shim_dir}:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
                "GLM_TEST_READLINK_COUNT": str(self.readlink_count),
                "GLM_TEST_READLINK_ACTION": "exit",
            },
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate exited during identity sample; "
            "monitoring controller and process group",
            main,
        )
        self.assertNotIn("executed candidate identity changed", main)

    def test_unavailable_identity_while_candidate_is_live_is_rejected(self):
        self.readlink_count.write_text("0\n", encoding="ascii")
        result = self.run_current_user_mutation(
            "clean",
            {
                "PATH": f"{self.shim_dir}:/usr/local/sbin:/usr/local/bin:"
                "/usr/sbin:/usr/bin:/sbin:/bin",
                "GLM_TEST_READLINK_COUNT": str(self.readlink_count),
                "GLM_TEST_READLINK_ACTION": "unavailable",
            },
        )
        self.assertEqual(result.returncode, 11, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate identity unavailable while process remained live",
            main,
        )

    def test_clean_reaped_candidate_between_sampler_ticks_is_attested(self):
        result = self.run_current_user_mutation("reaped")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate was verified alive at least once; no identity "
            "contradiction observed by the periodic sampler; actual cadence is "
            "recorded in samples.log; wrapper and descendant checks clean",
            main,
        )

    def test_clean_reaped_candidate_with_postprocessing_is_attested(self):
        result = self.run_current_user_mutation("postprocess")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate was verified alive at least once; no identity "
            "contradiction observed by the periodic sampler; actual cadence is "
            "recorded in samples.log; wrapper and descendant checks clean",
            main,
        )

    def test_zombie_candidate_during_postprocessing_is_attested(self):
        result = self.run_current_user_mutation("linger")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn(
            "executed candidate was verified alive at least once; no identity "
            "contradiction observed by the periodic sampler; actual cadence is "
            "recorded in samples.log; wrapper and descendant checks clean",
            main,
        )

    def test_replacement_candidate_during_postprocessing_is_rejected(self):
        result = self.run_current_user_mutation("replace")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonzero_controller_after_candidate_exit_is_rejected(self):
        result = self.run_current_user_mutation("fail")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_final_artifact_is_bound_to_contained_execution(self):
        state = Path("/home/bmarti44/.local/state")
        state.mkdir(parents=True, exist_ok=True)
        artifact_root = Path(tempfile.mkdtemp(prefix="glm52-final-artifact-", dir=state))
        try:
            artifact = artifact_root / "result.json"
            artifact.write_text('{"ok":true}\n', encoding="ascii")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            result = self.run_current_user_mutation(
                "postprocess", {"GLM_SAFE_FINAL_ARTIFACTS": str(artifact)}
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            match = re.search(r" dir=(\S+)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
            self.assertIn(
                f"final_artifact_verified path={artifact} sha256={digest} ", main
            )
            for name in ("samples.log", "kernel.log"):
                evidence = Path(match.group(1)) / name
                evidence_digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
                self.assertIn(
                    f"safety_artifact_verified name={name} "
                    f"sha256={evidence_digest} size={evidence.stat().st_size}",
                    main,
                )
        finally:
            shutil.rmtree(artifact_root)

    def test_missing_final_artifact_fails_closed(self):
        state = Path("/home/bmarti44/.local/state")
        state.mkdir(parents=True, exist_ok=True)
        artifact_root = Path(tempfile.mkdtemp(prefix="glm52-final-missing-", dir=state))
        try:
            result = self.run_current_user_mutation(
                "postprocess",
                {"GLM_SAFE_FINAL_ARTIFACTS": str(artifact_root / "missing.json")},
            )
            self.assertEqual(result.returncode, 17, result.stdout + result.stderr)
            match = re.search(r" dir=(\S+)", result.stdout)
            self.assertIsNotNone(match, result.stdout)
            main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
            self.assertIn("final artifact is absent or unsafe", main)
        finally:
            shutil.rmtree(artifact_root)

    def test_unrelated_live_descendant_is_rejected_and_cleaned_up(self):
        result = self.run_current_user_mutation("survivor")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        match = re.search(r" dir=(\S+)", result.stdout)
        self.assertIsNotNone(match, result.stdout)
        main = (Path(match.group(1)) / "main.log").read_text(encoding="utf-8")
        self.assertIn("isolated process group survived command completion", main)
        self.assertNotIn(
            "executed candidate was verified alive at least once; no identity "
            "contradiction observed by the periodic sampler; actual cadence is "
            "recorded in samples.log; wrapper and descendant checks clean",
            main,
        )

    def test_real_wrapper_emits_only_utc_under_host_timezone_variants(self):
        for timezone_name in (
            "America/New_York",
            "UTC",
            "Pacific/Kiritimati",
        ):
            with self.subTest(timezone_name=timezone_name):
                environment = {
                    "HOME": "/home/bmarti44",
                    "PATH": (
                        "/usr/local/sbin:/usr/local/bin:/usr/sbin:"
                        "/usr/bin:/sbin:/bin"
                    ),
                    "TZ": timezone_name,
                    "GLM_CANDIDATE_SRC": str(self.candidate_src),
                    "GLM_SAFE_LOG_CANDIDATE_PROVENANCE": "1",
                    "GLM_SAFE_EXPECTED_BINARY_SHA256": self.digest,
                    "GLM_SAFE_KILL_FLOOR_GIB": "40",
                    "GLM_SAFE_MIN_START_GIB": "110",
                    "GLM_SAFE_TIMEOUT_S": "30",
                    "GLM_SAFE_RUN_AS_CURRENT_USER": "1",
                }
                result = subprocess.run(
                    [
                        "env",
                        *[
                            f"{key}={value}"
                            for key, value in environment.items()
                        ],
                        "bash",
                        str(SAFE),
                        "--tag",
                        f"utc-{timezone_name.replace('/', '-')}",
                        "--",
                        str(self.runner),
                        "clean",
                        str(self.candidate_src / "ds4-server"),
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=45,
                    check=False,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                matches = re.findall(
                    r"SAFE_RUN_DONE rc=0 killed=no dir=(\S+)",
                    result.stdout,
                )
                self.assertEqual(
                    len(matches), 1, result.stdout + result.stderr
                )
                main = (Path(matches[0]) / "main.log").read_text()
                samples = (Path(matches[0]) / "samples.log").read_text()
                lifecycle = [
                    line.split()[0]
                    for line in main.splitlines()
                    if "executed_candidate_verified" in line
                    or "SAFE_RUN end" in line
                ]
                self.assertEqual(len(lifecycle), 2, main)
                self.assertTrue(
                    all(value.endswith("+00:00") for value in lifecycle),
                    lifecycle,
                )
                self.assertTrue(
                    all(
                        re.search(r"[.,]\d+\+00:00$", value)
                        for value in lifecycle
                    ),
                    lifecycle,
                )
                sample_timestamps = [
                    line.split()[0] for line in samples.splitlines() if line
                ]
                self.assertGreaterEqual(len(sample_timestamps), 2, samples)
                self.assertTrue(
                    all(
                        re.fullmatch(
                            r"\S+ mem_avail_kb=\d+ eng_rss_kb=\d+ "
                            r"read_bytes=\d+ "
                            r"cgroup_current_bytes=(?:\d+|na) "
                            r"cgroup_peak_bytes=(?:\d+|na) "
                            r"cgroup_swap_current_bytes=(?:\d+|na)",
                            line,
                        )
                        for line in samples.splitlines()
                    ),
                    samples,
                )
                self.assertTrue(
                    all(
                        value.endswith("+00:00")
                        for value in sample_timestamps
                    ),
                    sample_timestamps,
                )


if __name__ == "__main__":
    unittest.main()
