#!/usr/bin/env python3
"""Mutation tests for bundle_attempt.py (review round 2, findings 1-3).

Builds synthetic attempt directories from the real, clean diag-cache42gb run
(its containment logs and timestamps) and checks that every missing, stale or
failing piece of evidence yields FAIL, that the text arm stays PENDING_REVIEW
without an explicit review, and that exit codes are 0/1/2.

Candidate 5 (review round 3) adds: text reviews bound to the output.txt hash,
fidelity bundles re-scored from in-bundle copies at frozen hashes with scorer
exit 0, frozen-input checks after the run and after scoring, and full-UTC
start/finish bounds on artifact mtimes.

Candidate 6 (review round 4): frozen checks must share one commit and one
actual-hash map that matches the scorer and frozen-inputs file on disk.

Candidate 7 (review round 5): records come from frozen_check.record() at the
committed HEAD (so this suite must run on a clean, committed candidate;
test_genuine_record_is_ok says so if not), and forged records - invented
commit, abbreviated map, altered hashes, missing checks - must FAIL.

Candidate 8 (review round 6): records carry their phase (file name) and
recorded_at, placed in time as the runner writes them; a launch record
replayed into later phases and fixture drift after the run must FAIL.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLER = os.path.join(HERE, "bundle_attempt.py")
SCORER = os.path.join(HERE, "score_fidelity.py")
FROZEN = os.path.join(HERE, "..", "smoke-2026-09-25", "frozen-inputs-c8.json")
sys.path.insert(0, HERE)
import frozen_check as fc  # noqa: E402
GENUINE = fc.record(FROZEN, "genuine")
SOURCE = os.path.join(HERE, "..", "smoke-2026-09-25", "diag-cache42gb")
KEEP = ["command.txt", "unit.txt", "exit_code.txt", "containment"]
FIX = ("/home/bmarti44/.cache/ds4-v41-0aaea5a2/gguf-tools/quality-testing/"
       "deepseek-v4.1-flash-20260919-router")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


LAUNCH, POST_RUN, POST_SCORE = fc.LAUNCH, fc.POST_RUN, fc.POST_SCORE


def frozen_check(phase, **over):
    rec = json.loads(json.dumps(GENUINE))
    rec["phase"] = phase
    rec.update(over)
    return rec


def utc(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + ",%09d+00:00" % int((ts % 1) * 1e9)


def steps(n):
    word = list(b" token")
    return {"source": "ds4", "prompt_tokens": 20, "ctx": 8192, "top_k": 5,
            "steps": [{"step": i, "selected": {"id": 1000 + i, "text": " token", "bytes": word},
                       "top_logprobs": []} for i in range(n)]}


class BundleMutations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = os.path.join(self.tmp.name, "attempt")
        os.makedirs(self.d)
        for name in KEEP:
            src = os.path.join(SOURCE, name)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(self.d, name))
            else:
                shutil.copy2(src, os.path.join(self.d, name))
        # Fresh mtimes inside a run window that brackets "now".
        now = time.time()
        self.start, self.finish = now - 60, now + 60
        self.at = {LAUNCH: self.start - 2, POST_RUN: self.finish + 2, POST_SCORE: self.finish + 10}
        for root, _, files in os.walk(self.d):
            for f in files:
                os.utime(os.path.join(root, f))
        self.write("started_at.txt", utc(now - 60) + "\n")
        self.write("finished_at.txt", utc(now + 60) + "\n")
        self.write("identity.json", {"verified": True})
        for name in (LAUNCH, POST_RUN):
            self.put(name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, obj):
        with open(os.path.join(self.d, name), "w") as f:
            if isinstance(obj, str):
                f.write(obj)
            else:
                json.dump(obj, f)

    def put(self, name, rec=None, **over):
        """Write a phase record where and when the runner would."""
        if rec is None:
            over.setdefault("recorded_at", self.at[name])
            rec = frozen_check(name, **over)
        self.write(name, rec)
        t = rec.get("recorded_at", self.at[name]) if isinstance(rec.get("recorded_at"), (int, float)) \
            else self.at[name]
        os.utime(os.path.join(self.d, name), (t, t))

    def review(self, ok, output=None):
        if output is None:
            with open(os.path.join(self.d, "output.txt"), "rb") as f:
                output = f.read()
        self.write("text-review.json", {"coherent_on_topic": ok,
                                        "artifact_sha256": {"output.txt": hashlib.sha256(output).hexdigest()}})

    def fidelity(self, cuda_src=None):
        """A self-comparison fidelity attempt scored exactly as the runner does."""
        shutil.copy(cuda_src or os.path.join(FIX, "results", "base-default.tsv"), os.path.join(self.d, "cuda.tsv"))
        shutil.copy(os.path.join(FIX, "results", "base-default.tsv"), os.path.join(self.d, "metal-reference.tsv"))
        shutil.copy(os.path.join(FIX, "manifest.tsv"), os.path.join(self.d, "fixture-manifest.tsv"))
        rc = subprocess.run([sys.executable, SCORER] + [os.path.join(self.d, n) for n in
                            ("cuda.tsv", "metal-reference.tsv", "fixture-manifest.tsv", "fidelity-summary.json")],
                            capture_output=True).returncode
        self.write("fidelity-score-exit.txt", f"{rc}\n")
        self.put(POST_SCORE)

    def bundle(self, arm):
        proc = subprocess.run([sys.executable, BUNDLER, self.d, arm], capture_output=True, text=True)
        with open(os.path.join(self.d, "summary.json")) as f:
            return proc.returncode, json.load(f)["verdict"]

    def test_clean_diag_passes(self):
        self.assertEqual(self.bundle("diag"), (0, "PASS"))

    def test_missing_kernel_log_fails(self):
        os.remove(os.path.join(self.d, "containment", "kernel.log"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_samples_fails(self):
        os.remove(os.path.join(self.d, "containment", "samples.log"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_identity_fails(self):
        os.remove(os.path.join(self.d, "identity.json"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_unverified_identity_fails(self):
        self.write("identity.json", {"verified": False})
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_stale_artifact_fails(self):
        p = os.path.join(self.d, "containment", "cmd.log")
        os.utime(p, (1_000_000_000, 1_000_000_000))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_kernel_fault_fails(self):
        with open(os.path.join(self.d, "containment", "kernel.log"), "a") as f:
            f.write("kernel: NVRM: Xid (PCI:0000:01:00): 31, pid=1\n")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_nonzero_exit_fails(self):
        self.write("exit_code.txt", "16\n")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_text_without_steps_fails(self):
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_short_fails(self):
        self.write("steps.json", steps(100))
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_without_review_is_pending(self):
        self.write("steps.json", steps(130))
        self.assertEqual(self.bundle("text"), (2, "PENDING_REVIEW"))
        with open(os.path.join(self.d, "output.txt")) as f:
            self.assertTrue(f.read().startswith(" token token"))

    def test_text_negative_review_fails(self):
        self.write("steps.json", steps(130))
        self.bundle("text")
        self.review(False)
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_positive_review_passes(self):
        self.write("steps.json", steps(130))
        self.bundle("text")
        self.review(True)
        self.assertEqual(self.bundle("text"), (0, "PASS"))

    def test_text_unbound_review_fails(self):
        self.write("steps.json", steps(130))
        self.write("text-review.json", {"coherent_on_topic": True})
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_text_review_of_other_output_fails(self):
        self.write("steps.json", steps(130))
        self.review(True, output=b"a different, coherent answer")
        self.assertEqual(self.bundle("text"), (1, "FAIL"))

    def test_missing_post_run_frozen_check_fails(self):
        os.remove(os.path.join(self.d, "frozen-check-post-run.json"))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_post_run_frozen_mismatch_fails(self):
        self.put(POST_RUN, ok=False)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_genuine_record_is_ok(self):
        self.assertTrue(GENUINE["ok"], "run on a clean, committed candidate: %s" % GENUINE["checks"])
        self.assertEqual(self.bundle("diag"), (0, "PASS"))

    def test_bare_ok_frozen_check_fails(self):
        self.put(POST_RUN, {"ok": True})
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def write_all(self, **over):
        for name in (LAUNCH, POST_RUN):
            self.put(name, **over)

    def test_invented_commit_fails(self):
        # Sol round 5: consistent phases, ok=true, commit 111...111.
        self.write_all(commit="1" * 40)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_abbreviated_map_fails(self):
        amap = {k: v for k, v in GENUINE["component_sha256"].items()
                if k in ("score_fidelity.py", "frozen_inputs")}
        self.write_all(component_sha256=amap)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_consistently_altered_hash_fails(self):
        amap = dict(GENUINE["component_sha256"], **{"run_smoke.sh": "0" * 64})
        self.write_all(component_sha256=amap)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_check_key_fails(self):
        checks = {k: v for k, v in GENUINE["checks"].items() if k != "frozen_paths_clean"}
        self.write_all(checks=checks)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_real_older_commit_fails(self):
        self.write_all(commit="025427486e2da118b2c980a0c2bf555fdacfd73a")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_frozen_commit_changed_between_phases_fails(self):
        self.put(POST_RUN, commit="2" * 40)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_frozen_hashes_changed_between_phases_fails(self):
        rec = frozen_check(POST_RUN, recorded_at=self.at[POST_RUN])
        rec["component_sha256"]["run_smoke.sh"] = "0" * 64
        self.put(POST_RUN, rec)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_launch_record_replayed_into_post_run_fails(self):
        # Sol round 6: the launch record copied verbatim into a later phase.
        with open(os.path.join(self.d, LAUNCH)) as f:
            launch = json.load(f)
        self.put(POST_RUN, launch)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_launch_record_replayed_with_phase_renamed_fails(self):
        with open(os.path.join(self.d, LAUNCH)) as f:
            launch = json.load(f)
        launch["phase"] = POST_RUN
        self.put(POST_RUN, launch)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_launch_record_replayed_into_post_score_fails(self):
        self.fidelity()
        with open(os.path.join(self.d, POST_RUN)) as f:
            rec = json.load(f)
        rec["phase"] = POST_SCORE
        self.put(POST_SCORE, rec)
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_mtime_disagreeing_with_recorded_at_fails(self):
        os.utime(os.path.join(self.d, POST_RUN), (self.at[POST_RUN] + 120,) * 2)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_stale_launch_record_fails(self):
        self.put(LAUNCH, recorded_at=self.start - 3600)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_missing_recorded_at_fails(self):
        rec = frozen_check(POST_RUN)
        rec.pop("recorded_at", None)
        self.put(POST_RUN, rec)
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_fixture_drift_at_bundling_fails(self):
        # Records genuine, but an external fixture no longer hashes to them.
        drift = dict(fc.UNTRACKED)
        tmp = os.path.join(self.tmp.name, "drifted-manifest.tsv")
        with open(fc.UNTRACKED["fixture_manifest.tsv"], "rb") as f, open(tmp, "wb") as g:
            g.write(f.read() + b"\n")
        drift["fixture_manifest.tsv"] = tmp
        env = dict(os.environ, PYTHONPATH=HERE)
        code = ("import sys, frozen_check as fc, runpy; fc.UNTRACKED.update(%r); "
                "sys.argv = ['bundle_attempt.py', %r, 'diag']; "
                "runpy.run_path(%r, run_name='__main__')" % (drift, self.d, BUNDLER))
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        with open(os.path.join(self.d, "summary.json")) as f:
            summary = json.load(f)
        self.assertEqual((proc.returncode, summary["verdict"]), (1, "FAIL"))
        self.assertIn("fixture_manifest.tsv on disk differs from the recorded hash",
                      summary["metrics"]["frozen_check_failures"])

    def test_non_utc_start_fails(self):
        self.write("started_at.txt", utc(time.time() - 60).replace("+00:00", "-12:00") + "\n")
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_artifact_after_finish_fails(self):
        p = os.path.join(self.d, "containment", "cmd.log")
        later = time.time() + 3600
        os.utime(p, (later, later))
        self.assertEqual(self.bundle("diag"), (1, "FAIL"))

    def test_fidelity_summary_without_tsv_fails(self):
        self.write("fidelity-summary.json", {"verdict": "PASS"})
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_failed_score_fails(self):
        self.write("cuda.tsv", "id\tnll\ncase_000\t1.0\n")
        self.write("fidelity-summary.json", {"verdict": "FAIL"})
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_clean_self_comparison_passes(self):
        self.fidelity()
        self.assertEqual(self.bundle("fidelity"), (0, "PASS"))
        with open(os.path.join(self.d, "raw.jsonl")) as f:
            kinds = [json.loads(l)["type"] for l in f]
        self.assertEqual(kinds.count("fidelity_case"), 112)
        self.assertEqual(kinds.count("fidelity_reference_case"), 112)

    def test_fidelity_scorer_nonzero_exit_fails(self):
        self.fidelity()
        self.write("fidelity-score-exit.txt", "1\n")
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_forged_summary_fails(self):
        self.fidelity()
        p = os.path.join(self.d, "fidelity-summary.json")
        with open(p) as f:
            s = json.load(f)
        s["nll_delta_upper95"] = 0.0
        s["cases"] = 111
        self.write("fidelity-summary.json", s)
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_cuda_swapped_after_scoring_fails(self):
        self.fidelity()
        with open(os.path.join(self.d, "cuda.tsv")) as f:
            rows = f.read().splitlines()
        head, body = rows[0], rows[1:]
        nll = head.split("\t").index("nll")
        body = ["\t".join(c if i != nll else str(float(c) * 1.5) for i, c in enumerate(r.split("\t")))
                for r in body]
        self.write("cuda.tsv", "\n".join([head] + body) + "\n")
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_tampered_reference_copy_fails(self):
        self.fidelity()
        with open(os.path.join(self.d, "metal-reference.tsv"), "a") as f:
            f.write("\n")
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))

    def test_fidelity_missing_post_score_check_fails(self):
        self.fidelity()
        os.remove(os.path.join(self.d, "frozen-check-post-score.json"))
        self.assertEqual(self.bundle("fidelity"), (1, "FAIL"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
