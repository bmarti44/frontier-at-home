#!/usr/bin/env python3
"""Harmless real-systemd verifier for W7 candidate-8 interruption cleanup."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import time
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/91_run_w7_cache_generation_campaign.py"
SPEC = importlib.util.spec_from_file_location("w7_candidate8_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def emit(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)


def unit_state(unit: str) -> dict[str, object]:
    completed = subprocess.run(
        [
            "/usr/bin/systemctl", "--user", "show", unit, "--no-pager",
            "--property=LoadState", "--property=ActiveState", "--property=SubState",
            "--property=MainPID", "--property=ControlPID", "--property=ControlGroup",
        ],
        capture_output=True, text=True, check=False, timeout=15,
    )
    return {
        "rc": completed.returncode,
        "stdout": completed.stdout.strip().splitlines(),
        "stderr": completed.stderr.strip(),
    }


def marker_processes(case: str) -> list[str]:
    completed = subprocess.run(
        ["/usr/bin/pgrep", "-af", f"^glm52-c8-{case}-child 60$"],
        capture_output=True, text=True, check=False, timeout=5,
    )
    return completed.stdout.strip().splitlines()


def snapshot(case: str, phase: str, unit: str, process: subprocess.Popen[str]) -> None:
    emit({
        "type": "observation",
        "time_ns": time.time_ns(),
        "case": case,
        "phase": phase,
        "unit": unit,
        "launcher_pid": process.pid,
        "launcher_rc": process.poll(),
        "unit_state": unit_state(unit),
        "session_members": MODULE._live_launcher_session_members(process.pid),
        "marker_processes": marker_processes(case),
        "server_pids": MODULE.server_pids(),
        "listener": MODULE._listener_is_active(),
    })


def run_case(case: str, delay: int, resistant: bool, bus_failure: bool) -> None:
    tag = f"w7-c8-{case}"
    marker = f"glm52-c8-{case}-child"
    if resistant:
        command = (
            f"trap 'exit 0' TERM; "
            f"/usr/bin/bash -c 'trap \"\" TERM; exec -a {marker} /usr/bin/sleep 60' & wait"
        )
    else:
        command = (
            f"sleep {delay}; exec /usr/bin/systemd-run --user --wait --collect --pipe --quiet "
            f"--unit=\"glm52-{tag}-$$\" --service-type=exec --property=KillMode=control-group "
            f"-- /usr/bin/bash -c 'exec -a {marker} /usr/bin/sleep 60'"
        )
    process = subprocess.Popen(
        ["/usr/bin/bash", "-c", command],
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    unit = MODULE.containment_unit_name(tag, process.pid)
    if delay:
        time.sleep(0.15)
    else:
        for _ in range(80):
            if resistant or not MODULE._unit_is_stopped(unit):
                break
            time.sleep(0.05)
    snapshot(case, "before", unit, process)
    original_stop = MODULE.stop_exact_containment_unit
    original_cgroup = MODULE._kill_and_verify_containment_cgroup

    def traced_stop(selected: str) -> None:
        emit({"type": "action", "time_ns": time.time_ns(), "case": case, "action": "stop", "unit": selected})
        if bus_failure:
            raise MODULE.CampaignError("simulated bus unavailable")
        original_stop(selected)

    def traced_cgroup(selected: str) -> None:
        emit({"type": "action", "time_ns": time.time_ns(), "case": case, "action": "cgroup-kill-empty-proof", "unit": selected})
        original_cgroup(selected)

    observed_error = ""
    try:
        with mock.patch.object(MODULE, "stop_exact_containment_unit", side_effect=traced_stop), mock.patch.object(
            MODULE, "_kill_and_verify_containment_cgroup", side_effect=traced_cgroup,
        ):
            MODULE._cleanup_interrupted_containment(process, unit)
    except MODULE.CampaignError as error:
        observed_error = str(error)
    snapshot(case, "after", unit, process)
    if process.poll() is None or MODULE._live_launcher_session_members(process.pid):
        raise AssertionError(f"launcher session survived: {case}")
    if marker_processes(case) or MODULE.server_pids() or MODULE._listener_is_active():
        raise AssertionError(f"containment survivor remained: {case}")
    if not MODULE._unit_is_stopped(unit):
        raise AssertionError(f"unit remained active: {case}")
    if bus_failure and observed_error != "simulated bus unavailable":
        raise AssertionError("control-plane failure did not remain fail-closed")
    if not bus_failure and observed_error:
        raise AssertionError(observed_error)
    emit({
        "type": "case_result", "time_ns": time.time_ns(), "case": case,
        "verdict": "PASS", "expected_error": observed_error,
    })


def main() -> int:
    cases = (
        ("delayed", 2, False, False),
        ("active", 0, False, False),
        ("resistant", 0, True, False),
        ("busfail", 0, False, True),
    )
    for case in cases:
        run_case(*case)
    emit({"type": "summary", "time_ns": time.time_ns(), "cases": 4, "verdict": "PASS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
