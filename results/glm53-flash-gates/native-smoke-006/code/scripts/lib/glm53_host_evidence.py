"""Score raw host and Python identity observations for model-free GLM probes.

The caller must separately freeze this scorer and its expected arguments, bind
the runtime/model/tokenizer/fixture inventories and public seed, and validate the
inner probe. This component never authorizes serving or claims model capacity.
No Torch, CUDA, model loading, system mutation or synthetic replacement evidence.
"""
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import stat

UNIT_PROPERTIES = ("Id", "LoadState", "ActiveState", "SubState", "MainPID", "ControlGroup",
                   "MemoryHigh", "MemoryMax", "MemorySwapMax", "OOMPolicy", "KillMode")
SAMPLE_FIELDS = {"mem_avail_kb", "eng_rss_kb", "read_bytes", "cgroup_current_bytes", "cgroup_peak_bytes", "cgroup_swap_current_bytes"}
FAULTS = re.compile(r"NVRM.*(?:Xid|NV_ERR_NO_MEMORY|Out of memory)|oom-kill|Out of memory: Killed process|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|CUDA.{0,160}(?:allocation failed|out of memory)", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unit_query(unit):
    require(isinstance(unit, str) and re.fullmatch(r"glm52-glm53-[a-z0-9-]+-[1-9][0-9]*\.service", unit), "invalid probe unit")
    return ["/usr/bin/systemctl", "--user", "show", unit, *("--property=" + name for name in UNIT_PROPERTIES)]


def read(path):
    path = Path(path)
    for part in (path, *path.parents):
        require(not part.is_symlink(), "symlink in host evidence")
    value = path.stat()
    require(stat.S_ISREG(value.st_mode) and value.st_size <= 16 * 1024 * 1024, "invalid host evidence file size or type")
    data = path.read_bytes()
    identity = lambda info: (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    require(len(data) == value.st_size and identity(path.stat()) == identity(value), "host evidence changed during read")
    return data


def object_text(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate evidence JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite evidence JSON value")
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite evidence JSON value")
        return result
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)
    require(isinstance(value, dict), "evidence JSON object required")
    return value


def obj(path):
    return object_text(read(path))


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "finite evidence number required")
    return value


def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, "invalid evidence integer")
    return value


def one(pattern, text, message):
    matches = list(re.finditer(pattern, text, re.M))
    require(len(matches) == 1, message)
    return matches[0]


def record(marker, pattern, text, message):
    # Count all records of this kind before validating success/field syntax.
    # Failed or malformed records must not disappear behind a success-only regex.
    lines = [line for line in text.splitlines() if re.search(r"(?:^|\s)" + re.escape(marker) + r"(?:=|\s|$)", line)]
    require(len(lines) == 1, message)
    match = re.fullmatch(pattern, lines[0])
    require(match is not None, message)
    return match


def timestamp(text):
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None, "timestamp lacks timezone")
    return value.timestamp()


def parse_samples(text):
    result = []
    for line in text.splitlines():
        pieces = line.split()
        require(len(pieces) == len(SAMPLE_FIELDS) + 1, "malformed memory sample")
        fields = {}
        for piece in pieces[1:]:
            require(re.fullmatch(r"[a-z_]+=[0-9]+", piece), "malformed memory sample field")
            key, value = piece.split("=")
            require(key not in fields, "duplicate memory sample field")
            fields[key] = int(value)
        require(set(fields) == SAMPLE_FIELDS, "missing memory sample coverage")
        result.append({"time": timestamp(pieces[0]), **fields})
    require(len(result) >= 3, "insufficient memory sampling coverage")
    return result


def ordered_times(rows, key, gap):
    values = [number(row[key]) for row in rows]
    require(all(b > a and b - a <= gap for a, b in zip(values, values[1:])), "missing or unordered sampling coverage")
    return values


def parse_unit(path, unit):
    record = obj(path)
    require(record["command"] == unit_query(unit) and type(record["returncode"]) is int and record["returncode"] == 0 and record["stderr"] == "", "unit observation command failed or changed")
    properties = {}
    for line in record["stdout"].splitlines():
        require("=" in line, "malformed unit observation")
        key, value = line.split("=", 1)
        require(key in UNIT_PROPERTIES and key not in properties, "unknown or duplicate unit property")
        properties[key] = value
    require(set(properties) == set(UNIT_PROPERTIES) and properties["Id"] == unit, "unit property coverage mismatch")
    return properties, number(record["observed_at"])


def swap_snapshot(path):
    record = obj(path)
    require(record["path"] == "/proc/vmstat", "wrong swap observation source")
    fields = {}
    for line in record["text"].splitlines():
        require(re.fullmatch(r"[a-z0-9_]+ [0-9]+", line), "malformed vmstat observation")
        key, value = line.split()
        require(key not in fields, "duplicate vmstat counter")
        fields[key] = int(value)
    require({"pswpin", "pswpout"} <= fields.keys(), "missing swap counter coverage")
    return {key: fields[key] for key in ("pswpin", "pswpout")}, number(record["observed_at"])


def score_host_observations(directory, expected):
    """Return a host-only verdict; raise on missing or contradictory evidence."""
    root = Path(directory)
    floor = integer(expected["kill_floor_gib"], 40)
    start = integer(expected["minimum_start_gib"], 110)
    timeout = integer(expected["timeout_seconds"], 1)
    gap = number(expected["maximum_sample_gap_seconds"])
    require(0 < gap <= 2, "sampling gap may not exceed two seconds")
    prefix = expected["unit_prefix"]
    require(isinstance(prefix, str) and re.fullmatch(r"glm52-glm53-[a-z0-9-]+-", prefix), "invalid expected unit prefix")
    for name in ("binary_sha256", "guard_sha256", "probe_sha256", "environment_sha256"):
        require(isinstance(expected[name], str) and re.fullmatch(r"[0-9a-f]{64}", expected[name]), "invalid expected digest")
    logs = {name: read(root / name) for name in ("wrapper.log", "main.log", "samples.log", "kernel.log")}
    done = record("SAFE_RUN_DONE", r"SAFE_RUN_DONE rc=0 killed=no dir=(\S+) main_sha256=([0-9a-f]{64}) samples_sha256=([0-9a-f]{64}) kernel_sha256=([0-9a-f]{64})",
               logs["wrapper.log"].decode(), "missing successful digest-bound wrapper completion")
    for name, digest in zip(("main.log", "samples.log", "kernel.log"), done.groups()[1:]):
        require(hashlib.sha256(logs[name]).hexdigest() == digest, "wrapper raw log digest mismatch")
    main, kernel = logs["main.log"].decode(), logs["kernel.log"].decode()
    require("FATAL" not in main and not FAULTS.search(main), "fatal wrapper or GPU failure")
    require(bool(kernel.strip()) and not FAULTS.search(kernel) and
            not re.search(r"Failed to (?:read|open)|Permission denied|Cannot access|No journal files", kernel, re.I), "kernel fault or incomplete journal evidence")
    launch = record("SAFE_RUN start", r"(\S+) SAFE_RUN start tag=(\S+) vlimit_kb=([1-9][0-9]*) kill_floor_gib=([0-9]+) min_start_gib=([0-9]+) timeout_s=([0-9]+) allow_cgroup_high=0", main, "missing or invalid wrapper start controls")
    launch_time = timestamp(launch.group(1))
    require(launch.group(2) == prefix.removeprefix("glm52-").removesuffix("-") and
            tuple(map(int, launch.groups()[3:])) == (floor, start, timeout), "frozen wrapper start controls mismatch")
    end = record("SAFE_RUN end", r"(\S+) SAFE_RUN end rc=0 killed=no \(124=timeout, 137=SIGKILL/ENOMEM-adjacent\)", main, "wrapper terminal exit missing, duplicated or failed")
    end_time = timestamp(end.group(1))
    controls = record("cgroup_verified", r"(\S+) cgroup_verified path=(\S+) memory_high=([0-9]+) memory_max=([0-9]+) memory_swap_max=0 memory_oom_group=1", main, "missing verified cgroup controls")
    control_time = timestamp(controls.group(1))
    cgroup, high, maximum = controls.groups()[1:]; high, maximum = int(high), int(maximum)
    unit = cgroup.rsplit("/", 1)[-1]
    require(re.fullmatch(re.escape(prefix) + r"[1-9][0-9]*\.service", unit), "wrong attempt cgroup identity")
    require(cgroup == "/user.slice/user-1000.slice/user@1000.service/app.slice/" + unit, "unexpected probe containment path")
    require(0 < high < maximum, "invalid cgroup memory envelope")
    total = int(one(r"^MemTotal:\s+([0-9]+) kB$", main, "missing physical host memory").group(1)) * 1024
    require(maximum + floor * 2**30 <= total, "cgroup memory envelope violates whole-host floor")
    starts = [object_text(line) for line in main.splitlines() if line.startswith('{"pass":')]
    require(len(starts) == 1 and starts[0].get("pass") is True and number(starts[0]["required_gib"]) == start and
            number(starts[0]["mem_available_gib"]) >= start and integer(starts[0]["stable_samples_observed"]) >= 3, "stable start-memory evidence missing")
    process = record("wrapper_pid", r"(\S+) wrapper_pid=([0-9]+) engine_pid=([0-9]+) pgid=([0-9]+) .*", main, "missing sampled process identity")
    process_time = timestamp(process.group(1))
    wrapper_pid, engine_pid, pgid = map(int, process.groups()[1:])
    require(wrapper_pid == pgid and wrapper_pid != engine_pid and engine_pid > 0, "invalid sampled process identity")
    final = record("cgroup_final", r"(\S+) cgroup_final current_bytes=([0-9]+) peak_bytes=([0-9]+) swap_current_bytes=0 events=(.+)", main, "missing final cgroup counters")
    final_time = timestamp(final.group(1))
    current, peak = int(final.group(2)), int(final.group(3))
    events = {}
    for field in final.group(4).rstrip(",").split(","):
        require(re.fullmatch(r"[a-z_]+ [0-9]+", field), "malformed cgroup event")
        key, value = field.split(); require(key not in events, "duplicate cgroup event"); events[key] = int(value)
    require({"low", "high", "max", "oom", "oom_kill", "oom_group_kill"} <= events.keys(), "missing cgroup event coverage")
    require(all(value == 0 for key, value in events.items() if key != "low"), "cgroup pressure, OOM or kill event")
    require(current <= peak <= maximum, "invalid final cgroup memory accounting")
    samples = parse_samples(logs["samples.log"].decode())
    host_times = ordered_times(samples, "time", gap)
    minimum = min(row["mem_avail_kb"] for row in samples)
    require(minimum >= floor * 2**20, "whole-host memory floor failed")
    require(all(row["mem_avail_kb"] * 1024 <= total for row in samples), "impossible host memory sample")
    require(all(row["cgroup_swap_current_bytes"] == 0 for row in samples), "unexpected cgroup swap")
    require(all(row["eng_rss_kb"] > 0 and row["cgroup_current_bytes"] <= row["cgroup_peak_bytes"] <= peak for row in samples), "invalid sampled memory accounting")
    require(all(b["cgroup_peak_bytes"] >= a["cgroup_peak_bytes"] for a, b in zip(samples, samples[1:])), "cgroup peak counter decreased")

    identity = obj(root / "identity/manifest.json")
    require(identity["qualification"] == "Python_probe_identity_only" and identity["executable"] == expected["executable"] and
            identity["binary_sha256"] == expected["binary_sha256"] and identity["environment_sha256"] == expected["environment_sha256"] and
            identity["cgroup"] == "0::" + cgroup + "\n" and number(identity["period_seconds"]) == 0.25, "frozen identity manifest mismatch")
    require(isinstance(identity["device_inode"], list) and len(identity["device_inode"]) == 2, "missing interpreter inode binding")
    for value in identity["device_inode"]: integer(value, 1)
    source_files = identity["source_files"]
    require(isinstance(source_files, list) and len(source_files) == 2 and
            {tuple(sorted(row.items())) for row in source_files} == {
                tuple(sorted({"path": expected["guard"], "sha256": expected["guard_sha256"]}.items())),
                tuple(sorted({"path": expected["probe"], "sha256": expected["probe_sha256"]}.items()))}, "probe/guard source binding mismatch")
    argv = identity["argv"]
    require(isinstance(argv, list) and len(argv) == 8 + len(expected["probe_arguments"]) and
            argv[:5] == [expected["executable"], "-I", "-B", expected["guard"], "--child"] and
            argv[7:] == [expected["probe"], *expected["probe_arguments"]], "identity argv binding mismatch")
    require(all(isinstance(fd, str) and re.fullmatch(r"[0-9]+", fd) and 3 <= int(fd) < 1048576 for fd in argv[5:7]) and argv[5] != argv[6], "invalid identity handshake descriptors")
    raw = read(root / "identity/raw.jsonl")
    rows = [object_text(line) for line in raw.splitlines()]
    require(len(rows) >= 3 and rows[-1].get("event") == "cleanup" and rows[-1].get("live_process_group_after") == [] and
            all(row.get("event") == "identity" for row in rows[:-1]), "incomplete identity or cleanup event coverage")
    times = ordered_times(rows, "time_unix", gap)
    monotonic = [integer(row["monotonic_ns"], 1) for row in rows]
    require(all(0 < b - a <= gap * 1e9 for a, b in zip(monotonic, monotonic[1:])), "identity monotonic coverage mismatch")
    require(all(abs((b - a) - (mb - ma) / 1e9) <= 0.05 for a, b, ma, mb in zip(times, times[1:], monotonic, monotonic[1:])), "identity clocks disagree")
    identities = rows[:-1]
    first_ticks = integer(identities[0]["start_ticks"], 1)
    for index, row in enumerate(identities):
        require(type(row["pid"]) is int and row["pid"] == engine_pid and row["pgid"] == engine_pid and
                type(row["start_ticks"]) is int and row["start_ticks"] == first_ticks and row["cgroup"] == identity["cgroup"] and
                row["binary_sha256"] == expected["binary_sha256"], "continuous process identity contradiction")
        require(all(row.get(key) is True for key in ("executable_verified", "argv_verified", "environment_verified")), "identity observation did not verify")
        terminal = index == len(identities) - 1
        require(row.get("completion_verified", False) is terminal and row.get("terminal_exec_filter_verified") is terminal, "missing or misplaced terminal identity/filter completion")
        integer(row["seccomp_filters"])
    require(identities[-1]["seccomp_filters"] > identities[0]["seccomp_filters"], "terminal exec filter count did not increase")
    require(times[0] >= number(identity["start_unix"]) and times[-1] - number(identity["start_unix"]) <= timeout and
            abs(host_times[0] - times[0]) <= gap and abs(host_times[-1] - times[-2]) <= gap, "host/identity sampling windows lack coverage")
    require(launch_time <= control_time <= number(identity["start_unix"]) <= times[0] and
            control_time <= process_time <= host_times[0] and abs(process_time - times[0]) <= gap and
            max(host_times[-1], times[-1]) <= final_time <= end_time and final_time - times[-1] <= gap,
            "wrapper timestamp chronology contradicts probe or memory windows")
    summary = obj(root / "identity/summary.json")
    require(summary["verdict"] == "PASS" and summary["qualification"] == "Python_probe_identity_only" and
            type(summary["probe_exit_code"]) is int and summary["probe_exit_code"] == 0 and summary["failure"] is None and
            summary["live_process_group_after"] == [] and type(summary["identity_samples"]) is int and summary["identity_samples"] == len(identities) and
            summary["raw_sha256"] == hashlib.sha256(raw).hexdigest(), "identity verdict or raw digest mismatch")

    live, live_time = parse_unit(root / "unit-live.json", unit)
    require(live["LoadState"] == "loaded" and live["ActiveState"] == "active" and live["SubState"] == "running" and
            re.fullmatch(r"[1-9][0-9]*", live["MainPID"]) and live["ControlGroup"] == cgroup and
            live["MemoryHigh"] == str(high) and live["MemoryMax"] == str(maximum) and live["MemorySwapMax"] == "0" and
            live["OOMPolicy"] == "kill" and live["KillMode"] == "control-group", "actual unit containment mismatch")
    require(times[0] - gap <= live_time <= times[-1], "unit was not observed during probe lifetime")
    after, after_time = parse_unit(root / "unit-after.json", unit)
    require(after["LoadState"] in ("loaded", "not-found") and after["ActiveState"] == "inactive" and
            after["MainPID"] == "0" and after["ControlGroup"] == "" and after_time >= end_time, "unit cleanup was not verified")
    cgroup_after = obj(root / "cgroup-after.json")
    require(cgroup_after["path"] == "/sys/fs/cgroup" + cgroup and cgroup_after["exists"] is False and
            number(cgroup_after["observed_at"]) >= after_time, "cgroup cleanup was not verified")
    swap_before, before_time = swap_snapshot(root / "swap-before.json")
    swap_after, swap_time = swap_snapshot(root / "swap-after.json")
    require(before_time <= launch_time and swap_time >= number(cgroup_after["observed_at"]), "swap observations do not cover attempt")
    require(swap_before == swap_after, "unexpected whole-system swap counter delta")
    return {"verdict": "PASS", "qualification": "host_and_probe_identity_observations_only", "unit": unit,
            "minimum_mem_available_kib": minimum, "maximum_cgroup_peak_bytes": peak, "memory_samples": len(samples),
            "identity_samples": len(identities), "identity_pid": engine_pid, "identity_start_ticks": first_ticks,
            "maximum_memory_sample_gap_seconds": max(b - a for a, b in zip(host_times, host_times[1:])),
            "maximum_identity_sample_gap_seconds": max(b - a for a, b in zip(times, times[1:])),
            "whole_system_swap_delta": {"pswpin": 0, "pswpout": 0},
            "required_separate_checks": ["frozen runtime and metadata inventories", "post-freeze verified public seed", "inner probe correctness", "model qualification"]}
