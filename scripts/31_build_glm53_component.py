#!/usr/bin/env python3
"""Build one frozen GLM component through the existing guarded cgroup wrapper.

No model is loaded and no wheel is installed. Each attempt gets a new output
directory. Failed outputs are retained; retry from a newly prepared clean tree.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git(source, *args):
    return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--component", choices=("exllamav3", "vllm", "vllm-exl3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared, runtime, uv = (p.resolve() for p in (args.prepared, args.runtime, args.uv))
    output = args.output.resolve()
    python = runtime / "bin/python3"
    if "include-system-site-packages = false" not in (runtime / "pyvenv.cfg").read_text():
        raise ValueError("build environment must exclude system site packages")
    manifest = json.loads((prepared / "manifest.json").read_text())
    if manifest["source_lock_sha256"] != sha(ROOT / "configs/build-manifests/glm53-flash-sources.json"):
        raise ValueError("prepared source lock is stale")
    if manifest["binary_sha256"] != sha(python.resolve()):
        raise ValueError("prepared interpreter identity mismatch")
    for name, entry in manifest["sources"].items():
        path = prepared / name
        if git(path, "rev-parse", "HEAD") != entry["revision"] or git(path, "rev-parse", "HEAD^{tree}") != entry["tree"]:
            raise ValueError(f"{name}: source identity mismatch")
        # Previously built components may contain ignored build artifacts;
        # this component must be clean including ignored files before building.
        status_args = ["status", "--porcelain", "--untracked-files=all"]
        if name == args.component:
            status_args.append("--ignored")
        if git(path, *status_args):
            raise ValueError(f"{name}: source is not clean")
    env = {"HOME": str(Path.home()), "PATH": f"{runtime}/bin:/usr/local/cuda-13.0/bin:/usr/bin:/bin",
           "LANG": "C.UTF-8", "CUDA_HOME": "/usr/local/cuda-13.0",
           "MAX_JOBS": "2", "CMAKE_BUILD_PARALLEL_LEVEL": "2", "NVCC_THREADS": "1", "MAKEFLAGS": "-j2",
           "TORCH_CUDA_ARCH_LIST": "12.1a", "FLASHINFER_CUDA_ARCH_LIST": "12.1a",
           "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
           "EXL3_EXT_INCLUDE": str(prepared / "exllamav3/exllamav3/exllamav3_ext"),
           "VLLM_TARGET_DEVICE": "cuda", "VLLM_USE_PRECOMPILED": "0", "VLLM_USE_PRECOMPILED_RUST": "0"}
    subprocess.run([str(uv), "pip", "check", "--python", str(python)], env=env, check=True)
    output.mkdir(parents=True, exist_ok=False)
    command = ["/usr/bin/env", "-i", *(f"{k}={v}" for k, v in sorted(env.items())), str(uv),
               "build", "--wheel", "--no-build-isolation", "--no-cache", "--python", str(python),
               "--out-dir", str(output / "wheels"), str(prepared / args.component)]
    wrapper = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
    safe = wrapper.with_name("glm_safe_run.sh")
    receipt = {"schema_version": 1, "status": "build_attempt_only", "component": args.component,
               "source_manifest_sha256": sha(prepared / "manifest.json"), "sources": manifest["sources"],
               "scorer_sha256": sha(Path(__file__)), "binary_sha256": sha(python.resolve()),
               "wrapper_sha256": sha(wrapper), "safe_run_sha256": sha(safe), "uv_sha256": sha(uv),
               "command": command, "start_unix": time.time()}
    (output / "manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    outer = dict(os.environ)
    # Do not inherit optional wrapper overrides or parent-lock exemptions.
    for key in list(outer):
        if key.startswith(("GLM_SAFE_", "GLM_W1_")):
            del outer[key]
    outer.update(GLM_SAFE_RUN_AS_CURRENT_USER="1", GLM_SAFE_TIMEOUT_S="9000",
                 GLM_SAFE_KILL_FLOOR_GIB="18", GLM_SAFE_MIN_START_GIB="110")
    with (output / "raw.log").open("wb") as log:
        result = subprocess.run(["bash", str(wrapper), "--tag", f"glm53-build-{args.component}", "--", *command],
                                cwd=ROOT, env=outer, stdout=log, stderr=subprocess.STDOUT)
    wheels = [{"path": p.name, "sha256": sha(p), "size_bytes": p.stat().st_size}
              for p in sorted((output / "wheels").glob("*.whl"))]
    summary = {"verdict": "PASS" if result.returncode == 0 and len(wheels) == 1 else "FAIL",
               "qualification": "build_only", "exit_code": result.returncode,
               "end_unix": time.time(), "wheels": wheels, "raw_sha256": sha(output / "raw.log")}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
