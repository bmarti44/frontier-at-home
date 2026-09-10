"""Actual unchanged launch validator against the declared next configuration."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('original_probe', ROOT / 'scripts/48_probe_glm53_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
profile = json.loads((ROOT / 'configs/profiles/glm-5.3-flash/cuda-spark-128g-1m-experimental.json').read_text())
argv = profile['launch']['args'][4:]
for flag, value in [('--max-num-batched-tokens', '256'), ('--long-prefill-token-threshold', '64')]:
    argv[argv.index(flag) + 1] = value
probe.check_launch({'arguments': argv})
