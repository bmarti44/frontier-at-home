#!/usr/bin/env python3
"""Run exllamav3's eval/model_diff.py with the pinned offline wikitext shim.

Usage: <runtime python> -B run_model_diff.py <model_diff.py> <workdir> -- <model_diff args>

model_diff caches the tokenized dataset next to sys.argv[0]; pointing argv[0]
into <workdir> keeps the pinned runtime source tree unmodified.
"""
import os
import runpy
import sys

here = os.path.dirname(os.path.abspath(__file__))
script, workdir = sys.argv[1], sys.argv[2]
if sys.argv[3] != "--":
    raise SystemExit("usage: run_model_diff.py <model_diff.py> <workdir> -- <args>")
sys.path.insert(0, os.path.join(here, "shim"))
os.makedirs(workdir, exist_ok=True)
sys.argv = [os.path.join(workdir, "model_diff.py"), *sys.argv[4:]]
runpy.run_path(script, run_name="__main__")
