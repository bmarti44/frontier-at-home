#!/usr/bin/env python3
"""The smoke probe (../smoke-2026-11-23/probe.py, unchanged) with every request rendered at
reasoning effort low, the effort screen carrier a6's reference probe ran at by default.

vLLM merges a request's chat_template_kwargs over --default-chat-template-kwargs
(vllm/renderers/params.py merge_kwargs: defaults | overrides), so the rendered prompt is
a6's whatever effort this candidate serves, and clause 6 compares like with like
(review A-M3-1). Usage: as probe.py (--base-url URL --out PATH).
"""
import importlib.util
from pathlib import Path

PROBE = Path(__file__).resolve().parents[1] / "smoke-2026-11-23" / "probe.py"
spec = importlib.util.spec_from_file_location("smoke_probe", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
post = probe.post


def post_low(base, body, stream=False):
    return post(base, {**body, "chat_template_kwargs": {"reasoning_effort": "low"}}, stream)


probe.post = post_low

if __name__ == "__main__":
    probe.main()
