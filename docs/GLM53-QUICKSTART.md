# GLM-5.3-Flash local bring-up

GLM uses a separate authenticated endpoint at `http://127.0.0.1:8015/v1`, with
model name `glm-5.3-flash`. Qwen remains the recorded default.

The last agent preset ran in
`/home/bmarti44/.cache/glm53-flash/server-20260909-201848`. It received SIGTERM
and stopped at 20:55 EDT on September 9; the sender was not established.
The owner has resumed full million-token qualification. The development port
is reserved for that campaign until its next serving handoff.
Its private API key is in `api-key` in that directory. Chat, correct tool calls,
a tool-result round trip, four overlapping requests, four images and a 16-frame
video passed. [Raw serving evidence](../results/glm53-flash-gates/agent-fast-001/README.md)
is preserved. Media checks used 224x224 fixtures.

This preset uses the existing compressed weights, 65,536 tokens per request,
four slots, a 4 GiB KV reservation and 512-token prompt batches. Its observed
memory low point during the short serving checks was 24.18 GiB with zero cgroup
swap. Maximum context and fidelity remain unqualified; no extra quantization was
introduced. Start it with the following command when it is not already running:

```bash
python3 scripts/47_run_glm53_dev.py --start --preset agent-fast
```

The original experimental configuration remains available:

```bash
python3 scripts/47_run_glm53_dev.py --start
```

That configuration requests four slots of 262,144 tokens each, totaling
1,048,576 tokens. Its direct full-context run failed with a CUDA memory access
error. The smaller candidate does not establish a fix for that failure.

The launcher verifies the pinned weights and uses the prepared runtime. It writes
an owner-only `api-key` file inside the printed output directory. Use that key in
the `Authorization: Bearer ...` header. Set `--output /path/to/a/new/run` to choose
the directory, or `--api-key-file /path/to/existing/api-key` to retain a client key.
Exact settings and logs are saved with each launch. `--prefill-batch` overrides
the preset's prompt batch size; `--text-only` disables media.

Host-control access has been restored. Finish memory-heavy background work
before starting GLM; the launcher requires the other large model to be stopped
and memory to recover first. The launch command stays running while the server
serves requests; startup includes verification of the model files.

This manual development server has a 2.5-hour safety timeout and retains the
shared inference lock, memory watchdog and containment. It does not change
reboot defaults or enable the production switch. Larger media, full context and
paired fidelity remain unqualified. Qualified production performance is **not yet measured**. Short development
timings are retained in the serving evidence, with their limitations.
See [GLM status](../results/glm53-flash-gates/STATUS.md).
