# GLM-5.3-Flash local bring-up

GLM uses a separate authenticated endpoint at `http://127.0.0.1:8015/v1`.
The model name is `glm-5.3-flash`. Qwen remains the recorded default.
The replacement launch directory is
`/home/bmarti44/.cache/glm53-flash/server-bringup-018-restore`; its private key is
in `api-key` within that directory. The preceding session 017 passed authenticated
chat, tool-call, four concurrent-request, single-image, four-image and 16-frame
video checks with 224x224 media inputs. [Raw serving evidence](../results/glm53-flash-gates/server-bringup-017-restore/README.md)
is preserved. Session 017 then stopped cleanly for a small native operator test.
Session 018 uses the same serving settings. Finish memory-heavy background work
before starting GLM.

The requested capacity is **four slots of 262,144 tokens each**, totaling
1,048,576 tokens. The separate full-context test **failed with a CUDA memory
access error**. The restored server supports basic use; this restart does not
resolve that long-context failure.

Start the optional local server from the repository (the launcher now defaults
to the settings that passed the text/tool/media checks):

```bash
python3 scripts/47_run_glm53_dev.py --start
```

The launcher verifies the pinned weights, uses the prepared runtime, and writes
its random API key to `api-key` inside the printed output directory with owner-only permissions. Use that
key in the `Authorization: Bearer ...` header. Use `--output /path/to/a/new/run` to choose that directory. It also holds
the exact launch settings and logs. The launcher uses the shared inference lock
and requires the other large model to be stopped and memory to recover first.

This is a manual development server with a 2.5-hour safety timeout. It does not
change reboot defaults or authorize the production switch. The existing memory
watchdog and containment remain active. This recipe enables text, tools, images and video. Use `--text-only` for the working text fallback. Larger media and full context remain unqualified. Current results are in
[GLM status](../results/glm53-flash-gates/STATUS.md).
