# GLM-5.3-Flash local bring-up

GLM uses a separate authenticated endpoint at `http://127.0.0.1:8015/v1`.
The model name is `glm-5.3-flash`. Qwen remains the recorded default.

The requested capacity is **four slots of 262,144 tokens each**, totaling
1,048,576 tokens. Configuration and short answers do not establish that all
four slots can process their maximum inputs; that test remains separate.

Start the optional local server from the repository:

```bash
GLM53_RUN="$HOME/.cache/glm53-flash/server-$(date +%Y%m%d-%H%M%S)"
python3 scripts/47_run_glm53_dev.py --start --output "$GLM53_RUN" \
  --skip-mm-profiling --prefill-batch 512 --standard-cuda-allocator \
  --release-warmup-cache --prepared-flashinfer --skip-autotune
```

The launcher verifies the pinned weights, uses the prepared runtime, and writes
its random API key to `$GLM53_RUN/api-key` with owner-only permissions. Use that
key in the `Authorization: Bearer ...` header. The output directory also holds
the exact launch settings and logs. The launcher uses the shared inference lock
and requires the other large model to be stopped and memory to recover first.

This is a manual development server with a 2.5-hour safety timeout. It does not
change reboot defaults or authorize the production switch. The existing memory
watchdog and containment remain active. The model supports the configured text,
tool and media routes; current measured results are in
[GLM status](../results/glm53-flash-gates/STATUS.md).
