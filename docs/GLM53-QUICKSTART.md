# Run GLM-5.3-Flash on this Spark

GLM is available through two optional experimental profiles. Qwen remains the
recorded/reboot default. Only one large model can run at a time.

For agent work, start the profile with four 65,536-token slots:

```bash
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-agent-fast start
```

The command stays in the foreground. Wait for the `ready` event before
connecting. Startup checks the pinned model/runtime files, authentication and
a completed reply. The printed output directory contains the owner-only
`api-key` file.

Use these connection settings in your client:

- Base URL: `http://127.0.0.1:8015/v1`
- Model: `glm-5.3-flash`
- Authentication: the current output directory's `api-key`, sent as a Bearer token

From another terminal, check or stop that exact profile:

```bash
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-agent-fast status
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-agent-fast stop
```

The separate full-context profile has four 262,144-token slots. Its total is
1,048,576 tokens across four concurrent conversations; each conversation is
limited to 262,144 tokens. Use its own start/status/stop commands:

```bash
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental start
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental status
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental stop
```

Named profiles supply their exact settings and reject parameter overrides. The
agent profile uses a 4 GiB KV reservation and 512-token prompt batches; the
full-context profile uses a 9,565,304,320-byte reservation. Its current scheduler
candidate uses 512-token batches and a 128-token prompt-chunk cap per conversation;
that candidate awaits qualification.
To list the profiles this host can serve:

```bash
scripts/92_resolve_profile.py list --model glm-5.3-flash
```

Stop an existing model through its normal control command before starting GLM.
Startup requires at least 110 GiB available memory. The experimental server keeps
the shared inference lock, memory watchdog and containment, with a 2.5-hour
safety timeout. These profiles do not activate GLM in the production switch.

The actual named-profile start/authentication/READY/stop lifecycle has passed,
including a clean guard receipt and memory recovery. See [current GLM status](../results/glm53-flash-gates/STATUS.md)
for the latest completed context result and remaining gates. Paired fidelity and
production switching remain pending. The latest 30-minute durability attempt
completed 68 correct replies but failed its fixed request-count requirement and
its prohibition on new host swap. Qualified production performance is not yet measured.
The full-context profile now selects two startup-only memory cleanup flags;
its latest launch and orderly stop completed with no recorded kernel OOM/Xid.
The earlier aggregate million-token result used the preceding startup and
scheduler configuration.
The agent profile's earlier basic chat, tool, four-image and 16-frame video checks
used 224x224 fixtures; they do not establish maximum media or context capability.
