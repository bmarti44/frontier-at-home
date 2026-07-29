# DeepSeek-V4-Flash on a single DGX Spark

This repository records a frozen comparison of `entrpi/ds4-on-spark` and upstream
llama.cpp on an NVIDIA GB10 DGX Spark, plus the hardened production service.

## Decision and production status

The frozen ≤28K benchmark selected **ds4**: composite accuracy 86.03% versus 81.62%,
with higher measured speed. That result remains the benchmark record in
[results/DECISION.md](results/DECISION.md).

Brian made a product override in
[results/DECISION-OVERRIDE.md](results/DECISION-OVERRIDE.md): **llama.cpp is the
production engine** because the product roadmap requires contexts approaching 1M tokens,
while ds4 failed warm requests above roughly 28K on this host. ds4 is parked as the
faster small-context alternative; its benchmark evidence is unchanged.

Production traffic follows `Tailscale Serve → Caddy :8010 → authenticated streaming
helper :8014 → llama.cpp :8011`. Every listener is loopback-only on the host, Funnel is
forbidden, the helper strips credentials before the engine, and a watchdog protects the
shared-memory machine from an unrecoverable UMA freeze.

## Reproduce and operate

- [REPRODUCING.md](REPRODUCING.md) gives the pinned host, build, benchmark, audit, and
  `llamacpp` production-install sequence.
- [docs/runbook.md](docs/runbook.md) covers day-2 operation and incidents.
- [PROTOCOL.md](PROTOCOL.md) defines the frozen evaluation versions.
- [docs/threat-model.md](docs/threat-model.md) states what the evidence does and does not
  prove.

## Contributing models and optimizations

Read [AGENTS.md](AGENTS.md) before changing an engine, model profile, benchmark,
or service. It is the working contract for both human and agent contributors.

The short version:

1. Reproduce inherited claims from clean source and independently hashed
   artifacts.
2. Write and commit a production-path test, demonstrate RED, then implement the
   smallest default-off diagnostic arm.
3. Clean-build only after safely unloading the active large model and recovering
   at least 110 GiB available memory.
4. Freeze source/binary/scorer/fixture/configuration hashes, obtain public
   randomness after the freeze, and run equal-fixture contained arms.
5. Use fixed scorers and preserve `manifest.json`, `raw.jsonl`, and
   `summary.json` for every outcome—including failures.
6. For context qualification, test the largest requested context directly.
   Smaller prompts are useful fidelity falsifiers, not context-capability
   evidence.
7. Never load two large models together. Experimental GLM runs use hard cgroup
   limits, disabled swap, continuous memory sampling, and an emergency kill
   floor.
8. Keep the authenticated endpoint and rollback behavior unchanged; DeepSeek
   remains the default until another profile passes all quality, safety,
   direct-1M, switching, and review gates.

The autonomous goal controller is:

```bash
scripts/glm52_goal.py run
scripts/glm52_goal.py resume
scripts/glm52_goal.py status --json
```

The stable operator interface is:

```bash
scripts/52_engine_switch.sh status --json
sudo scripts/52_engine_switch.sh glm52
sudo scripts/52_engine_switch.sh dsv4
```

Avoid routine reboots and repeated interactive privilege requests. Use the
installed delegated controls for exact, identity-verified operations; request
new authority only when no safe in-scope path exists.
