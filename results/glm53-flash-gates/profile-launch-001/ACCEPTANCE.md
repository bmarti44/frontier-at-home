# Actual named-profile launch acceptance

This is a prospective lifecycle check of the existing reviewed implementation.
Run only after context-direct-004 is terminal, its identity-verified model unit
is stopped, all descendants are gone, and at least 110 GiB is stably available.
It does not replace direct context, fidelity, switching or performance gates.
No new serving code or diagnostic flag is introduced by this check.

Use the real operator entry point, without launch overrides:

```bash
scripts/93_profile_serve.sh --profile glm-5.3-flash/cuda-spark-128g-1m-experimental start
```

Capture the command's stdout/stderr and the printed unique output directory.
Retain the resolved profile, artifact verification records, launch configuration,
process identities, systemd invocation/properties, watchdog samples and raw HTTP
observations. Never publish the API key or Authorization header. Bind the source,
profile, model/runtime inventories, prepared cache, configuration and this
acceptance document in the attempt manifest before making acceptance requests.
The readiness request made by the profile launcher is part of request history.

The fixed lifecycle verdict is PASS only when every condition below holds:

1. The actual operator command emits `ready` within its configured startup
   deadline and `status` reports the same named profile as running.
2. Observed native argv/env and resolved configuration agree: four 262,144-token
   slots, aggregate cap 1,048,576, localhost port 8015 and diagnostics disabled.
   These are configuration checks, not evidence of tokens processed.
3. A raw `/health` observation returns 200, unauthenticated `/v1/models` returns
   401, authenticated model listing contains `glm-5.3-flash`, and an independent
   temperature-zero chat request (`Reply with exactly READY`, max_tokens 64,
   same readiness template settings) finishes with `stop` and content `READY`.
4. The fresh owned systemd cgroup has the exact declared containment, including
   MemorySwapMax=0, OOMPolicy=kill and KillMode=control-group. Raw whole-host
   samples never cross the configured 18 GiB kill floor; cgroup swap stays zero.
   No OOM, Xid, crash, timeout, wrong identity or unplanned process is accepted.
5. The actual matching `stop` command succeeds; `status` then reports stopped,
   all recorded model identities are absent and its cgroup is empty or gone.
   At least 110 GiB available memory recovers without reboot.
6. The recorded production/default state hashes, authenticated proxy listener
   identity, and restore/guard service states remain unchanged from preflight.

Preserve every observed failure and the raw record. A failed condition produces
FAIL, and missing observations produce NO_RESULT. A failed lifecycle must be
fixed under the repository's existing convergence rules before another candidate.
Keep the already reviewed source components closed unless execution identifies
a concrete new defect.

The stop check may follow the next separately frozen direct-context request set
on this same named server, avoiding an unnecessary model reload. That context
attempt needs its own acceptance, freeze and post-freeze public randomness;
this lifecycle check does not authorize a context PASS by itself. Any context
crash also fails this launch's clean-lifecycle condition. The launcher readiness
request and independent readiness confirmation must both be recorded in the
context attempt's startup history.
