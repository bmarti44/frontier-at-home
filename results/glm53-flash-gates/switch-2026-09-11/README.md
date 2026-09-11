# Production switch: glm53-1m in, qwen38-1m back (2026-09-11)

Run from the production checkout at branch head `36a0815a` with the
owner-installed NOPASSWD grant (`sudo -n scripts/52_engine_switch.sh <alias>`);
no new sudo setup. Full transcript: `switch-log.txt`.

| step | result |
|---|---|
| `status --json` before | `qwen38-1m` recorded |
| `sudo -n scripts/52_engine_switch.sh glm53-1m` | exit 0 in 54 s; memory guard 114.4 GiB >= 110; digest checks, health, auth and semantics checks inside the switch passed; `/v1/models` on 8013 lists `glm-5.3-flash` with `max_model_len` 262144; MemAvailable 16.7 GiB idle |
| `status --json` after | `glm53-1m` recorded |
| `sudo -n scripts/52_engine_switch.sh qwen38-1m` | exit 0 in 54 s; memory guard 115.4 GiB; default restored |
| `status --json` after rollback | `qwen38-1m` recorded, MemAvailable 21.6 GiB |
| authentication | unchanged: the auth proxy on 8010 returns 401 without a key for GLM exactly as for Qwen; the engine port 8013 is loopback-only and unauthenticated for both (existing design) |

Switch tests: `scripts/tests/test_engine_switch.py` (+10 glm53 cases: stale
PID, wrong model, startup death, low memory, digest mismatch, rollback),
`test_switch_safety_paths.py`, `test_profile_render_conformance.py`.
