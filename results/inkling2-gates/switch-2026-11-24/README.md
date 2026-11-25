# Production switch round trip: qwen38fn → inkling2mini → qwen38fn (2026-11-24/25)

PR #25 qualified Inkling 2 Mini as the switchable alias `inkling2mini`. It is
faster than the default on every speed leg but less accurate, so `qwen38fn`
stays the default. This run took the production switch through the new alias
and back.

Inkling served production for nine hours in between, from 18:31Z to 03:30Z. During
that time the Inkling arm of the perturbation experiment sent it 1,898
requests, all answered without a transport error. That experiment lands
separately, under `results/perturbation-2026-11/`.

How it ran:
- **Checkout:** the production checkout was on `main` at `4cdd13f8` (PR #25 merged).
- **Privileges:** only the owner-installed NOPASSWD grant was used
  (`sudo -n scripts/52_engine_switch.sh <alias>`); no new sudo setup.
- **Driver:** `run-switch-step.sh <step> <alias>`, run once per leg. Switch
  output goes to a file, never a pipe. Full transcript: `switch-log.txt`.
- **`observe/`** holds, per leg:
  - the 1 Hz host memory log (`meminfo-<step>.log`);
  - the kernel log (`kernel-<step>.log`);
  - `nvidia-smi` after the leg.
- **Probes:** `<step>.probe.json` is the switch's own authenticated semantic
  probe response, copied from its state directory.

| Step | Result |
|---|---|
| `status --json` before | `qwen38fn` recorded and serving; MemAvailable 11.38 GiB |
| `sudo -n scripts/52_engine_switch.sh inkling2mini` (18:28:16Z) | **exit 0 in 187 s.** The digest walk ran while `qwen38fn` was still serving (18:28:16–18:28:43Z). `qwen38fn` then stopped (MemAvailable 115.76 GiB at 18:28:48Z). After the cache release, MemFree was 116.32 GiB. The memory guard read 115.885 GiB against the 102 required. The free-memory gate read MemFree 116.25 ≥ 102. |
| After the leg | `inkling2mini` recorded. `/v1/models` on 8013 lists `inkling-2-mini` (`max_model_len` 262144). The unauthenticated `8010/health` returned 401. The probe answered `4.0` (finish `stop`, 51 tokens, `reasoning_content` present). EngineCore held 86,227 MiB. MemAvailable was 25.57 GiB. |
| While Inkling served (02:21:52Z) | Front-door `8010/v1/models` returned 200 with the production key and 401 with a wrong key. |
| `sudo -n scripts/52_engine_switch.sh qwen38fn` (03:30:00Z) | **exit 0 in 695 s** (the 2026-10-11 run took 623 s). After the cache release, MemFree was 116.42 GiB. The memory guard read 116.053 GiB against the 114 required. The free-memory gate read MemFree 116.42 ≥ 114. **The default was restored.** |
| After the leg | `qwen38fn` recorded. `/v1/models` on 8013 lists `qwen3.8-flash-next` (`max_model_len` 262144). The unauthenticated `8010/health` returned 401. The probe answered `**4**` (finish `stop`, 27 tokens, `reasoning_content` present). EngineCore held 99,233 MiB. MemAvailable was 12.18 GiB, and the `qwen38fn` memwatch is armed. At 03:41:50Z the front door returned 200 with the key and 401 with a wrong key. |

## Memory and kernel

- **Kernel:** both legs logged 0 NVRM, Xid, OOM or cgroup-kill lines. Each
  kernel log holds exactly one line, the switch's own `drop_caches: 3`.
- **Leg 1 low:** MemAvailable 11.33 GiB at 18:28:18Z and MemFree 0.96 GiB at
  18:28:25Z. Both came during the digest walk, while `qwen38fn` still served.
- **Leg 2 low:** MemAvailable 12.18 GiB at 03:41:34Z. That was the end of
  the load, as `qwen38fn` settled at its normal serving level. Inkling
  stopped within 2 s of the leg's start (MemAvailable 114.7 GiB at
  03:30:02Z). During the load, MemAvailable stayed between 22.7 and 36.0 GiB
  until the last 20 s, then fell to 13.1 GiB at 03:41:19Z.
- **Swap: 336 KiB was swapped out during leg 1.** SwapFree went from 16,777,212
  to 16,776,876 kB at 18:28:26Z, in the middle of the digest walk. At that
  moment:
  - the switch was hashing Inkling's 20 GB of weights through the page cache;
  - `qwen38fn` was still holding production memory;
  - MemFree had just bottomed out at 0.96 GiB.

  Nothing else was swapped for the rest of the run: SwapFree stayed at
  16,776,876 kB through nine hours of Inkling serving and all of leg 2.

The repo counts swap use as a failure for gate runs. This is recorded as an
**open item against the switch's pre-stop digest walk while the default
serves with about 12 GiB available**, not against the new alias. It is the
same pattern as the 2026-10-11 note in
`../../qwen38fn-gates/switch-2026-10-11/`: low MemFree behind page cache read
during the hash walk. The cache release and the MemFree gate (PR #22) run only
after the old engine stops, so they do not cover the walk itself. Every switch
check passed, and the restore was clean.

## End state

`qwen38fn` is serving as the default on 8013 behind the authenticated 8010
front door. `inkling2mini` stays a qualified switchable engine:

```bash
sudo scripts/52_engine_switch.sh inkling2mini
```
