# MiMo-V2.6-Flash long-generation breakdown: result

**Label: DIAGNOSTIC.** Nothing here is headline, README or acceptance evidence.

**Preregistered verdict: `INCONCLUSIVE`** (`longgen.py verdict` → `summary.json`).

## What it means

The breakdown reproduced on the qualified launch. With TabbyAPI's output
chunking turned off, every fixture ran clean.

The preregistered rule still says `INCONCLUSIVE`. Arm c (no drafter) answered
one positive control in 810 tokens, so it never reached the break position.
The rule needs both b and c decided before it names a cause, so this result
does not settle whether the drafter takes part.

## Per fixture

`*` marks a positive control that broke in arm a and decides the verdict.

| fixture | a: qualified | b: `--output-chunking false` | c: no DFlash drafter |
|---|---|---|---|
| `count-raw` | clean, exact | clean, exact | clean, exact |
| `count-chat` | clean, exact | clean, exact | clean, exact |
| `holdout-06889` * | **broken at ≈4,138** | clean | inconclusive: stopped at 934 total tokens |
| `holdout-08349` * | **broken at ≈4,196** | clean | clean |
| `essay-chat` (reported only) | **broken at ≈4,191** | clean | clean |

Positions are approximate absolute token counts (prompt plus generated).

- **Arms a and b are both target-greedy.** Their text is identical until
  ≈4,133, ≈4,153 and ≈4,159 tokens on the three fixtures that broke in a. That
  is just past absolute position 4,096, where the preregistration predicted
  the first output-chunking requeue for these prompt lengths.
- **The counting fixtures never broke, in any arm.**
- **The drafter stops agreeing after the break.** In arm a's broken fixtures
  it was rejected about as often as accepted: 4,336 rejected against 3,833
  accepted on 06889. On the clean counting fixtures it was 8 against 5,375.

### Decode rate (diagnostic only)

| | a | b | c |
|---|---|---|---|
| count-raw | 82.8 tok/s | 82.7 | 28.6 |
| essay-chat | 28.1 | 29.2 | 28.6 |

Turning chunking off did not slow decode on these fixtures.

## What the preregistration proposes next

These are proposals to the owner. None was carried out.

The rule's proposal for `INCONCLUSIVE` is the next bounded alternative: vary
the recurrent checkpoint interval and the cache mode, and warm the server
first.

The evidence points most directly at a smaller candidate:
- **Change:** `--output-chunking false` in the qualified profile.
- **Why:** arm b was clean on every fixture.
- **Cost:** it changes the launch, so it needs a test-first change and a new
  acceptance candidate.

Until one of these lands, the qualified `mimo26flash` launch degenerates on
answers that run past about 4,096 tokens. The holdout README row says so.

## Run and safety

`run-longgen.sh` at d62ef1d1 (clean tree) ran one window on 2026-10-14 from
08:18 to 09:08Z, with three fresh starts on :8019 in the fixed order a, b, c.

**Identity**, checked before the first start:
- the runtime sources and the exllamav3 wheel matched;
- every weight and drafter file matched its manifest (0 mismatches).

**Memory and kernel:**
- 2,914 one-second samples; available memory was never below 10.93 GiB
  (lowest at 08:26:23Z, in arm a), and free memory was never below 8.39 GiB;
- swap never moved (SwapFree constant);
- no memwatch breach;
- no kernel lines while the engines ran (`observe/kernel.log` is empty); the
  restore window has only the switch's own `drop_caches`;
- no error lines in any server log.

**Restore:** qwen38fn came back on the first attempt, with exit code 0 at
09:07:44Z.

## Files

- `PREREG.md`, `longgen.py`, `run-longgen.sh`: committed before the run
  (bc7166fb, fa4782b3, 511e4d13).
- `arm-{a,b,c}.json`: each fixture's request, usage and full text.
- `probe-{a,b,c}.json`: the restart probe at each start, recorded only.
  - Arms a and b give the same three greedy hashes as the qualified c3 run
    (`../qualify-2026-10-13-c1/probe-c1.json`).
  - Arm c, without the drafter, differs on two of the three probes.
- `summary.json`: the fixed scorer's verdict.
- `observe/`:
  - `driver.log`, `meminfo.log` (1 Hz);
  - `kernel.log`, `kernel-restore.log`, `restore.log`;
  - per arm: `arm-*/server.log`, `arm-*/memwatch.log`.
