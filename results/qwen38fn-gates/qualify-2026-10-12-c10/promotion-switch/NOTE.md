# Promotion switch window (PREREG plan A3 item 2): clean

Candidate 10 became the serving `qwen38fn` launch on 2026-10-12, after PR #18 merged (`c1c7b389`). The owner had seen the CJK decode diagnostic before promotion.

**Steps** (`window.log`):
1. At 11:22:49Z, `sudo -n scripts/52_engine_switch.sh stop` ran from the production checkout at `3f57ab69`, which still served candidate 9's profile. It returned rc 0 (`stop.log`).
2. The checkout was fast-forwarded to `c1c7b389`, and `verification/MANIFEST.sha256` checked out.
3. At 11:22:52Z, `sudo -n scripts/52_engine_switch.sh qwen38fn` ran and returned rc 0 at 11:34:24Z (`switch.log`). The memory guard saw 114.69 GiB available against the 114 GiB required, with three stable samples.

**Checks:**
- **Kernel:** `journalctl -k -f` ran over the whole window (`kernel.log`) and logged nothing. A direct query for 07:22:40–07:36:00 local time (`kernel-journal-query.log`) returned no entries. So there were no NVRM, Xid or OOM lines; the same capture had recorded unrelated AppArmor lines earlier that day.
- **Memory:** `meminfo.log` samples once a second. External MemAvailable bottomed at 11.03 GiB, and memwatch's low was 11.02 GiB, both at idle after the load. SwapFree went from 16,520,880 to 16,520,944 kB, so swap did not grow.
- **Identity:**
  - The live engine runs runtime-002's interpreter with candidate 10's argv and environment (`live-process.txt`).
  - The server log has the lever identity lines: PLE `MADV_WILLNEED`, draft vocabulary `[0, 65536)` and the NVFP4 draft lm_head (`server-identity-lines.log`). It also shows KV capacity of 1,079,117 tokens, the release of `~/.cache/qwen38-flash-next` (136,225 files), and MemFree 22.44 GiB at the 17.90 GiB arena + 2 GiB.
- **Serving:** the switch's authenticated request through `:8010` answered (`probe.json`).
