# Adversarial falsification review: F11-warm-ttft

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: WEAKENED

1. The 1.764 s observation is real but cherry-picked. In ttft3-a4t0-BAR-MET.txt, warm1 is 3.369 s and warm2 is 1.764 s. The adopted-profile re-gate repeats the pattern: 2.340 s, then 1.770 s. Thus only the third identical request passes; neither first-warm result meets the bar, and no p50 or p95 was measured.

2. The defensible label is “steady-state second exact replay,” not general warm TTFT. Identically replaying a prompt also repeats its MoE routing and maximizes cache locality. Normal agent traffic appends generated text, tool results, and new user content.

3. Real appended-turn evidence breaks the serving interpretation. appended-turn-ttft.txt records HTTP-200 one-token requests at 5.633–5.675 s. The strict/cold controls take 138.8–150.2 s. append-probe-evidence.txt and append-clean-evidence.txt also show resumed outputs differing from cold controls at character zero. The exact cause remained disputed later, but the latency and cold-equivalence failures are not.

4. The store-length arithmetic itself is correct for this fixture. vendor/ds4/ds4_kvstore.c:700 computes stable = tokens - trim and floors it by align:
   5047, trim32, align64 → 4992, suffix55
   5047, trim32, align16 → 5008, suffix39
   5047, trim32, align4 → 5012, suffix35
   5047, trim0, align4 → 5044, suffix3
   However, the align64 raw trace is absent; 4992/5.52 exists only in the ledger summary. The formula also has minimum-token fallbacks and can be bypassed by chat-anchor selection, so it is not an unconditional global rule.

5. max_tokens=1 does not omit first-token work. The harness measures the complete non-streaming one-token response, and ds4_server.c samples and evaluates the token before returning it. The wall time is therefore an upper bound on actual token availability. It is still not a directly timestamped streaming TTFT measurement.

6. The original 1.764 s run has a provenance gap. ttft_probe3.sh neither captures HTTP status nor checks curl success, and the committed BAR-MET file contains no response bodies or hashes. The later re-gate establishes HTTP 200 and b344d80e24a3 text identity, but at 1.770 s, not in the original 1.764 s execution.

7. “Outputs byte-identical” means only choices[0].text for one generated token, hashed to a 12-hex prefix. It does not mean byte-identical HTTP responses or longer continuations. The later appended tests demonstrate that this one-token exact-replay identity cannot establish general resume correctness.

8. trim=0 has a material omitted cost. At align4, cold latency rises from 153.729 s with trim32 to 196.673 s with trim0, about 28%, while LOADPROF count rises from 300 to 450. Appended traffic also writes roughly 920 MiB checkpoints, with observed saves of 0.87–1.23 s plus reload cost.

9. The append divergence is not proven to be caused specifically by trim=0; the repository reports reproduction at align64/trim32 too. What it proves is that the 1.76–1.77 s result is confined to the exact-replay fast path and cannot support a general agent-serving warm-TTFT claim.
