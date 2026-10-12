# MiMo-V2.6-Flash readiness (plan B0): NO_GO (readiness)

On 2026-10-11 the owner chose to tune Qwen3.8-Flash-Next first and then run a gated readiness check on Xiaomi's MiMo-V2.6-Flash (309B total / 15B active, MIT). The bounded GPU falsifier (B1) runs only if **all three** criteria below hold. They were fixed in the plan before this check. This record covers web research and a local inventory only. No GPU work was done and nothing was downloaded.

**Verdict: `NO_GO (readiness)`.**
- Criterion 2 fails for every quant that fits.
- Criterion 1 fails on its EXL3 path.
- Criterion 3 is only partly met.

There is no reason to spend GPU time yet.

## Criteria (fixed in the plan, 2026-10-11)

1. **A quant that fits.** It must be made with exllamav3 ≥ 1.5.4, or be a llama.cpp GGUF build with MiMo MTP/DFlash. Its weights must be ≤ about 82 GiB, so the measured or modelled peak stays ≤ 106 GiB at ≥ 262K context.
2. **Published fidelity.** Mean KLD ≤ 0.10 against the release, or top-1 agreement ≥ 90%.
3. **A runtime path we can build safely on GB10.** One of:
   - exllamav3 ≥ 1.5.4, through TabbyAPI or vllm-exl3 with MiMo support;
   - a llama.cpp build at `-j2` with production stopped.

## Criterion 1: a quant that fits. Not met on the EXL3 path; met on size only for GGUF

**EXL3 packs that fit by size:**
- [vcruz305/MiMo-V2.6-Flash-RL-EXL3](https://huggingface.co/vcruz305/MiMo-V2.6-Flash-RL-EXL3), 2.20 bpw, 86.94 GB (80.97 GiB), uploaded 2026-09-24 to 09-28. It is the only one.
- It was made with a fork. Its card says upstream does not include the MiMo-V2 architecture and points to the port in [exllamav3#399](https://github.com/turboderp-org/exllamav3/pull/399). It was not made with 1.5.4.

**EXL3 packs that do not fit:** 2.27 bpw ([benthecarman](https://huggingface.co/benthecarman/MiMo-V2.6-Flash-RL-exl3), 85.3 GiB, exllamav3 1.5.2), 2.52 bpw (about 94 GiB) and 3.0 bpw (113 GB).

**The maintainer says all existing EXL3 quants must be redone.** [exllamav3#418](https://github.com/turboderp-org/exllamav3/issues/418), opened 2026-09-28, is still open. turboderp wrote that the model is scale-sensitive, that the quantizer was changed, and: "The existing quants are no good and will have to be redone." No re-made quant has been published. turboderp has published no MiMo quant at all.

**GGUF quants that fit by size:**
- [AesSedai BPW2.0](https://huggingface.co/AesSedai/MiMo-V2.6-Flash-MOPD-GGUF), 72.07 GiB;
- [fresherbz IQ2_XXS](https://huggingface.co/fresherbz/MiMo-V2.6-Flash-MOPD-IQ2_XXS-GGUF), 76.13 GiB, with the [ggml-org](https://huggingface.co/ggml-org/MiMo-V2.6-Flash-MOPD-GGUF) MTP/DFlash files;
- an expert-pruned REAP50 Q2_K, 61.6 GiB.

## Criterion 2: published fidelity. Not met

| Quant | Size | Mean KLD | Top-1 | Reference | Source |
|---|---|---|---|---|---|
| vcruz RL EXL3 2.20 | 80.97 GiB | 0.193 / 0.199 (two sets) | 82.2% / 82.6% | Xiaomi modeling code, original weights, fp32 activations, 10,240 positions per set | model card (checked 2026-10-12) |
| vcruz RL EXL3 2.50 | 91.7 GiB | 0.206 / 0.109 | 83.8% / 87.3% | same | same |
| AesSedai BPW2.0 GGUF | 72.07 GiB | 0.366 (PPL +27.2%) | not reported | the repo's MXFP4 "full quality" row | model card (checked 2026-10-12) |
| AesSedai BPW2.5 / 3.0 / 3.5 | 90.1 / 108.2 / 126.2 GiB | 0.227 / 0.173 / 0.134 | not reported | same | same |
| fresherbz IQ2_XXS | 76.13 GiB | none published | none | — | — |
| REAP50 Q2_K (pruned) | 61.6 GiB | none ("no quality benchmark run") | none | — | — |

Two quants do meet the fidelity bar, but neither fits nor runs here:
- [beamster 2.3](https://huggingface.co/beamster/MiMo-V2.6-Flash-Sushi-2.3bpw): KLD 0.086, top-1 92%; 100.8 GiB; Apple-only runtime.
- [yamz-labs](https://huggingface.co/yamz-labs/MiMo-V2.6-Flash-MOPD-EXL3-Yamz): KLD 0.071, top-1 92%; 97.8 GiB; ROCm-only engine.

**What the numbers say:**
- Every quant at or below 82 GiB with a published measurement is at about double the 0.10 bar or more.
- AesSedai's GGUF ladder stays above 0.10 even at 3.5 bpw and 126 GiB.
- This matches what we saw with GLM-5.3-Flash at about 2 bits: ΔNLL 0.079 against a BF16 teacher (`results/glm53-flash-gates/`).

## Criterion 3: runtime. Partly met

**exllamav3:**
- Releases: 1.5.2 (2026-09-26) added MiMo; 1.5.4 (2026-10-03) lists "Fixes for MiMo-V2.6"; 1.6.0 (10-07); 1.6.1 (10-10).
- The aarch64 build fix (#393) and the integrated-GPU MemAvailable fix (#398) are merged.
- There are no aarch64 wheels, so we would build from source at `-j2`.
- We have 1.4.9 locally (GLM-5.3 runtime-003). No vllm-exl3 MiMo support was found.
- This path can be built, but there is no qualifying pack to load (criteria 1 and 2).

**llama.cpp:**
- Upstream converts V2.6 (#29257, 2026-09-22) and has MiMo DFlash (#29650, first released in b11429 on 10-05); the current build is b11565.
- Our local builds are b10488 (`9d77fa172`, 2026-08-18) and `06f8cebd7`. Both have `LLM_ARCH_MIMO2` with nextn/MTP loading but predate V2.6 support, so we would need a new build.
- Open issue #29931 reports tool-call JSON corruption at long context.

**Single-Spark stability reports** (single-source, both September):
- An exllamav3 1.5.2 run peaked at 113 GiB.
- A 91.9 GiB llama.cpp GGUF froze a Spark during a 32K benchmark.

No single-Spark run on exllamav3 ≥ 1.5.4, or at 262K with a quant at or below 82 GiB, has been reported.

## Recheck when

Any of these would make B0 worth repeating:
- An EXL3 quant made with exllamav3 ≥ 1.5.4 is published at ≤ 82 GiB with mean KLD ≤ 0.10 or top-1 ≥ 90% against the release, from turboderp's re-quantization or anyone else.
- A GGUF or other CUDA-runnable quant at ≤ 82 GiB publishes that fidelity.
- One of the two quants that meet the bar (98–101 GiB) gets a CUDA runtime and a smaller build.

## Method

**Research:**
- Web research ran 2026-10-11 and was rechecked 2026-10-12 by a research subagent. It fetched Hugging Face model cards and file listings, GitHub issues and releases, and forum threads.
- The decisive figures were rechecked directly on 2026-10-12: vcruz's fidelity table, issue #418's status and comment, and AesSedai's KLD ladder.
- All fidelity figures are the quantizers' own measurements. We did not reproduce any of them; doing so would need a download and GPU time.

**Local inventory:**
- exllamav3 1.4.4 and 1.4.9 dist-info under `~/.cache/glm53-flash`;
- llama.cpp sources under `~/.cache/llamacpp-*`;
- 1.6 TB free disk;
- nothing downloaded.
