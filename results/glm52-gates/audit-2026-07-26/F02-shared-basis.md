# Adversarial falsification review: F02-shared-basis

sol xhigh was tasked to PROVE THE CLAIM FALSE (not merely review it).

---

---FINAL---
VERDICT: FALSIFIED
1. The harness tests the wrong factorization. S = sum_e W_e^T W_e correctly finds the best single common right subspace in W_e ≈ A_e B. Published MoBE instead uses W_e ≈ A_e f(sum_j alpha_ej B_j): multiple matrix-valued bases, expert-specific coefficients and transforms, and a nonlinear activation. It sets r=p, which is 2048 here—not 1024. See results/glm52-gates/harness/shared_basis_probe.py:5-8 versus the [MoBE methodology](https://arxiv.org/html/2508.05257v1).

2. The 90%-at-rank-1024 gate is a strawman. MoBE’s reported compression does not require 90% of all experts’ energy in one rank-1024 subspace. Using the cited artifact’s n=256, d=6144, p=r=2048 and DeepSeek-style m=64 bases, MoBE’s own formula gives:
   retained ratio = 1/3 + 2r/(3d) + 2mr/(3np) = 0.7222,
   i.e. 27.78% expert-parameter reduction despite using full expert rank. That matches MoBE’s reported 24-30% regime. The repository’s own results/glm52-gates/logs/loadprof1/sol-compress-B.txt:21-27 explicitly acknowledges this, then nevertheless applies the unrelated 90% criterion at lines 174-177.

3. “Experts are near-isotropic” is false as stated. Each 2048x6144 expert has rank at most 2048, so its individual W_e^T W_e has at least 4096 zero eigenvalues and cannot be isotropic in 6144-dimensional hidden space. The measurement only says the aggregate row spaces of 32 experts are broadly dispersed. An aggregate can satisfy sum_e W_e^T W_e proportional to I while being represented exactly by several shared basis matrices—for example, three 2048-dimensional bases covering disjoint hidden-space blocks. Thus aggregate isotropy does not kill multi-basis MoBE.

4. There is no concat-versus-sum arithmetic error: for vertical concatenation C=[W_1;...;W_E], C^T C=sum_e W_e^T W_e. The reported population-isotropic baselines r/6144 are also arithmetically correct. But they are not the appropriate finite-sample null. With 32x2048=65,536 rows and d=6144, an iid isotropic Marchenko-Pastur null predicts approximately 3.44%, 6.74%, 13.06%, 24.78%, and 35.51% at the reported ranks—not 2.08%, 4.17%, 8.33%, 16.67%, and 25%. Thus 28.9% at rank 1024 is only about 1.17x the finite-sample null, not the advertised 1.7x. This supports diffuse aggregate covariance, but it says nothing decisive about MoBE’s matrix-basis structure.

5. The sampling does not support a model-wide claim. The code deterministically samples 32 of 256 experts using linspace, from only blk.40.ffn_gate_exps. It tests no other layer and no up projection. The preregistered recommendation in sol-compress-B.txt:160-178 called for early, middle, and late layers and all projection orientations. The ledger silently broadens this one-tensor result into “does not transfer to GLM-5.2.”

6. The objective is incomplete. The probe minimizes raw, unweighted weight-space Frobenius error. Functional error depends on hidden-state covariance, expert routing frequency/weight, the SwiGLU nonlinearity, and interaction with the up projection. Published MoBE also z-score-normalizes layer expert weights before optimization. A whitened or activation/router-weighted probe answers a different—and deployment-relevant—question. Raw 71% residual energy does not establish unacceptable output error or NLL.

7. IQ2_XXS is an unresolved confound for the broad transfer claim. The probe measures dequantized 2-bit weights, whereas published MoBE is learned from original BF16 weights. Approximately isotropic quantization error shifts the Gram spectrum toward r/d and may erase subtler expert-mode structure. No BF16 comparison, quantization-error covariance, or SQNR is committed. Quantization alone probably cannot explain the entire narrow 90%-to-28.87% gap: under an additive isotropic-noise model it would require noise energy about five times signal energy. But it can invalidate conclusions about the different, weaker structure MoBE exploits.

8. The numerical evidence is not independently auditable. results/glm52-gates/logs/loadprof1 contains no probe stdout, eigenvalues, extracted-tensor hash, extraction command, tensor offset, or raw file. The exact 28.87% value does not appear anywhere in the reachable repository; only rounded 28.9% appears in the ledger. The harness also fails to validate raw-file length, tensor offset, dtype, or block alignment. A mis-extracted but correctly sized IQ2 byte range would still dequantize and could look artificially isotropic.

9. There is also a topology inconsistency: the supplied context says 160 routed experts, while the harness and ledger say 256, and results/glm52-gates/logs/g4a/trace-analysis.txt reports 19,200 possible layer/expert pairs across 75 layers, exactly 75x256. This does not rescue F02; it shows the experiment’s target metadata must be reconciled.

10. The only conclusion that survives is narrow: assuming correct extraction, the sampled layer-40 gate weights are a poor fit for one unweighted rank-1024 common right subspace. That can kill that specific aggressive single-basis design. It cannot kill MoBE-class multi-basis factorization or establish that MoBE does not transfer to GLM-5.2.
