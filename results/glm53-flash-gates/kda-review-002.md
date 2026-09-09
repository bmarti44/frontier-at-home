# KDA metadata execution correction

Candidate 2 / campaign round 37, implementation `ed9fffd9`, complete audit
`f8d1b560` with 191 passing CPU tests. Both persistent reviewers closed the
metadata defect with zero high or critical findings in the correction.

The actual pinned checkpoint normalizes through Glm5NextConfig to 64 heads,
128 dimensions, -5.0 lower bound and convolution size 4 before CUDA access.
The adversarial reviewer passed four focused tests; the gap reviewer passed
seven. CUDA remained uninitialized. Six analytic/scorer/fixture functions and
the kernel case loop remain AST-identical to failed kda-preflight-001.

The first attempt remains FAIL. A fresh freeze and public seed are required.
This correction does not change the previously recorded staging limitation or
turn JIT preparation into binary/model/context/performance qualification.
