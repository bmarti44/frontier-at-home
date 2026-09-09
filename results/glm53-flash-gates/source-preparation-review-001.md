# Source preparation review

Candidate `77203388`, prepared source directory `build-source-001`. Both
persistent reviewers verified clean source worktrees and matching recorded
HEAD/tree/diff hashes, with no demonstrated critical/high defect in the
source-only checkpoint. Three medium issues receive focused fixes:

1. Resolve source/output paths before `git -C ... worktree add` so relative
   output names refer to the same location as subsequent Python operations.
2. Exclude the recipe's DFlash cache-allocation and unused MTP source changes
   from this no-speculation baseline; require byte-identical originals.
3. Resolve and pin mutable CUTLASS `v4.4.2` and Triton-kernels `v3.5.1` tags to
   commits before compiling, and reject unbound build source overrides.

The plugin build must explicitly bind `EXL3_EXT_INCLUDE` to prepared headers.
The ARM adaptation removes CPU expert/reduction implementations; those paths
must remain unavailable. The initial unchanged plugin is an evidence-only
baseline until repeated environment reads and instrumentation are requalified.

Separately, installed dependency checking found a cuSPARSELt WHEEL tag using
`manylinux2014_sbsa` despite its ELF library being AArch64. Preserve the
failure and resolve provenance explicitly before declaring dependency closure.
