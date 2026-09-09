# Terminal RSS execution-defect closure

Implementation `034ff2eb`; gate candidate 1 / campaign review round 29.
Both persistent reviewers independently report zero high or critical findings
in this narrow change. Both passed all 23 host tests. The author passed all
158 scoped tests on packaged Python and secret-lint self-tests.

The gap reviewer verified that native006's zero RSS sample occurred 1.166
seconds after terminal acknowledgement and 0.099 seconds before cleanup.
The adversarial reviewer independently checked retained sample coverage and
continued rejection of memory-floor, swap and cgroup-peak faults. Missing
positive RSS, zero RSS at/before terminal acknowledgement, reappearance,
failed exit and surviving groups reject. No other frozen component changed.

Native006 remains FAIL, with original bytes preserved. A new freeze and
public seed are required; this is CPU scorer review, not hardware qualification.
