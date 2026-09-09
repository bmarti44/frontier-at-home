Convolution preparation test baseline
=====================================

Tests committed at 9a1d46c5 before implementation. The unchanged code failed all three tests because scripts/44_probe_glm53_conv.py does not exist. Raw unittest output is retained in convolution-probe-red-001.txt. Before implementation, source review also added a second ragged case with reversed history flags to cover both fresh and continued length-1 state updates. Fixed tolerances and all-byte guard/state checks are unchanged. This is candidate 1 preparation for campaign review round 43; no GPU execution has occurred.
