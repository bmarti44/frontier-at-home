Indexer candidate 1 review finding H1
=====================================

Campaign 46, candidate d37e5655. The adversarial reviewer reproduced acceptance of a native return allocation above the same case peak, including 2**63. Post-case allocation/peak/reservation also lack a bound against device capacity. This is a measurement-validity finding: impossible counters could satisfy overlap evidence. No GPU indexer run exists. The proposed acceptance retains the lower overlap bound and requires each return allocation <= case peak <= reserved <= device total. A focused regression changes a return counter from 3,500,000,000 to 3,500,000,001 and 2**63, and supplies peak/reservation above a smaller reported device total.

The same capacity check also applies to profiling peak/reserved counters. Add an observed device-total field to that receipt and reject reservation above it.

H2: the gap reviewer identified BF16 APE input, whereas the actual model parameter and both compressor entry points require FP32. The regression evaluates the probe's exact staging expression on CPU and passes it through the actual upstream zero-pool compressor; its dtype assertion runs before any GPU launch. The fix will change only this input to FP32.
