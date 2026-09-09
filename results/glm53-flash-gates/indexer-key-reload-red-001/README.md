Raw-key reload RED and postmortem
================================

At 14c52d87 the regression executes the actual probe reload expression on CPU. It fails exact BF16 bit preservation; eight other focused tests pass. All 256 captured tail mismatches equal numeric conversion of staged uint16 values, independently confirming the diagnosis. The captured 131,070 valid logits match the fixed reference byte-for-byte, but this is postmortem evidence only and cannot replace the failed attempt verdict. The correction is a BF16 view on the existing persistent pinned host staging at the reload copy. No numerical reference or engine change is required.
