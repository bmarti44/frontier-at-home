# Tail slot-mapping alias audit: NO_RESULT

The installed recipe writes its circular tail mapping in place, while upstream
PR7 uses a builder-owned output buffer. This follow-up checked whether the local
write could corrupt full-attention mapping even with CUDA graphs disabled.

The installed V2 path assigns the tail a separate KV group and a separate row of
the persistent slot-mapping tensor. Attention metadata selects the row by group;
the compressed indexer also owns a separate output buffer. No source-supported
cross-group overwrite was found. This source audit does not establish runtime
alias/lifetime equivalence or reproduce the CUDA failure. No engine change is
justified by this branch; the exact-input synchronized replay remains active.

The manifest binds the inspected files. No GPU request, build, restart, runtime
edit, performance or capability claim was made.
