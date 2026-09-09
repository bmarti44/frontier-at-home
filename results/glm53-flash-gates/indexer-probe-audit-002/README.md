Indexer candidate 2 audit
=========================

Campaign 47, candidate 456528a5. All 247 scoped CPU tests pass. The only implementation changes close campaign 46 findings H1 (impossible memory counters) and H2 (BF16 APE rejected by the actual FP32 compressor contract). Tests and genuine failures are preserved in indexer-review-red-001. Seven focused tests now pass, including actual zero-pool upstream CPU execution before any GPU launch. Controller/freezer and all previously reviewed helpers remain byte-identical to candidate 1. No GPU indexer execution has occurred.
