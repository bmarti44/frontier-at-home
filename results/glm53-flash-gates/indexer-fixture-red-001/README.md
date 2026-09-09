Indexer fixture RED
===================

Tests and fixed contract committed at e591655f before implementation. All four CPU tests fail because glm53_indexer_fixture does not exist. Before implementation the contract was strengthened to capture all valid FP32 logits as well as indices and exact cache/tail bytes. Masked uninitialized logits will not be archived. No GPU indexer execution has occurred. The next candidate will be reviewed separately from frozen convolution gates.
