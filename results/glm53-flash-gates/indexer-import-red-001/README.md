Indexer import-order regression RED
===================================

At 1a4e6b8f, the new test extracts the actual run_native imports and executes them in a fresh isolated packaged-Python process with CUDA devices hidden. It reproduces the same circular-import traceback as the contained attempt. Seven other focused tests pass. The proposed correction swaps the GLM cache-class import before the sparse-indexer import, matching normal package initialization; numerical functions, fixtures, controller and runtime remain unchanged.
