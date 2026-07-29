#!/usr/bin/env python3
"""Source contract for the bounded CUDA DRAM bandwidth measurement."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts/68_measure_cuda_bandwidth.sh"
SOURCE = ROOT / "scripts/68_cuda_read_bandwidth.cu"


class CudaBandwidthProbeTests(unittest.TestCase):
    def test_wrapper_requires_safe_start_memory_and_no_inference_owner(self):
        text = WRAPPER.read_text(encoding="utf-8")
        self.assertIn("--required-gib 110", text)
        self.assertIn("inference.lock", text)
        self.assertIn("flock -n", text)
        self.assertIn("MemorySwapMax", text)

    def test_probe_is_bounded_and_emits_five_event_samples(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("2ull * 1024ull * 1024ull * 1024ull", text)
        self.assertIn("constexpr int kSamples = 5", text)
        self.assertIn("cudaEventRecord", text)
        self.assertIn("cudaEventSynchronize", text)
        self.assertIn("cudaEventElapsedTime", text)
        self.assertIn('\\"bandwidth_gb_s\\"', text)

    def test_probe_measures_reads_and_consumes_a_checksum(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIn("read_bandwidth_kernel", text)
        self.assertIn("cudaMemcpy", text)
        self.assertIn('\\"checksum\\"', text)
        self.assertNotIn("cudaMemsetAsync", text)


if __name__ == "__main__":
    unittest.main()
