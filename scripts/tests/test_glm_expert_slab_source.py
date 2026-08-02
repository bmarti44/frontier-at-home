#!/usr/bin/env python3
"""Source contract for the default-off GLM contiguous expert-slab path."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATCHES = (
    ROOT / "results/glm52-gates/harness/ds4-iq2xxs-down-cuda.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-io.patch",
)


class ExpertSlabSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = "\n".join(
            patch.read_text(encoding="utf-8") for patch in ENGINE_PATCHES
        )

    def test_slab_path_is_explicit_and_default_off(self) -> None:
        for marker in (
            'getenv("DS4_CUDA_EXPERT_SLAB_PATH")',
            "CUDA contiguous expert slab enabled",
            "CUDA contiguous expert slab disabled",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_explicit_mode_requires_frozen_identity_and_qd(self) -> None:
        for marker in (
            'getenv("DS4_CUDA_EXPERT_SLAB_SHA256")',
            'getenv("DS4_CUDA_EXPERT_SLAB_MODEL_SHA256")',
            "expert slab requires DS4_CUDA_FETCH_THREADS=8..32",
            "frozen full-sidecar identity mismatch",
            "direct full-model identity mismatch",
            "cuda_expert_slab_hash_model_fd",
            "validated/direct descriptor mismatch",
            "O_NOFOLLOW",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_full_model_identity_hash_uses_bounded_direct_io(self) -> None:
        for marker in (
            "cuda_expert_slab_hash_model_fd",
            "g_model_direct_fd",
            "g_model_direct_align",
            "model changed during identity hash",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")
        self.assertNotIn(
            "cuda_expert_slab_hash_model_map(table->model_map",
            self.source,
        )

    def test_one_checksummed_record_read_replaces_three_model_reads(self) -> None:
        for marker in (
            "cuda_expert_slab_read",
            "expert slab record checksum mismatch",
            "expert slab model identity mismatch",
            "expert_slab_offset",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_lifecycle_and_worker_device_are_explicit(self) -> None:
        for marker in (
            "g_expert_slab_init_mu",
            "int fd = -1;",
            "active_readers",
            "cuda_expert_slab_cleanup();",
            "while (g_expert_slab.active_readers.load() != 0)",
            "cudaSetDevice(g_gpu[logical_tier].device_id)",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_slab_mode_is_attested_in_load_profile(self) -> None:
        for marker in (
            "slab_mode=%s",
            "slab_reads=%llu",
            "slab_bytes=%llu",
            "slab_actual_bytes=%llu",
            "slab_peak_qd=%u",
            "SLABIO worker=%d",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_builder_publication_is_bounded_and_non_replacing(self) -> None:
        builder = (ROOT / "results/glm52-gates/harness/glm_expert_slab.py").read_text(
            encoding="utf-8"
        )
        for marker in (
            "MAX_ARTIFACT_BYTES",
            "FREE_SPACE_FLOOR",
            "os.posix_fallocate",
            "tempfile.mkstemp",
            "fcntl.flock",
            "os.link(temporary, output_path",
            "os.fsync(directory_fd)",
            "open_regular(source_path)",
            "MAX_METADATA_DEPTH",
            "MAX_ARRAY_ELEMENTS",
            "MAX_STRING_BYTES",
        ):
            self.assertTrue(marker in builder, f"missing builder marker: {marker}")


if __name__ == "__main__":
    unittest.main()
