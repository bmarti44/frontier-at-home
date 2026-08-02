#!/usr/bin/env python3
"""Source contract for the default-off GLM contiguous expert-slab path."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
ENGINE_PATCHES = (
    ROOT / "results/glm52-gates/harness/ds4-iq2xxs-down-cuda.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-io.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-pinned-staging.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-pinned-on-only.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-accelerated-sha.patch",
    ROOT / "results/glm52-gates/harness/ds4-expert-slab-allocation-telemetry.patch",
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
            "cuda_expert_slab_hash_direct_fd",
            "g_model_direct_fd",
            "g_model_direct_align",
            "model changed during identity hash",
            "full-sidecar identity verified via O_DIRECT",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")
        self.assertNotIn(
            "cuda_expert_slab_hash_model_map(table->model_map",
            self.source,
        )

    def test_one_contiguous_record_read_replaces_three_model_reads(self) -> None:
        for marker in (
            "cuda_expert_slab_read",
            "expert slab model identity mismatch",
            "expert_slab_offset",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_hot_read_uses_accelerated_end_to_end_record_sha(self) -> None:
        """Keep record authentication without the 50 ms scalar SHA path.

        The afdf7dc slab arm measured 2.742 ms mean O_DIRECT time inside a
        58.514 ms mean miss window. Its scalar record SHA ran between those
        observations. The replacement must authenticate the exact pinned bytes
        that will be copied to CUDA, using the system's ARM-accelerated SHA-256.
        Stable file metadata is not equivalent: it cannot detect a post-pread
        staging-buffer mutation.
        """
        for marker in (
            "cuda_expert_slab_sha256",
            "expert slab accelerated SHA-256 unavailable",
            "record checksum mismatch",
            "integrity=startup-sha256+openssl-sha256-per-record",
            'const char *libraries[] = {"libcrypto.so.3", "libcrypto.so"}',
            "dlopen(library, RTLD_NOW | RTLD_LOCAL)",
            "RTLD_NOW | RTLD_LOCAL",
            "dlclose",
        ):
            self.assertIn(marker, self.source)

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

    def test_slab_miss_staging_is_persistent_and_cuda_pinned(self) -> None:
        """O_DIRECT completion must not feed CUDA through pageable buffers.

        The live RED arm measured 2--9 ms slab reads but 50--70 ms fetch
        windows because every layer allocated pageable worker buffers and
        synchronous cudaMemcpy had to stage them.  The default-off slab path
        must instead retain bounded page-locked staging across layer visits.
        Runtime adoption still requires the fixed paired decode lower bound;
        this source contract only prevents the reproduced staging regression.
        """
        for marker in (
            "cuda_expert_slab_staging_ensure",
            "cudaHostAllocPortable",
            "cudaFreeHost",
            "g_expert_slab_staging.buffers",
        ):
            self.assertTrue(marker in self.source, f"missing source marker: {marker}")

    def test_pinned_staging_reports_bounded_geometry_and_cuda_failures(self) -> None:
        """The safety canary must identify exactly what CUDA tried to pin.

        A prior slab arm left an NV_ERR_NO_MEMORY kernel event while the engine
        swallowed cudaHostAlloc's status.  Require a stable success marker with
        pool count, per-buffer bytes, and total bytes, plus the CUDA error name
        and the failed allocation index on failure.  The existing count <= 32
        guard remains the production bound.
        """
        for marker in (
            "expert slab pinned staging ready count=%u buffer_bytes=%llu total_bytes=%llu",
            "expert slab pinned staging allocation failed index=%u count=%u",
            "cudaGetErrorName(err)",
            "count > 32",
        ):
            self.assertIn(marker, self.source)

    def test_pinned_staging_does_not_change_the_slab_off_arm(self) -> None:
        for marker in (
            "std::unique_lock<std::mutex> staging_lock;",
            "if (slab_mode) {",
            "if (!slab_mode &&",
            "if (!slab_mode) free(buf);",
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
