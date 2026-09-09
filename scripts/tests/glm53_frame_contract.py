#!/usr/bin/env python3
"""Execute the real pinned frame sampler on CPU; no model qualification claim."""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("glm53_audit", ROOT / "scripts/27_audit_glm53_sources.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def suite(source):
    import numpy as np
    path = source / "vllm/transformers_utils/processors/glm5next.py"
    namespace = {"np": np, "math": math, "GLM_VIDEO_DEFAULT_MAX_FRAMES": 2048,
                 "GLM_VIDEO_DEFAULT_FPS": 2.0}
    exec(audit.isolated_functions(path, ["glm_sample_frame_indices"],
         [("Glm5NextVideoProcessor", "sample_frames")]), namespace)

    class FrameContract(unittest.TestCase):
        def test_actual_sampler_obeys_hard_cap_with_config_and_request_overrides(self):
            for total in (1, 3, 15, 16, 17, 18000):
                metadata = SimpleNamespace(total_num_frames=total, fps=30, duration=max(1, total / 30))
                for configured in (16, 2048):
                    receiver = SimpleNamespace(fps_interval=2.0,
                        max_frame_count_dynamic=configured, temporal_patch_size=2)
                    for overrides in ({}, {"num_frames": 16}, {"max_frames": 2048}, {"max_frames": 15}):
                        with self.subTest(total=total, configured=configured, overrides=overrides):
                            frames = namespace["sample_frames"](receiver, metadata, **overrides)
                            self.assertGreater(len(frames), 0)
                            self.assertLessEqual(len(frames), 16)
                            self.assertEqual(len(frames) % 2, 0)
                            self.assertTrue(all(0 <= f < total for f in frames))

        def test_existing_within_cap_sampling_is_identical(self):
            metadata = SimpleNamespace(total_num_frames=1800, fps=30, duration=60)
            receiver = SimpleNamespace(fps_interval=2.0,
                max_frame_count_dynamic=16, temporal_patch_size=2)
            for cap in (2, 4, 8, 16):
                expected = namespace["glm_sample_frame_indices"](1800, 30, 60,
                    target_fps=2.0, max_frame_count=cap, temporal_patch_size=2)
                observed = namespace["sample_frames"](receiver, metadata, max_frames=cap)
                self.assertEqual(observed.tolist(), expected)

    return unittest.defaultTestLoader.loadTestsFromTestCase(FrameContract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(suite(args.source))
    raise SystemExit(0 if result.wasSuccessful() else 1)
