import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run_w7_evict_store_v1.sh"
FROZEN_CGROUP = ROOT / "results/glm52-gates/harness/glm_cgroup_run.sh"
RED_PATCH = ROOT / "results/glm52-gates/harness/0044-test-expose-preload-evict-store-selector-RED.patch"
IMPLEMENTATION_PATCH = ROOT / "results/glm52-gates/harness/0045-feat-diagnose-preload-evict-store-cost.patch"
FLAG = "DS4_KV_SKIP_PRELOAD_EVICT_STORE_DIAGNOSTIC"


class W7EvictStoreContractTests(unittest.TestCase):
    def test_cgroup_forwards_exact_diagnostic_flag(self) -> None:
        source = CGROUP.read_text(encoding="utf-8")
        self.assertEqual(source.count(FLAG), 1)
        self.assertRegex(source, rf"(?m)^  {FLAG} \\\s*$")
        self.assertNotIn(FLAG, FROZEN_CGROUP.read_text(encoding="utf-8"))

    def test_red_patch_is_behavioral_not_missing_symbol(self) -> None:
        source = RED_PATCH.read_text(encoding="utf-8")
        self.assertIn("kv_cache_should_store_before_disk_lookup", source)
        self.assertIn("(void)diagnostic_skip", source)
        self.assertIn("TEST_ASSERT(!kv_cache_should_store_before_disk_lookup", source)

    def test_implementation_is_default_off_and_path_bounded(self) -> None:
        source = IMPLEMENTATION_PATCH.read_text(encoding="utf-8")
        self.assertIn(FLAG, source)
        self.assertIn('!strcmp(value, "1")', source)
        self.assertIn("!diagnostic_skip", source)
        self.assertIn("diagnostic skipped preload evict store", source)
        self.assertIn("kv_cache_try_load(s, slot", source)


if __name__ == "__main__":
    unittest.main()
