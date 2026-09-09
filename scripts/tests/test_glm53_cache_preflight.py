"""Mutation controls for the four simultaneously resident cache reservations."""
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "37_probe_glm53_cache.py"
spec = importlib.util.spec_from_file_location("cache_probe", PATH)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Blocks:
    def __init__(self, ids):
        self.ids = ids

    def get_block_ids(self):
        return self.ids


class Manager:
    def __init__(self, defect=None):
        self.block_pool = self
        self.live = {}
        self.defect = defect

    def get_num_free_blocks(self):
        return 144 - 36 * len(self.live)

    def allocate_slots(self, request, *, num_new_tokens, full_sequence_must_fit):
        assert num_new_tokens == 262144 and full_sequence_must_fit
        if len(self.live) == 4:
            return Blocks([[1] * 31, [2], [3], [4], [5], [6]]) if self.defect == "fifth" else None
        start = 1 + 36 * len(self.live)
        ids = [list(range(start, start + 31)), *[[i] for i in range(start + 31, start + 36)]]
        if self.defect == "duplicate":
            ids[0][1] = ids[0][0]
        if self.defect == "out_of_range":
            ids = [[value + 1000 for value in group] for group in ids]
        self.live[request] = Blocks(ids)
        return self.live[request]

    def get_blocks(self, request):
        return self.live[request]

    def free(self, request):
        if self.defect != "leak":
            del self.live[request]


class CachePreflightTests(unittest.TestCase):
    def test_four_remain_live_until_fifth_rejects(self):
        manager = Manager()
        rows = []
        probe.reserve_four(manager, list(range(5)), 123, rows.append)
        self.assertFalse(manager.live)
        self.assertEqual([r["free_blocks"] for r in rows if r["event"] == "reserve"], [108, 72, 36, 0])

    def test_duplicate_physical_blocks_reject(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            probe.reserve_four(Manager("duplicate"), list(range(5)), 1, lambda row: None)

    def test_fifth_admission_reject(self):
        with self.assertRaisesRegex(ValueError, "fifth"):
            probe.reserve_four(Manager("fifth"), list(range(5)), 1, lambda row: None)

    def test_out_of_range_physical_blocks_reject(self):
        with self.assertRaisesRegex(ValueError, "backing"):
            probe.reserve_four(Manager("out_of_range"), list(range(5)), 1, lambda row: None)

    def test_leaked_reservations_reject(self):
        with self.assertRaisesRegex(ValueError, "restore"):
            probe.reserve_four(Manager("leak"), list(range(5)), 1, lambda row: None)


if __name__ == "__main__":
    unittest.main()
