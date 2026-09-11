"""Fixture generation and scoring of the media-maximum probe (no server)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "51_probe_media_max.py"
spec = importlib.util.spec_from_file_location("probe_media_max", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class Scoring(unittest.TestCase):
    def test_colours_must_all_appear_in_order(self):
        ok, pos = mod.score_colours("Red, green, blue, yellow.", ["red", "green", "blue", "yellow"])
        self.assertTrue(ok)
        self.assertEqual(pos, sorted(pos))
        self.assertFalse(mod.score_colours("green, red, blue, yellow", ["red", "green", "blue", "yellow"])[0])
        self.assertFalse(mod.score_colours("red, green, blue", ["red", "green", "blue", "yellow"])[0])

    def test_direction_requires_right_and_no_left(self):
        self.assertTrue(mod.score_direction("Right."))
        self.assertTrue(mod.score_direction("The square moves to the right."))
        self.assertFalse(mod.score_direction("Left"))
        self.assertFalse(mod.score_direction("It moves left, not right"))


class Fixtures(unittest.TestCase):
    def test_png_and_mp4_have_requested_geometry(self):
        try:
            from PIL import Image
            import cv2
        except ImportError:
            self.skipTest("PIL/cv2 not importable under this interpreter")
        import io
        png = mod.solid_png((255, 0, 0), 64)
        self.assertEqual(Image.open(io.BytesIO(png)).size, (64, 64))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.mp4"
            blob = mod.moving_square_mp4(5, 64, path)
            self.assertGreater(len(blob), 0)
            cap = cv2.VideoCapture(str(path))
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 5)
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)


if __name__ == "__main__":
    unittest.main()
