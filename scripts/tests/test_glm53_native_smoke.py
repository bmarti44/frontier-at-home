"""CPU-only acceptance regressions for the native smoke adapter."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

RUNNER = Path(__file__).resolve().parents[1] / "35_smoke_glm53_native.py"
spec = importlib.util.spec_from_file_location("glm53_native_smoke", RUNNER)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class NativeSmokeTests(unittest.TestCase):
    def test_extracted_assertions_survive_optimized_python(self):
        code = '''
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location("smoke", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
namespace = {}
module.load_functions(pathlib.Path(sys.argv[2]), {"must_fail"}, namespace)
try:
    namespace["must_fail"]()
except AssertionError:
    sys.exit(0)
sys.exit("required assertion was removed")
'''
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.py"
            fixture.write_text("def must_fail():\n    assert False, 'required acceptance assertion'\n")
            result = subprocess.run([sys.executable, "-O", "-B", "-c", code, str(RUNNER), str(fixture)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_optimized_runner_rejects_before_output_or_source_access(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "attempt"
            result = subprocess.run([sys.executable, "-O", "-B", str(RUNNER), "--prepared", str(Path(directory) / "missing"),
                                     "--output", str(output), "--seed", "1"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("optimized Python", result.stderr)
            self.assertFalse(output.exists())

    def test_required_check_cannot_skip(self):
        with self.assertRaisesRegex(RuntimeError, "cannot skip"):
            smoke.RequiredChecks.skip("missing fixture")

    def test_missing_extracted_function_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.py"
            fixture.write_text("def unrelated():\n    pass\n")
            with self.assertRaisesRegex(ValueError, "fixture names changed"):
                smoke.load_functions(fixture, {"must_fail"}, {})

    def test_bare_linear_fixture_gets_native_method_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.py"
            fixture.write_text("def construct(method):\n    layer = torch.nn.Module()\n    assert layer.quant_method is method\n    return layer\n")
            namespace = {"torch": SimpleNamespace(nn=SimpleNamespace(Module=SimpleNamespace))}
            smoke.load_functions(fixture, {"construct"}, namespace, linear_fixture_bindings=1)
            method = object()
            self.assertIs(namespace["construct"](method).quant_method, method)

    def test_changed_linear_fixture_shape_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.py"
            fixture.write_text("def construct(method):\n    layer = torch.nn.Module(123)\n    return layer\n")
            with self.assertRaisesRegex(ValueError, "linear fixture setup changed"):
                smoke.load_functions(fixture, {"construct"}, {}, linear_fixture_bindings=1)

    def test_fixture_seed_is_stable_per_case_and_public_seed(self):
        self.assertEqual(smoke.fixture_seed(123, "case_a"), smoke.fixture_seed(123, "case_a"))
        self.assertNotEqual(smoke.fixture_seed(123, "case_a"), smoke.fixture_seed(123, "case_b"))
        self.assertNotEqual(smoke.fixture_seed(123, "case_a"), smoke.fixture_seed(124, "case_a"))
        self.assertGreaterEqual(smoke.fixture_seed(123, "case_a"), 0)
        self.assertLess(smoke.fixture_seed(123, "case_a"), 2**63)


if __name__ == "__main__":
    unittest.main()
