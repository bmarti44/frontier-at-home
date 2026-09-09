"""KDA preparation cannot become a binary or context qualification verdict."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('kda_controller_test', Path(__file__).resolve().parents[1] / '39_run_glm53_probe.py')
runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)


class KDARunnerTests(unittest.TestCase):
    def test_kda_preparation_success_stays_no_result(self):
        self.assertEqual(runner.probe_verdict('kda', None), 'NO_RESULT')
        self.assertEqual(runner.probe_verdict('kda', 'failed kernel'), 'FAIL')

    def test_kda_scorer_rejects_missing_or_malformed_rows(self):
        for rows in ([], [{'event': 'configured'}], [{}] * 11):
            with self.assertRaises(ValueError): runner.validate_kda_rows(Path('/nonexistent'), rows, 123)


if __name__ == '__main__': unittest.main()
