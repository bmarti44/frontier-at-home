"""Mutation tests for native completion prompt-logprob reduction."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from glm53_prompt_metrics import reduce_prompt_response


def response():
    return {"model": "glm-5.3-flash", "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "choices": [{"index": 0, "finish_reason": "length", "token_ids": [4], "prompt_token_ids": [1, 2, 3],
                         "prompt_logprobs": [None,
                            {"2": {"logprob": -0.25, "rank": 1, "decoded_token": "a"}},
                            {"3": {"logprob": -2.5, "rank": 3, "decoded_token": "b"},
                             "4": {"logprob": -0.2, "rank": 1, "decoded_token": "c"}}]}]}


class PromptMetricTests(unittest.TestCase):
    def reduce(self, value):
        return reduce_prompt_response(value, [1, 2, 3], "glm-5.3-flash", 10)

    def test_native_truth_logprobs_and_top1_reduce(self):
        self.assertEqual(self.reduce(response()), {"tokens": 2, "nll_sum": 2.75, "top1_correct": 1})

    def test_wrong_model_tokens_usage_or_completion_rejects(self):
        changes = [lambda r: r.update(model="wrong"),
                   lambda r: r["choices"][0].update(prompt_token_ids=[1, 2, 4]),
                   lambda r: r["usage"].update(prompt_tokens=2),
                   lambda r: r["usage"].update(total_tokens=5),
                   lambda r: r["choices"][0].update(token_ids=[]),
                   lambda r: r["choices"].append(copy.deepcopy(r["choices"][0])),
                   lambda r: r["choices"][0].update(finish_reason=None)]
        for change in changes:
            value = response()
            change(value)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reduce(value)

    def test_missing_or_misaligned_positions_rejects(self):
        for rows in ([None, None, {}], [None], [{}, {}, {}]):
            value = response()
            value["choices"][0]["prompt_logprobs"] = rows
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.reduce(value)

    def test_nonfinite_positive_logprobs_and_invalid_rank_rejects(self):
        for key, item in (("logprob", float("nan")), ("logprob", float("inf")),
                          ("logprob", float("-inf")), ("logprob", 0.1),
                          ("logprob", True), ("rank", 0), ("rank", True), ("rank", 11)):
            value = response()
            value["choices"][0]["prompt_logprobs"][1]["2"][key] = item
            with self.subTest(key=key, item=item), self.assertRaises(ValueError):
                self.reduce(value)

    def test_missing_truth_missing_top1_ties_and_order_rejects(self):
        changes = [lambda r: r.pop("3"), lambda r: r.pop("4"),
                   lambda r: r["3"].update(rank=1),
                   lambda r: r["3"].update(logprob=-0.1),
                   lambda r: r.update({"04": r.pop("4")}),
                   lambda r: r.update({"10": r.pop("4")})]
        for change in changes:
            value = response()
            change(value["choices"][0]["prompt_logprobs"][2])
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.reduce(value)

    def test_invalid_frozen_token_ids_rejects(self):
        for tokens in ([True, 2, 3], [1, 2, 10], [1], [1, 2, -1]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                reduce_prompt_response(response(), tokens, "glm-5.3-flash", 10)


if __name__ == "__main__":
    unittest.main()
