#!/usr/bin/env python3

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "39_bench_toolcall.py"
CASES = ROOT / "evalsets" / "toolcall" / "cases.jsonl"
PINS = CASES.parent / "pins.json"


def load_module():
    spec = importlib.util.spec_from_file_location("bench_toolcall", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def response(tool_calls=None, content=""):
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}


def call(name, arguments):
    return {
        "id": "synthetic-call",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def assert_json_schema(testcase, schema, location="parameters"):
    testcase.assertIsInstance(schema, dict, location)
    testcase.assertEqual(schema.get("type"), "object", location)
    properties = schema.get("properties")
    testcase.assertIsInstance(properties, dict, location)
    required = schema.get("required", [])
    testcase.assertIsInstance(required, list, location)
    testcase.assertTrue(all(isinstance(name, str) and name in properties for name in required))
    allowed = {"string", "integer", "number", "boolean", "array", "object"}
    for name, child in properties.items():
        child_location = f"{location}.properties.{name}"
        testcase.assertIsInstance(child, dict, child_location)
        testcase.assertIn(child.get("type"), allowed, child_location)
        if "enum" in child:
            testcase.assertIsInstance(child["enum"], list, child_location)
            testcase.assertTrue(child["enum"], child_location)
        if child["type"] == "object":
            assert_json_schema(testcase, child, child_location)
        elif child["type"] == "array":
            testcase.assertIsInstance(child.get("items"), dict, child_location)
            testcase.assertIn(child["items"].get("type"), allowed, child_location)


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_module()

    def test_pins_match_cases_and_count(self):
        pins = self.bench.verify_pins(CASES, PINS)
        self.assertEqual(pins["sha256"], hashlib.sha256(CASES.read_bytes()).hexdigest())
        self.assertEqual(pins["case_count"], 20)
        self.assertEqual(pins["created"], "2026-08-20")

    def test_jsonl_parses_and_has_unique_ids(self):
        pins = json.loads(PINS.read_text(encoding="utf-8"))
        cases = self.bench.load_cases(CASES, pins)
        self.assertEqual(len(cases), pins["case_count"])
        self.assertEqual(len({case["id"] for case in cases}), len(cases))

    def test_tools_are_valid_openai_function_schemas(self):
        pins = self.bench.verify_pins(CASES, PINS)
        for case in self.bench.load_cases(CASES, pins):
            for tool in case["tools"]:
                with self.subTest(case=case["id"], tool=tool["function"]["name"]):
                    self.assertEqual(set(tool), {"type", "function"})
                    self.assertEqual(tool["type"], "function")
                    function = tool["function"]
                    self.assertEqual(set(function), {"name", "description", "parameters"})
                    self.assertRegex(function["name"], r"^[A-Za-z_][A-Za-z0-9_]*$")
                    self.assertTrue(function["description"])
                    assert_json_schema(self, function["parameters"])


class ScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_module()
        cls.tool_case = {
            "expect": {
                "type": "tool_call",
                "name": "reserve_room",
                "required_args": {"room": "Cedar", "attendees": 7},
            }
        }
        cls.text_case = {"expect": {"type": "text", "forbidden_call": True}}

    def test_four_synthetic_canned_responses(self):
        canned = (
            (
                "matching subset",
                self.tool_case,
                response([call("reserve_room", '{"room":"Cedar","attendees":7,"floor":2}')]),
                True,
            ),
            (
                "wrong function",
                self.tool_case,
                response([call("cancel_room", '{"room":"Cedar","attendees":7}')]),
                False,
            ),
            (
                "malformed arguments",
                self.tool_case,
                response([call("reserve_room", '{"room":"Cedar"')]),
                False,
            ),
            ("text without calls", self.text_case, response(content="Hello is friendlier."), True),
        )
        for label, case, canned_response, expected in canned:
            with self.subTest(label=label):
                self.assertEqual(
                    self.bench.score_response(case, canned_response)["passed"], expected
                )


if __name__ == "__main__":
    unittest.main()
