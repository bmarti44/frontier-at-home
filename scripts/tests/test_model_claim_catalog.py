#!/usr/bin/env python3
"""Contracts for the public model-integration claim queue."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "models/catalog.json"
README = ROOT / "README.md"
AGENTS = ROOT / "AGENTS.md"
WORKFLOW = ROOT / ".github/workflows/model-claim.yml"


class ModelClaimCatalogTests(unittest.TestCase):
    def test_catalog_matches_the_manual_queue_schema(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(catalog["source"], "manually maintained")
        self.assertRegex(catalog["refreshed_at"], r"^\d{4}-\d{2}-\d{2}$")
        backends_registry = json.loads(
            (ROOT / "configs/backends.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            catalog["claim_backends"], list(backends_registry["backends"])
        )
        expected = {
            "glm-5.2",
            "glm-5.3-flash",
            "kimi-k3",
            "gemma4",
            "qwen3.8-max",
            "qwen3.8-27b",
            "qwen3.8-flash-next",
            "laguna-s-2.1",
            "minimax-m3",
            "nemotron-3-super",
            "nemotron-3-ultra",
            "nemotron-3-nano",
            "kimi-k2.7-code",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "gpt-oss",
            "mistral-large-3",
        }
        models = catalog["models"]
        self.assertEqual({model["slug"] for model in models}, expected)
        self.assertEqual(len(models), len(expected))
        required = {
            "slug",
            "context",
            "parameters",
            "modalities",
            "repo_status",
        }
        # Entries may carry their upstream provenance inline; these keys
        # are permitted but never required.
        optional = {"source", "license"}
        for model in models:
            self.assertEqual(set(model) & required, required)
            self.assertFalse(set(model) - required - optional)
            self.assertIn(
                model["repo_status"],
                {"available", "active", "qualified", "reference_only"},
            )
        statuses = {model["slug"]: model["repo_status"] for model in models}
        self.assertEqual(statuses["deepseek-v4-flash"], "qualified")
        self.assertEqual(statuses["glm-5.2"], "active")
        self.assertEqual(statuses["qwen3.8-27b"], "qualified")
        self.assertEqual(statuses["laguna-s-2.1"], "qualified")

    def test_readme_claim_links_are_catalogued_and_keep_active_models(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        readme = README.read_text(encoding="utf-8")
        models = {model["slug"]: model for model in catalog["models"]}
        displayed = set(re.findall(r"claim%3A([a-z0-9.-]+)", readme))
        self.assertTrue(displayed)
        self.assertTrue(displayed <= set(models))
        required = {
            slug for slug, model in models.items()
            if model["repo_status"] in {"active", "qualified"}
        }
        self.assertTrue(required <= displayed)
        for slug in displayed:
            self.assertGreaterEqual(readme.count(f"claim%3A{slug}"), 2)

    def test_readme_and_agent_guide_explain_model_and_architecture_claims(self):
        readme = README.read_text(encoding="utf-8")
        agents = AGENTS.read_text(encoding="utf-8")
        for document in (readme, agents):
            self.assertIn(
                "claim-model/<catalog-slug>/<backend>",
                document,
            )
            self.assertIn("Architecture claim mapping", document)
        self.assertIn(
            "current trusted default-branch catalog",
            agents,
        )
        for backend in (
            "cuda",
            "apple-silicon",
            "rocm",
            "vulkan",
            "intel-xe",
            "qualcomm",
            "tenstorrent",
            "cpu",
        ):
            self.assertIn(f"`{backend}`", readme)
            self.assertIn(f"`{backend}`", agents)

    def test_claimed_integrations_must_create_a_builtin_agent_goal(self):
        readme = README.read_text(encoding="utf-8")
        agents = AGENTS.read_text(encoding="utf-8")
        claim_template = (
            ROOT / ".github" / "PULL_REQUEST_TEMPLATE" /
            "model-integration-claim.md"
        ).read_text(encoding="utf-8")
        for document in (readme, agents, claim_template):
            self.assertIn("goal tool", document)
        self.assertIn("built-in goal tool", readme)
        self.assertIn("built-in goal tool", claim_template)
        self.assertIn("MUST use the goal tool", agents)
        self.assertIn("persistent goal", agents)
        self.assertNotIn("<model-slug>_<backend>_goal.py", readme)
        self.assertNotIn("<model-slug>_<backend>_goal.py", agents)

    def test_privileged_claim_workflow_never_executes_fork_content(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pull_request_target:", workflow)
        self.assertIn("edited", workflow)
        self.assertIn("labeled", workflow)
        self.assertIn("unlabeled", workflow)
        self.assertIn("models/catalog.json", workflow)
        self.assertIn("push:", workflow)
        self.assertIn("models/catalog.json", workflow)
        self.assertIn("claim-model\\/", workflow)
        self.assertIn("catalog.claim_backends.includes(backend)", workflow)
        self.assertIn("issues.removeLabel", workflow)
        self.assertIn("status:self-declared", workflow)
        cleanup = "await removeManagedLabels(currentPr.number);"
        catalog_read = "const catalog = await readCatalog(defaultBranch);"
        self.assertIn(cleanup, workflow)
        self.assertIn(catalog_read, workflow)
        self.assertLess(workflow.index(cleanup), workflow.index(catalog_read))
        self.assertIn("github.rest.pulls.list", workflow)
        self.assertIn(
            "pr, catalog, pr.number === currentNumber", workflow
        )
        self.assertNotIn("base: defaultBranch,", workflow)
        self.assertNotIn("pull_request.base.sha", workflow)
        self.assertNotIn("actions/checkout", workflow)
        self.assertNotIn("pull_request.head.sha", workflow)
        self.assertNotIn("exec(", workflow)
        self.assertNotIn("\n        run:", workflow)
        self.assertNotIn("pull-requests: write", workflow)
        action_refs = re.findall(r"^\s+uses:\s+(\S+)$", workflow, re.MULTILINE)
        self.assertEqual(
            action_refs,
            [
                "actions/github-script"
                "@ed597411d8f924073f98dfc5c65a23a2325f34cd"
            ],
        )

    def test_catalog_schema_prevents_ambiguous_claim_admission(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        backends = catalog["claim_backends"]
        self.assertEqual(len(backends), len(set(backends)))
        self.assertTrue(backends)
        for backend in backends:
            self.assertRegex(backend, r"^[a-z0-9][a-z0-9-]*$")
        slugs = [model["slug"] for model in catalog["models"]]
        self.assertEqual(len(slugs), len(set(slugs)))
        for slug in slugs:
            self.assertRegex(slug, r"^[a-z0-9][a-z0-9.-]*$")


if __name__ == "__main__":
    unittest.main()
