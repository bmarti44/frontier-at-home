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
    def test_catalog_matches_the_refreshed_ollama_cloud_families(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        self.assertEqual(catalog["schema_version"], 1)
        self.assertEqual(
            catalog["source"], "https://ollama.com/search?c=cloud"
        )
        self.assertEqual(catalog["refreshed_at"], "2026-07-29")
        expected = {
            "glm-5.2",
            "kimi-k3",
            "gemma4",
            "qwen3.5",
            "glm-5.1",
            "minimax-m2.7",
            "nemotron-3-super",
            "minimax-m2.5",
            "minimax-m3",
            "kimi-k2.7-code",
            "kimi-k2.6",
            "deepseek-v4-pro",
            "deepseek-v4-flash",
            "nemotron-3-ultra",
            "gpt-oss",
            "gemini-3-flash-preview",
            "nemotron-3-nano",
            "kimi-k2.5",
            "mistral-large-3",
        }
        models = catalog["models"]
        self.assertEqual({model["slug"] for model in models}, expected)
        self.assertEqual(len(models), len(expected))
        for model in models:
            self.assertEqual(
                set(model),
                {
                    "slug",
                    "ollama_tag",
                    "context",
                    "parameters",
                    "modalities",
                    "repo_status",
                },
            )
            self.assertIn(
                model["repo_status"],
                {"available", "active", "qualified", "reference_only"},
            )
        statuses = {model["slug"]: model["repo_status"] for model in models}
        self.assertEqual(statuses["deepseek-v4-flash"], "qualified")
        self.assertEqual(statuses["glm-5.2"], "active")
        self.assertEqual(
            statuses["gemini-3-flash-preview"], "reference_only"
        )

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

    def test_claimed_integrations_must_create_an_autonomous_goal(self):
        readme = README.read_text(encoding="utf-8")
        agents = AGENTS.read_text(encoding="utf-8")
        claim_template = (
            ROOT / ".github" / "PULL_REQUEST_TEMPLATE" /
            "model-integration-claim.md"
        ).read_text(encoding="utf-8")
        goal_contract = ROOT / "docs" / "INTEGRATION_GOALS.md"
        self.assertTrue(goal_contract.is_file())
        for document in (readme, agents, claim_template):
            self.assertIn("docs/INTEGRATION_GOALS.md", document)
            self.assertIn("status --json", document)
        self.assertIn("MUST create the integration goal", agents)
        self.assertIn("PENDING", goal_contract.read_text(encoding="utf-8"))
        self.assertIn("NO_RESULT", goal_contract.read_text(encoding="utf-8"))

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
