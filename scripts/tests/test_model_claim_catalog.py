#!/usr/bin/env python3
"""Contracts for the public model-integration claim queue."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "models/catalog.json"
README = ROOT / "README.md"
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

    def test_readme_exposes_every_claim_and_live_status_link(self):
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        readme = README.read_text(encoding="utf-8")
        for model in catalog["models"]:
            slug = model["slug"]
            self.assertIn(f"https://ollama.com/library/{slug}", readme)
            self.assertIn(f"`{model['ollama_tag']}`", readme)
            if model["repo_status"] == "reference_only":
                self.assertIn(
                    "**Reference only:** not claimable until public local weights",
                    readme,
                )
            else:
                self.assertIn(f"claim%3A{slug}", readme)

    def test_privileged_claim_workflow_never_executes_fork_content(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("pull_request_target:", workflow)
        self.assertIn("context.payload.pull_request.base.sha", workflow)
        self.assertIn("models/catalog.json", workflow)
        self.assertIn("claim-model\\/", workflow)
        self.assertIn("status:claimed", workflow)
        self.assertNotIn("actions/checkout", workflow)
        self.assertNotIn("pull_request.head.sha", workflow)
        self.assertNotIn("exec(", workflow)


if __name__ == "__main__":
    unittest.main()
