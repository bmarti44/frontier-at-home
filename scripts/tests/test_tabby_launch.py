#!/usr/bin/env python3
"""Contracts for scripts/lib/tabby_launch.py, the TabbyAPI launcher shim.

TabbyAPI runs behind scripts/40_auth_helper.py, which strips the client's
Bearer token, so TabbyAPI must run with authentication disabled. That makes
every caller an admin, and TabbyAPI always mounts admin routes (model
load/unload, downloads, lora and template switching). The shim therefore
serves only the inference routes the switch and the kit use, derives the
model directories from the profile's absolute artifact paths, and refuses a
launch that would bind beyond loopback, keep runtime model loading on, or list
anything but the one served model.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import tabby_launch  # noqa: E402

BASE = ["--host", "127.0.0.1", "--port", "8013", "--disable-auth", "true",
        "--inline-model-loading", "false", "--backend", "exllamav3",
        "--max-seq-len", "262144"]


def run_asgi(app, scope: dict) -> list:
    """Drive one request and return the events sent to the client."""
    sent: list = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        sent.append(event)

    asyncio.run(app(scope, receive, send))
    return sent


class RecordingApp:
    def __init__(self) -> None:
        self.calls: list = []

    async def __call__(self, scope, receive, send):
        self.calls.append((scope["type"], scope.get("method"), scope.get("path")))
        if scope["type"] == "http":
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})


def http(method: str, path: str) -> dict:
    return {"type": "http", "method": method, "path": path, "headers": []}


class RouteAllowlist(unittest.TestCase):
    def test_inference_and_readiness_routes_pass(self) -> None:
        for method, path in (("GET", "/health"), ("GET", "/v1/models"),
                             ("GET", "/props"), ("POST", "/v1/chat/completions"),
                             ("POST", "/v1/completions")):
            with self.subTest(route=(method, path)):
                app = RecordingApp()
                sent = run_asgi(tabby_launch.RouteAllowlistMiddleware(app),
                                http(method, path))
                self.assertEqual(app.calls, [("http", method, path)])
                self.assertEqual(sent[0]["status"], 200)

    def test_admin_and_unlisted_routes_are_404_and_never_reach_tabby(self) -> None:
        for method, path in (
            ("POST", "/v1/model/load"), ("POST", "/v1/model/unload"),
            ("POST", "/v1/download"), ("GET", "/v1/model/list"),
            ("GET", "/v1/model"), ("POST", "/v1/lora/load"),
            ("POST", "/v1/template/switch"), ("POST", "/v1/sampling/override/switch"),
            ("GET", "/v1/auth/permission"), ("POST", "/v1/token/encode"),
            ("GET", "/docs"), ("GET", "/openapi.json"),
            ("GET", "/v1/chat/completions"), ("DELETE", "/v1/models"),
            ("HEAD", "/health"), ("GET", "/v1/models/"), ("GET", "//v1/models"),
            ("GET", "/v1/../v1/model/list"), ("POST", "/api/v1/generate"),
        ):
            with self.subTest(route=(method, path)):
                app = RecordingApp()
                sent = run_asgi(tabby_launch.RouteAllowlistMiddleware(app),
                                http(method, path))
                self.assertEqual(app.calls, [])
                self.assertEqual(sent[0]["type"], "http.response.start")
                self.assertEqual(sent[0]["status"], 404)
                body = json.loads(sent[1]["body"])
                self.assertEqual(body["error"]["code"], 404)

    def test_websockets_are_closed_and_lifespan_passes(self) -> None:
        app = RecordingApp()
        sent = run_asgi(tabby_launch.RouteAllowlistMiddleware(app),
                        {"type": "websocket", "path": "/v1/models"})
        self.assertEqual(app.calls, [])
        self.assertEqual(sent, [{"type": "websocket.close", "code": 1008}])
        app = RecordingApp()
        run_asgi(tabby_launch.RouteAllowlistMiddleware(app), {"type": "lifespan"})
        self.assertEqual(app.calls, [("lifespan", None, None)])


class LaunchArgv(unittest.TestCase):
    def test_model_and_draft_paths_become_dir_and_name(self) -> None:
        argv = tabby_launch.tabby_argv(BASE + [
            "--model-name", "/models/serve/mimo-pack",
            "--draft-model-name", "/models/draft/mimo-dflash"])
        self.assertEqual(argv[:len(BASE)], BASE)
        tail = argv[len(BASE):]
        self.assertEqual(tail, ["--model-dir", "/models/serve",
                                "--model-name", "mimo-pack",
                                "--draft-model-dir", "/models/draft",
                                "--draft-model-name", "mimo-dflash"])

    def test_draft_model_is_optional(self) -> None:
        argv = tabby_launch.tabby_argv(BASE + ["--model-name", "/m/p/pack"])
        self.assertEqual(argv[len(BASE):],
                         ["--model-dir", "/m/p", "--model-name", "pack"])

    def test_unsafe_or_ambiguous_argv_is_refused(self) -> None:
        model = ["--model-name", "/m/p/pack"]
        cases = {
            "no model": BASE,
            "relative model": BASE + ["--model-name", "pack"],
            "explicit model dir": BASE + model + ["--model-dir", "/m/p"],
            "explicit draft dir": BASE + model + ["--draft-model-dir", "/d"],
            "relative draft": BASE + model + ["--draft-model-name", "d"],
            "repeated model": BASE + model + model,
            "equals form": BASE + ["--model-name=/m/p/pack"],
            "auth on": [a if a != "true" else "false" for a in BASE] + model,
            "public bind": ["--host", "0.0.0.0"] + BASE[2:] + model,
            "inline loading": [a if a != "false" else "true" for a in BASE] + model,
            "no backend": BASE[:8] + BASE[10:] + model,
            "other backend": BASE[:9] + ["exllamav2"] + BASE[10:] + model,
            "dangling flag": BASE + ["--model-name"],
        }
        for label, argv in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(tabby_launch.LaunchError):
                    tabby_launch.tabby_argv(argv)


class ServedDirectory(unittest.TestCase):
    def test_model_parent_must_hold_only_the_served_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "serve"
            (parent / "mimo-pack").mkdir(parents=True)
            tabby_launch.check_model_parent(parent / "mimo-pack")
            (parent / "other").mkdir()
            with self.assertRaises(tabby_launch.LaunchError):
                tabby_launch.check_model_parent(parent / "mimo-pack")

    def test_served_name_must_be_lowercase_and_a_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a" / "Mimo").mkdir(parents=True)
            with self.assertRaises(tabby_launch.LaunchError):
                tabby_launch.check_model_parent(Path(tmp) / "a" / "Mimo")
            (Path(tmp) / "b").mkdir()
            (Path(tmp) / "b" / "pack").write_text("x")
            with self.assertRaises(tabby_launch.LaunchError):
                tabby_launch.check_model_parent(Path(tmp) / "b" / "pack")


if __name__ == "__main__":
    unittest.main()
