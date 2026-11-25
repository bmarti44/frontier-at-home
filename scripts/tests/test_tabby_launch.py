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
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/lib"))

import tabby_launch  # noqa: E402

BASE = ["--host", "127.0.0.1", "--port", "8013", "--disable-auth", "true",
        "--inline-model-loading", "false", "--backend", "exllamav3",
        "--max-seq-len", "262144", "--disable-fetch-requests", "true"]


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


def tabby_event(obj) -> bytes:
    """One event as TabbyAPI emits it: json.dumps defaults, sse_starlette CRLF framing."""
    data = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
    return ("data: " + data + "\r\n\r\n").encode()


def tabby_chunk(delta: dict, finish_reason=None, **extra) -> dict:
    return {"id": "chatcmpl-r1", "object": "chat.completion.chunk", "created": 1,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            "model": "mimo-v2.6-flash", **extra}


# A stream_options.include_usage stream as endpoints/OAI/utils/chat_completion.py
# writes it: finish_reason is suppressed on content and rides on the usage chunk.
USAGE = {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}
TABBY_USAGE_STREAM = [
    tabby_event(tabby_chunk({"role": "assistant", "reasoning_content": "think"})),
    tabby_event(tabby_chunk({"reasoning_content": " more"})),
    tabby_event(tabby_chunk({"content": "Hello"})),
    tabby_event(tabby_chunk({"content": " world"})),
    tabby_event(tabby_chunk({}, "length", usage=USAGE, timings={"predicted_n": 4})),
    tabby_event("[DONE]"),
]


class SseApp:
    """Sends `events` one body message at a time and logs each send."""

    def __init__(self, events: list, log: list) -> None:
        self.events, self.log = events, log

    async def __call__(self, scope, receive, send):
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"text/event-stream; charset=utf-8")]})
        for index, body in enumerate(self.events):
            self.log.append(("app", index))
            await send({"type": "http.response.body", "body": body, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


def guarded_stream(events: list) -> tuple[bytes, list]:
    log: list = []
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(event):
        if event["type"] == "http.response.body" and event.get("body"):
            log.append(("client", event["body"]))
        sent.append(event)

    app = tabby_launch.guard_app(SseApp(events, log))
    asyncio.run(app(http("POST", "/v1/chat/completions"), receive, send))
    return b"".join(e.get("body", b"") for e in sent if e["type"] == "http.response.body"), log


def sse_data(body: bytes) -> list:
    return [line[5:].strip().decode() for line in body.splitlines() if line.startswith(b"data:")]


def load_bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("bench_speed_tabby", ROOT / "scripts/30_bench_speed.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class StreamResponse:
    status = 200

    def __init__(self, body: bytes) -> None:
        self.lines = body.splitlines(keepends=True)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.lines)


class TabbySse(unittest.TestCase):
    """The shim streams TabbyAPI's SSE events as they arrive, framed the way
    every repository harness and the vLLM profiles do. M1 smoke 2026-10-13: the
    shared vLLM splitter cut only on LF LF, so sse_starlette's CRLF CRLF events
    were held until the response ended (probe stream_seconds 0.0001 s for 22
    events), and the usage chunk carried finish_reason, which the speed bench
    rejects."""

    def test_events_reach_the_client_as_they_are_produced(self) -> None:
        _, log = guarded_stream(TABBY_USAGE_STREAM)
        order = [entry if entry[0] == "app" else ("client",) for entry in log]
        # Every content event is forwarded before the app produces the next one.
        for index in range(4):
            app_at = order.index(("app", index))
            self.assertEqual(order[app_at + 1], ("client",), log)

    def test_usage_chunk_becomes_a_terminal_event_then_a_usage_event(self) -> None:
        body, _ = guarded_stream(TABBY_USAGE_STREAM)
        data = sse_data(body)
        self.assertEqual(data[-1], "[DONE]")
        events = [json.loads(d) for d in data[:-1]]
        self.assertEqual(len(events), 6)
        terminal, usage = events[-2], events[-1]
        self.assertEqual(terminal["choices"], [{"index": 0, "delta": {}, "finish_reason": "length"}])
        self.assertNotIn("usage", terminal)
        self.assertEqual(usage["choices"], [])
        self.assertEqual(usage["usage"], USAGE)
        self.assertEqual(usage["timings"], {"predicted_n": 4})
        self.assertEqual({e["id"] for e in events}, {"chatcmpl-r1"})
        for event in events[:-2]:
            self.assertIsNone(event["choices"][0]["finish_reason"])
        # Unchanged events keep TabbyAPI's bytes and CRLF framing.
        self.assertTrue(body.startswith(b"".join(TABBY_USAGE_STREAM[:4])))

    def test_terminal_content_without_usage_is_split(self) -> None:
        body, _ = guarded_stream([tabby_event(tabby_chunk({"content": "Hi"})),
                                  tabby_event(tabby_chunk({"content": "!"}, "stop")),
                                  tabby_event("[DONE]")])
        events = [json.loads(d) for d in sse_data(body)[:-1]]
        self.assertEqual([e["choices"][0]["delta"] for e in events],
                         [{"content": "Hi"}, {"content": "!"}, {}])
        self.assertEqual([e["choices"][0]["finish_reason"] for e in events], [None, None, "stop"])
        # Compact LF-framed events split the same way; quoted text never does.
        compact = b'data: {"id":"c","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":"stop"}]}\n\n'
        quoted = tabby_event(tabby_chunk({"content": '{"finish_reason": "stop"}'}))
        body, _ = guarded_stream([quoted, compact])
        self.assertTrue(body.startswith(quoted))
        events = [json.loads(d) for d in sse_data(body)]
        self.assertEqual([e["choices"][0]["finish_reason"] for e in events], [None, None, "stop"])
        self.assertEqual(events[2]["choices"][0]["delta"], {})

    def test_split_terminal_fields_go_to_one_half_each(self) -> None:
        # Review low (2026-10-13): the content half kept eos_reason and
        # timings, and both halves kept logprobs, so a streaming-logprobs
        # client would count the last chunk's logprobs twice.
        logprobs = {"content": [{"token": "!", "logprob": -0.1}]}
        last = tabby_chunk({"content": "!"}, "stop", timings={"predicted_n": 2})
        last["choices"][0].update(logprobs=logprobs, eos_reason="stop_string")
        body, _ = guarded_stream([tabby_event(tabby_chunk({"content": "Hi"})),
                                  tabby_event(last), tabby_event("[DONE]")])
        content, terminal = [json.loads(d) for d in sse_data(body)[1:3]]
        self.assertEqual(content["choices"], [{"index": 0, "delta": {"content": "!"},
                                               "finish_reason": None, "logprobs": logprobs}])
        self.assertNotIn("timings", content)
        self.assertEqual(terminal["choices"], [{"index": 0, "delta": {}, "finish_reason": "stop",
                                                "eos_reason": "stop_string"}])
        self.assertEqual(terminal["timings"], {"predicted_n": 2})

    def test_completions_usage_chunk_splits_like_chat(self) -> None:
        # /v1/completions' usage chunk (endpoints/OAI/utils/completion.py)
        # has `text: ""` instead of a delta; it splits the same way.
        usage_chunk = {"id": "cmpl-r1", "object": "text_completion", "created": 1,
                       "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                       "model": "mimo-v2.6-flash", "usage": USAGE}
        body, _ = guarded_stream([tabby_event({"id": "cmpl-r1", "choices": [
            {"index": 0, "text": "a", "finish_reason": None}]}), tabby_event(usage_chunk),
            tabby_event("[DONE]")])
        terminal, usage = [json.loads(d) for d in sse_data(body)[1:3]]
        self.assertEqual(terminal["choices"], usage_chunk["choices"])
        self.assertNotIn("usage", terminal)
        self.assertEqual((usage["choices"], usage["usage"]), ([], USAGE))

    def test_pings_lf_framing_and_partial_events_pass_through(self) -> None:
        ping = b": ping - 2026-10-13 04:43:00\r\n\r\n"
        content = tabby_event(tabby_chunk({"content": "a"}))
        body, _ = guarded_stream([ping, content[:10], content[10:],
                                  b'data: {"choices":[]}\n\n', tabby_event("[DONE]")])
        self.assertEqual(body, ping + content + b'data: {"choices":[]}\n\n' + tabby_event("[DONE]"))

    def test_the_speed_bench_accepts_the_guarded_stream_only(self) -> None:
        bench = load_bench()
        client = bench.Client("http://127.0.0.1:1", None)
        raw = b"".join(TABBY_USAGE_STREAM)
        with unittest.mock.patch.object(bench.urllib.request, "urlopen",
                                        return_value=StreamResponse(raw)):
            with self.assertRaisesRegex(RuntimeError, "usage precedes finish_reason"):
                client.stream_chat({"max_tokens": 4, "ignore_eos": True})
        body, _ = guarded_stream(TABBY_USAGE_STREAM)
        with unittest.mock.patch.object(bench.urllib.request, "urlopen",
                                        return_value=StreamResponse(body)):
            result = client.stream_chat({"max_tokens": 4, "ignore_eos": True})
        self.assertTrue(result["done"])
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(result["usage"], USAGE)
        self.assertEqual(result["generated_reasoning"], "think more")
        self.assertEqual(result["generated_content"], "Hello world")
        self.assertEqual(len(result["token_timestamps_ns"]), 4)

    def test_non_streaming_responses_pass_unchanged(self) -> None:
        app = tabby_launch.guard_app(RecordingApp())
        sent = run_asgi(app, http("GET", "/health"))
        self.assertEqual([e.get("body") for e in sent], [None, b"ok"])
        sent = run_asgi(app, http("GET", "/v1/model/list"))
        self.assertEqual(sent[0]["status"], 404)


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

    def test_output_chunking_is_a_listed_flag(self) -> None:
        # The long-generation diagnostic serves an arm with output chunking
        # off (results/mimo-v26-flash-gates/longgen-2026-10-14); the flag and
        # its value pass through to TabbyAPI unchanged.
        for value in ("false", "true"):
            with self.subTest(value=value):
                argv = tabby_launch.tabby_argv(
                    BASE + ["--output-chunking", value, "--model-name", "/m/p/pack"])
                self.assertEqual(argv[len(BASE):len(BASE) + 2], ["--output-chunking", value])

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
            # argparse expands unambiguous prefixes, so an abbreviation would
            # override a checked flag; only exact, listed flags are accepted.
            "abbreviated host": BASE + model + ["--hos", "0.0.0.0"],
            "abbreviated auth": BASE + model + ["--disable-au", "false"],
            "unlisted flag": BASE + model + ["--api-servers", "kobold"],
            "short flag": BASE + model + ["-h"],
            "equals on another flag": BASE + model + ["--max-seq-len=1"],
            "config file flag": BASE + model + ["--config", "/tmp/x.yml"],
            # Vision is on: the server must never fetch image URLs itself.
            "remote fetches on": [a if a != "--disable-fetch-requests" else "--vision"
                                  for a in BASE] + model,
            "remote fetches off flag false": BASE[:-1] + ["false"] + model,
        }
        for label, argv in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(tabby_launch.LaunchError):
                    tabby_launch.tabby_argv(argv)


class LaunchEnvironment(unittest.TestCase):
    """TabbyAPI merges config.yml from its working directory, TABBY_<SECTION>_
    <FIELD> environment variables and the model folder's tabby_config.yml
    (which always wins) over the command line's view; any of them could undo
    the checked flags, so the shim refuses all three."""

    def tree(self, tmp: str) -> tuple[Path, Path]:
        root = Path(tmp) / "tabbyAPI"
        root.mkdir()
        (root / "main.py").write_text("")
        model = Path(tmp) / "serve" / "pack"
        model.mkdir(parents=True)
        return root, model

    def test_clean_tree_and_environment_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, model = self.tree(tmp)
            tabby_launch.check_environment(
                root, [model], {"TABBY_ROOT": str(root), "PATH": "/bin"})

    def test_config_overrides_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, model = self.tree(tmp)
            draft = Path(tmp) / "draft" / "d"
            draft.mkdir(parents=True)
            env = {"TABBY_ROOT": str(root)}
            for label, path in (("server config", root / "config.yml"),
                                ("model override", model / "tabby_config.yml"),
                                ("draft override", draft / "tabby_config.yml")):
                with self.subTest(case=label):
                    path.write_text("network:\n  host: 0.0.0.0\n")
                    with self.assertRaises(tabby_launch.LaunchError):
                        tabby_launch.check_environment(root, [model, draft], env)
                    path.unlink()
            for name in ("TABBY_NETWORK_HOST", "TABBY_MODEL_MAX_SEQ_LEN",
                         "tabby_network_disable_auth"):
                with self.subTest(env=name):
                    with self.assertRaises(tabby_launch.LaunchError):
                        tabby_launch.check_environment(
                            root, [model], {**env, name: "x"})


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
