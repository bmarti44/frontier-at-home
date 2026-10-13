#!/usr/bin/env python3
"""Launch TabbyAPI (exllamav3) for a switch profile.

Profiles run ``<runtime python> -B -m tabby_launch <TabbyAPI CLI flags>`` with
``TABBY_ROOT`` naming the pinned TabbyAPI source tree and this directory on
``PYTHONPATH``. The shim:

* takes ``--model-name`` / ``--draft-model-name`` as absolute artifact paths
  (the switch's ``{model}`` / ``{draft_model}``) and passes TabbyAPI their
  parent directory and basename, because TabbyAPI lists every directory under
  ``model_dir`` as a model;
* refuses a launch that binds beyond loopback, keeps authentication on (the
  auth helper strips the client's Bearer token, so TabbyAPI's own keys could
  never match), keeps runtime model loading on, lets the server fetch remote
  image URLs, or selects another backend;
* accepts only exact, listed flags (argparse would expand an abbreviation such
  as ``--hos`` over a checked flag) and refuses the three sources TabbyAPI
  merges over its command line: ``config.yml`` in its working directory,
  ``TABBY_<SECTION>_<FIELD>`` environment variables, and a model folder's
  ``tabby_config.yml``, which always wins;
* serves only the readiness and inference routes. With authentication off
  every caller is an admin, and TabbyAPI always mounts admin routes (model
  load/unload, downloads, lora, template and sampler switching);
* forwards each SSE event as TabbyAPI produces it, in the shape every
  repository harness expects: content events carry no finish_reason, the
  terminal event has an empty delta, and the usage follows alone with
  ``choices: []`` (``TabbySseMiddleware``);
* runs from ``TABBY_ROOT``, which TabbyAPI needs for its templates, sampler
  presets and logs.

The allowlist is one set lookup per request, outside the token path. The SSE
pass looks at each streamed event once and parses only terminal candidates.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from glm53_runtime_policy import split_terminal_event

ALLOWED_ROUTES = frozenset({
    ("GET", "/health"),
    ("GET", "/v1/models"),
    ("GET", "/props"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
})
# Flags whose value the shim checks; each must appear exactly once.
REQUIRED_VALUES = {
    "--host": "127.0.0.1",
    "--disable-auth": "true",
    "--inline-model-loading": "false",
    "--backend": "exllamav3",
    # Vision is on; the server never fetches image URLs on a client's behalf.
    "--disable-fetch-requests": "true",
}
PATH_FLAGS = (("--model-name", "--model-dir", True),
              ("--draft-model-name", "--draft-model-dir", False))
# Every flag a profile may pass. Each takes exactly one value.
ALLOWED_FLAGS = frozenset({
    *REQUIRED_VALUES, "--port",
    "--model-name", "--draft-model-name",
    "--max-seq-len", "--cache-size", "--max-batch-size", "--cache-mode",
    "--chunk-size", "--vision", "--reasoning", "--tool-format",
    "--draft-mode", "--draft-cache-mode", "--draft-num-tokens",
    "--sysmem-multimodal-cache", "--warmup",
    "--log-prompt", "--log-generation-params", "--log-requests",
    "--log-chat-completion-requests",
})
CONFIG_FILE = "config.yml"
MODEL_OVERRIDE_FILE = "tabby_config.yml"


class LaunchError(ValueError):
    """The profile's TabbyAPI launch is unsafe or ambiguous."""


class RouteAllowlistMiddleware:
    """404 every HTTP route but ALLOWED_ROUTES; close websockets."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind == "http":
            if (scope.get("method"), scope.get("path")) in ALLOWED_ROUTES:
                return await self.app(scope, receive, send)
            body = json.dumps({"error": {"message": "not found",
                                         "type": "invalid_request_error",
                                         "code": 404}}).encode()
            await send({"type": "http.response.start", "status": 404,
                        "headers": [(b"content-type", b"application/json")]})
            await send({"type": "http.response.body", "body": body})
            return None
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return None
        return await self.app(scope, receive, send)


# sse_starlette ends an event with CRLF CRLF; LF LF is the other framing in use.
SSE_BOUNDARY = re.compile(rb"\r\n\r\n|\n\n")


def _encode_event(event: dict) -> bytes:
    return b"data: " + json.dumps(event, separators=(",", ":"), ensure_ascii=False).encode()


def split_usage_event(raw: bytes) -> bytes:
    """Return ``raw`` unchanged, or a terminal event and a usage event.

    With ``stream_options.include_usage`` TabbyAPI withholds finish_reason from
    the content events and sends it on the usage chunk, ``choices: [{delta: {},
    finish_reason}]`` plus ``usage``. OpenAI and vLLM end with the terminal
    event and then the usage alone with ``choices: []``, which the speed bench
    requires. The timings ride with the usage, as llama-server's do.
    """
    if not raw.startswith(b"data:") or b'"usage"' not in raw:
        return raw
    try:
        event = json.loads(raw[5:])
    except ValueError:
        return raw
    if not isinstance(event, dict) or not isinstance(event.get("usage"), dict):
        return raw
    choices = event.get("choices")
    if not (isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            and choices[0].get("finish_reason") is not None and not choices[0].get("delta")):
        return raw
    terminal = {key: value for key, value in event.items() if key not in ("usage", "timings")}
    return _encode_event(terminal) + b"\n\n" + _encode_event({**event, "choices": []})


def harness_events(raw: bytes) -> list:
    """One TabbyAPI SSE event (without its separator) as harness-shaped events."""
    if b'"finish_reason":' not in raw or b'"finish_reason": null' in raw:
        return [raw]
    events = []
    for part in split_terminal_event(raw).split(b"\n\n"):
        events.extend(split_usage_event(part).split(b"\n\n"))
    return events


class TabbySseMiddleware:
    """Forward TabbyAPI's SSE events as they arrive, harness-shaped.

    Events pass byte-for-byte with their own separator unless they are terminal
    (a finish_reason with a delta, or the usage chunk), which split as
    ``harness_events`` describes. Non-SSE responses pass untouched.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        streaming = False
        buffer = bytearray()

        async def wrapped_send(event):
            nonlocal streaming
            if event["type"] == "http.response.start":
                streaming = any(name.lower() == b"content-type"
                                and value.lower().startswith(b"text/event-stream")
                                for name, value in event.get("headers", []))
                return await send(event)
            if event["type"] != "http.response.body" or not streaming:
                return await send(event)
            buffer.extend(event.get("body", b""))
            more = event.get("more_body", False)
            out = bytearray()
            while (match := SSE_BOUNDARY.search(buffer)) is not None:
                separator = match.group()
                for part in harness_events(bytes(buffer[:match.start()])):
                    out += part + separator
                del buffer[:match.end()]
            if not more:
                out += buffer
                buffer.clear()
            if out or not more:
                await send({"type": "http.response.body", "body": bytes(out), "more_body": more})

        return await self.app(scope, receive, wrapped_send)


def guard_app(app):
    """TabbyAPI's ASGI app as the profile serves it."""
    return RouteAllowlistMiddleware(TabbySseMiddleware(app))


def _values(argv: list, flag: str) -> list:
    if any(arg.startswith(flag + "=") for arg in argv):
        raise LaunchError(f"{flag}=... is ambiguous; pass the value separately")
    hits = [i for i, arg in enumerate(argv) if arg == flag]
    if any(i + 1 == len(argv) for i in hits):
        raise LaunchError(f"{flag} has no value")
    return [argv[i + 1] for i in hits]


def _check_flags(argv: list) -> None:
    """Flags and values alternate, and every flag is exact and listed."""
    if len(argv) % 2:
        raise LaunchError("every flag takes exactly one value")
    for flag, value in zip(argv[0::2], argv[1::2]):
        if flag not in ALLOWED_FLAGS:
            raise LaunchError(f"{flag!r} is not a flag the launcher accepts")
        if value.startswith("-"):
            raise LaunchError(f"{flag} has no value")


def tabby_argv(argv: list) -> list:
    """Validate the profile argv and return TabbyAPI's."""
    argv = list(argv)
    for arg in argv:
        if arg.startswith("-") and "=" in arg:
            raise LaunchError(f"{arg.split('=', 1)[0]}=... is ambiguous; "
                              "pass the value separately")
    for flag, wanted in REQUIRED_VALUES.items():
        values = _values(argv, flag)
        if values != [wanted]:
            raise LaunchError(f"{flag} must be given once as {wanted!r}, got {values}")
    out = list(argv)
    for name_flag, dir_flag, required in PATH_FLAGS:
        if _values(argv, dir_flag):
            raise LaunchError(f"{dir_flag} is derived from {name_flag}; do not pass it")
        values = _values(argv, name_flag)
        if not values and not required:
            continue
        if len(values) != 1 or not os.path.isabs(values[0]):
            raise LaunchError(f"{name_flag} must be given once as an absolute path")
        path = Path(values[0])
        index = out.index(name_flag)
        del out[index:index + 2]
        out += [dir_flag, str(path.parent), name_flag, path.name]
    _check_flags(argv)
    return out


def check_environment(root: Path, model_dirs: list, environ) -> None:
    """Refuse every TabbyAPI configuration source besides the command line."""
    if (Path(root) / CONFIG_FILE).exists():
        raise LaunchError(f"{Path(root) / CONFIG_FILE} would override the profile")
    for directory in model_dirs:
        if (Path(directory) / MODEL_OVERRIDE_FILE).exists():
            raise LaunchError(
                f"{Path(directory) / MODEL_OVERRIDE_FILE} would override the profile")
    stray = sorted(key for key in environ
                   if key.upper().startswith("TABBY_") and key != "TABBY_ROOT")
    if stray:
        raise LaunchError(f"TabbyAPI configuration variables are set: {stray}")


def check_model_parent(model: Path) -> None:
    """The served model must be the only entry in its parent directory:
    /v1/models lists them all, and the switch requires exactly one."""
    model = Path(model)
    if model.name != model.name.lower():
        raise LaunchError(f"served model directory must be lowercase: {model.name}")
    if not model.is_dir():
        raise LaunchError(f"served model is not a directory: {model}")
    others = sorted(p.name for p in model.parent.iterdir() if p.name != model.name)
    if others:
        raise LaunchError(f"{model.parent} holds more than the served model: {others}")


def main() -> None:
    root = os.environ.get("TABBY_ROOT")
    if not root or not os.path.isfile(os.path.join(root, "main.py")):
        raise SystemExit("tabby_launch: TABBY_ROOT must name the TabbyAPI source tree")
    try:
        argv = tabby_argv(sys.argv[1:])
        model = Path(_values(sys.argv[1:], "--model-name")[0])
        check_model_parent(model)
        dirs = [model] + [Path(v) for v in _values(sys.argv[1:], "--draft-model-name")]
        check_environment(Path(root), dirs, os.environ)
    except LaunchError as error:
        raise SystemExit(f"tabby_launch: {error}") from None
    os.chdir(root)
    sys.path.insert(0, root)
    sys.argv = [os.path.join(root, "main.py"), *argv]

    import endpoints.server as server

    setup_app = server.setup_app

    def guarded_setup_app(host=None, port=None):
        return guard_app(setup_app(host, port))

    server.setup_app = guarded_setup_app
    import main as tabby_main
    tabby_main.entrypoint()


if __name__ == "__main__":
    main()
