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
  never match), keeps runtime model loading on, or selects another backend;
* serves only the readiness and inference routes. With authentication off
  every caller is an admin, and TabbyAPI always mounts admin routes (model
  load/unload, downloads, lora, template and sampler switching);
* splits vLLM-style terminal SSE events like the other engines
  (``glm53_runtime_policy.SplitTerminalSseMiddleware``);
* runs from ``TABBY_ROOT``, which TabbyAPI needs for its templates, sampler
  presets and logs.

The allowlist is one set lookup per request, outside the token path.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

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
}
PATH_FLAGS = (("--model-name", "--model-dir", True),
              ("--draft-model-name", "--draft-model-dir", False))


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


def _values(argv: list, flag: str) -> list:
    if any(arg.startswith(flag + "=") for arg in argv):
        raise LaunchError(f"{flag}=... is ambiguous; pass the value separately")
    hits = [i for i, arg in enumerate(argv) if arg == flag]
    if any(i + 1 == len(argv) for i in hits):
        raise LaunchError(f"{flag} has no value")
    return [argv[i + 1] for i in hits]


def tabby_argv(argv: list) -> list:
    """Validate the profile argv and return TabbyAPI's."""
    argv = list(argv)
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
    return out


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
        check_model_parent(Path(_values(sys.argv[1:], "--model-name")[0]))
    except LaunchError as error:
        raise SystemExit(f"tabby_launch: {error}") from None
    os.chdir(root)
    sys.path.insert(0, root)
    sys.argv = [os.path.join(root, "main.py"), *argv]

    import endpoints.server as server
    from glm53_runtime_policy import SplitTerminalSseMiddleware

    setup_app = server.setup_app

    def guarded_setup_app(host=None, port=None):
        return RouteAllowlistMiddleware(
            SplitTerminalSseMiddleware(setup_app(host, port)))

    server.setup_app = guarded_setup_app
    import main as tabby_main
    tabby_main.entrypoint()


if __name__ == "__main__":
    main()
