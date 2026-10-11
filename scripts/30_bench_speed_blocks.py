#!/usr/bin/env python3
"""Block-aware streaming speed bench for speculative-decoding servers.

scripts/30_bench_speed.py is byte-frozen. This wrapper runs it unchanged (same
fixture, client, tokenizer checks, stream validation and output schema) with
two substitutions, for servers whose SSE events each carry up to
--max-tokens-per-event tokens (vLLM MTP: one accepted token plus every draft):

* early stop counts client-tokenized completion tokens, not events, and a
  stream averaging more than K tokens per event is rejected: something
  coalesced events, so event timestamps no longer time tokens;
* decode_tok_s = (client_completion_tokens - K) / (t_last_event - t_first_event).
  The first event's tokens arrived before t_first and number at most K, so
  subtracting K gives a lower bound.

The frozen event-based value stays in each rep as event_decode_tok_s and never
enters a median. Every other argument is passed to the frozen bench.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

FROZEN = Path(__file__).resolve().parent / "30_bench_speed.py"
DECODE_METHOD = (
    "block-aware: (client_completion_tokens - max_tokens_per_event)"
    " / (t_last_event - t_first_event)"
)


def _load_frozen() -> Any:
    # A private module instance: patching it never reaches other importers.
    spec = importlib.util.spec_from_file_location("_bench_speed_frozen_for_blocks", FROZEN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = _load_frozen()
FROZEN_SHA256 = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
WRAPPER_SHA256 = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
_frozen_run_rep = bench.run_rep
_frozen_write_result = bench.write_result
MAX_TOKENS_PER_EVENT: int | None = None


def observable_output_errors(
    client_completion_tokens: int,
    event_completion_tokens: int,
    minimum_tokens: int,
    field_transitions: int = 0,
) -> list[str]:
    """The frozen check with token-counted early stop and a block-width bound."""
    reasons: list[str] = []
    if client_completion_tokens < minimum_tokens:
        reasons.append(
            f"early stop: {client_completion_tokens} client tokens, "
            f"minimum is {minimum_tokens}"
        )
    ceiling = client_completion_tokens + field_transitions
    if event_completion_tokens > ceiling:
        reasons.append(
            "timestamp/client token mismatch: "
            f"events={event_completion_tokens} exceeds client={client_completion_tokens} "
            f"+ field_transitions={field_transitions}; timestamps cannot outnumber tokens"
        )
    width = MAX_TOKENS_PER_EVENT
    if client_completion_tokens > width * event_completion_tokens + field_transitions:
        reasons.append(
            f"block mismatch: {client_completion_tokens} client tokens in "
            f"{event_completion_tokens} events exceeds {width} tokens per event "
            f"(+ field_transitions={field_transitions})"
        )
    return reasons


def block_decode_tok_s(
    timestamps_ns: list[int], client_completion_tokens: int, max_tokens_per_event: int
) -> float | None:
    if len(timestamps_ns) < 2 or client_completion_tokens <= max_tokens_per_event:
        return None
    elapsed_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
    if elapsed_s <= 0:
        return None
    return (client_completion_tokens - max_tokens_per_event) / elapsed_s


def run_rep(*args: Any, **kwargs: Any) -> dict[str, Any]:
    rep = _frozen_run_rep(*args, **kwargs)
    stamps = rep.get("token_timestamps_ns")
    tokens = rep.get("client_completion_tokens")
    if rep.get("timing_source") != "sse_content_events" or not isinstance(stamps, list) \
            or not isinstance(tokens, int):
        return rep
    rep["event_decode_tok_s"] = rep.get("decode_tok_s")
    rep["decode_tok_s"] = block_decode_tok_s(stamps, tokens, MAX_TOKENS_PER_EVENT)
    rep["max_tokens_per_event"] = MAX_TOKENS_PER_EVENT
    if rep["valid"] and rep["decode_tok_s"] is None:
        rep["valid"] = False
        rep["error"] = "no block-aware decode interval"
    return rep


def write_result(path: Path, result: dict[str, Any]) -> None:
    metadata = result.setdefault("metadata", {})
    metadata["decode_method"] = DECODE_METHOD
    metadata["max_tokens_per_event"] = MAX_TOKENS_PER_EVENT
    metadata["frozen_bench_sha256"] = FROZEN_SHA256
    metadata["wrapper_sha256"] = WRAPPER_SHA256
    _frozen_write_result(path, result)


def install(max_tokens_per_event: int) -> None:
    global MAX_TOKENS_PER_EVENT
    if max_tokens_per_event < 2:
        raise ValueError("max_tokens_per_event must be at least 2")
    MAX_TOKENS_PER_EVENT = max_tokens_per_event
    bench.observable_output_errors = observable_output_errors
    bench.run_rep = run_rep
    bench.write_result = write_result


def main() -> int:
    # allow_abbrev=False: the frozen --max-tokens must not match this flag.
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--max-tokens-per-event", type=int, required=True,
                        help="1 + the server's speculative draft length")
    known, rest = parser.parse_known_args(sys.argv[1:])
    if known.max_tokens_per_event < 2:
        parser.error("--max-tokens-per-event must be at least 2; "
                     "use scripts/30_bench_speed.py for one token per event")
    if any(a == "--token-timing-log" or a.startswith("--token-timing-log=") for a in rest):
        parser.error("--token-timing-log times raw tokens; block-aware decode applies "
                     "to SSE events only")
    install(known.max_tokens_per_event)
    sys.argv = [sys.argv[0], *rest]
    return bench.main()


if __name__ == "__main__":
    raise SystemExit(main())
