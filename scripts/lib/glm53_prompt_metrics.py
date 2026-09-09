"""Reduce native vLLM completion prompt logprobs into existing quality metrics.

Evidence-only adapter, never imported by serving. Call with strict-decoded JSON
from a frozen request using prompt_logprobs=1, return_token_ids=true, n=1 and
max_tokens=1. The pinned completion response includes native prompt_logprobs,
not the display-clamped echo logprobs. Exact source/runtime/tokenizer/fixture
and response hashes must be bound by the surrounding campaign; this reduction
alone cannot authenticate an engine or authorize a model quality verdict.

Reference: pinned vLLM entrypoints/openai/completion/serving.py, response choice
construction. Reuse scripts/glm52_goal.py:quality_verdict for the 100-case paired
formula after both arms have independently bound, aligned captures.
"""
import math
import re


def _token_ids(value, vocab_size):
    return isinstance(value, list) and all(type(token) is int and 0 <= token < vocab_size for token in value)


def reduce_prompt_response(response, token_ids, model_name, vocab_size):
    if type(vocab_size) is not int or vocab_size <= 1:
        raise ValueError("invalid frozen vocabulary size")
    if not _token_ids(token_ids, vocab_size) or len(token_ids) < 2:
        raise ValueError("invalid frozen prompt token IDs")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("invalid frozen model name")
    if not isinstance(response, dict) or response.get("model") != model_name:
        raise ValueError("native response model mismatch")
    choices, usage = response.get("choices"), response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("expected one native completion choice")
    if not isinstance(usage, dict) or any(type(usage.get(key)) is not int or usage[key] != expected
            for key, expected in (("prompt_tokens", len(token_ids)), ("completion_tokens", 1), ("total_tokens", len(token_ids) + 1))):
        raise ValueError("native token usage mismatch")
    choice = choices[0]
    if type(choice.get("index")) is not int or choice["index"] != 0 or choice.get("finish_reason") not in ("length", "stop"):
        raise ValueError("native completion is missing or unfinished")
    generated = choice.get("token_ids")
    if not _token_ids(generated, vocab_size) or len(generated) != 1:
        raise ValueError("native completion token is missing")
    prompt = choice.get("prompt_token_ids")
    if not _token_ids(prompt, vocab_size) or prompt != token_ids:
        raise ValueError("native prompt token alignment mismatch")
    rows = choice.get("prompt_logprobs")
    if not isinstance(rows, list) or len(rows) != len(token_ids) or rows[0] is not None:
        raise ValueError("native scored-position coverage mismatch")
    nll, correct = [], 0
    for target, row in zip(token_ids[1:], rows[1:], strict=True):
        if not isinstance(row, dict) or len(row) not in (1, 2) or str(target) not in row:
            raise ValueError("native truth/top1 logprobs are missing")
        top = []
        for key, entry in row.items():
            if not isinstance(key, str) or not re.fullmatch(r"0|[1-9][0-9]*", key) or int(key) >= vocab_size:
                raise ValueError("native logprob token ID is invalid")
            if not isinstance(entry, dict):
                raise ValueError("native logprob entry is invalid")
            logprob, rank = entry.get("logprob"), entry.get("rank")
            if type(logprob) not in (int, float) or not math.isfinite(logprob) or logprob > 0:
                raise ValueError("native logprob is nonfinite or invalid")
            if type(rank) is not int or not 1 <= rank <= vocab_size:
                raise ValueError("native token rank is invalid")
            if rank == 1:
                top.append(key)
        if len(top) != 1 or set(row) != {str(target), top[0]}:
            raise ValueError("native top1 is missing or ambiguous")
        truth, best = row[str(target)], row[top[0]]
        if str(target) != top[0] and truth["logprob"] >= best["logprob"]:
            raise ValueError("native logprob order disagrees with rank")
        nll.append(-truth["logprob"])
        correct += top[0] == str(target)
    total = math.fsum(nll)
    if not math.isfinite(total):
        raise ValueError("native NLL aggregate is nonfinite")
    return {"tokens": len(nll), "nll_sum": total, "top1_correct": correct}
