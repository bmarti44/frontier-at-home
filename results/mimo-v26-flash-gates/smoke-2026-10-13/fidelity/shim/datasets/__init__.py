"""Offline stand-in for ``datasets.load_dataset``, for exllamav3 eval/model_diff.py.

The MiMo runtime has no ``datasets`` package and the fidelity run is offline.
This serves exactly one split, wikitext-2-raw-v1 test, from a row export of
Salesforce/wikitext @ b08601e0 (test-00000-of-00001.parquet, sha256
5f1bea06...6d91, the Hub's LFS digest), and refuses anything else. The row
export's digest and count are pinned in ``fidelity/wiki2-pin.json``.
"""
import hashlib
import json
import os

PIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "wiki2-pin.json")


def load_dataset(path, name=None, split=None, **kwargs):
    if (path, name, split) != ("Salesforce/wikitext", "wikitext-2-raw-v1", "test") or kwargs:
        raise ValueError(f"only wikitext-2-raw-v1 test is pinned, not {path} {name} {split}")
    with open(PIN) as stream:
        pin = json.load(stream)
    with open(os.environ["MIMO_WIKI2_ROWS"], "rb") as stream:
        blob = stream.read()
    if hashlib.sha256(blob).hexdigest() != pin["content_sha256"]:
        raise ValueError("wikitext-2 row export does not match its pinned digest")
    rows = json.loads(blob)
    if len(rows) != pin["rows"]:
        raise ValueError("wikitext-2 row export has the wrong row count")
    return {"text": rows}
