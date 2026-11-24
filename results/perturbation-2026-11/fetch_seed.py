"""Fetch the public seed after the freeze (PREREG "Seed").

Usage: python3 -B fetch_seed.py <out seed.json>

Reads the latest round of drand's default (League of Entropy) chain from one relay, then that round
from two independent relays, and refuses unless both return the same randomness and signature.
The seed string is "drand-<round>-<randomness>". Refuses to overwrite an existing seed file.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

RELAYS = ("https://api.drand.sh", "https://drand.cloudflare.com")


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read())


def main() -> int:
    out = Path(sys.argv[1])
    if out.exists():
        raise SystemExit(f"{out} exists; a seed is drawn once")
    latest = get(f"{RELAYS[0]}/public/latest")
    round_number = int(latest["round"])
    beacons = [get(f"{relay}/public/{round_number}") for relay in RELAYS]
    for beacon in beacons:
        assert int(beacon["round"]) == round_number
        assert beacon["randomness"] == beacons[0]["randomness"], "relays disagree on the randomness"
        assert beacon["signature"] == beacons[0]["signature"], "relays disagree on the signature"
    randomness = beacons[0]["randomness"]
    document = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "chain": "drand default (League of Entropy mainnet)",
        "relays": [f"{relay}/public/{round_number}" for relay in RELAYS],
        "round": round_number,
        "drand_randomness": randomness,
        "drand_signature": beacons[0]["signature"],
        "seed": f"drand-{round_number}-{randomness}",
    }
    out.write_text(json.dumps(document, indent=1) + "\n")
    print(f"round {round_number}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
