"""Replay preserved native SSE through the legacy soak client, without HTTP."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('legacy_soak', ROOT / 'scripts/35_soak.py')
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)

def run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    clock = [0.0]
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def __iter__(self):
            for row in rows:
                clock[0] = row['monotonic_ns'] / 1e9
                if row['kind'] == 'chunk':
                    yield b'data: ' + json.dumps(row['chunk']).encode() + b'\n'
                elif row['kind'] == 'done':
                    yield b'data: [DONE]\n'
    with mock.patch.object(soak.urllib.request, 'urlopen', return_value=Response()), mock.patch.object(soak.time, 'monotonic', side_effect=lambda: clock[0]):
        result = soak.Client('http://unused.invalid', None, 1).stream_decode({})
    first = next(row['monotonic_ns'] for row in rows if row['kind'] == 'chunk' and any(choice.get('token_ids') for choice in row['chunk'].get('choices', [])))
    mismatch = result['first_generated_at'] != first / 1e9
    return {'scope': 'CPU replay compatibility only; no live timing or model-performance measurement',
            'source': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'legacy_source': {'sha256': hashlib.sha256((ROOT / 'scripts/35_soak.py').read_bytes()).hexdigest()},
            'native_first_generated_ns': first,
            'legacy_first_generated_seconds_from_replay': result['first_generated_at'],
            'legacy_misses_native_generation_start': mismatch,
            'verdict': 'FAIL' if mismatch else 'PASS'}

if __name__ == '__main__':
    print(json.dumps(run(Path(sys.argv[1])), indent=2))
