"""Explicit external phase marker; never imported by serving code."""
import argparse
import importlib.util
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('swap_phase_observer', HERE.parent / 'swap-observation-001/observe.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('phase')
    args = parser.parse_args()
    row = {'phase': args.phase, 'counters': api.swap_counters()}
    with args.output.open('x') as stream:
        json.dump(row, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(row), flush=True)
