"""Evidence-only512/128 adapter; reuse the closed window and terminal scorers."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('glm_second_scheduler_parent', HERE.parent / 'soak-scheduler-001/candidate.py')
previous = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = previous
spec.loader.exec_module(previous)
runner, require = previous.runner, previous.require

def check_launch(launch):
    argv = launch['arguments']
    require(type(argv) is list and all(type(arg) is str for arg in argv), 'malformed candidate argv')
    for flag, value in [('--max-model-len', '262144'), ('--max-num-seqs', '4'),
                        ('--max-num-batched-tokens', '512'), ('--long-prefill-token-threshold', '128')]:
        require([arg for arg in argv if arg.split('=', 1)[0] == flag] == [flag], 'missing or duplicate option: ' + flag)
        index = argv.index(flag)
        require(index + 1 < len(argv) and argv[index + 1] == value, 'wrong candidate value: ' + flag)
    require(argv.count('--no-enable-prefix-caching') == 1 and
            not any(arg.startswith('--enable-prefix-caching') or arg.startswith('--no-enable-prefix-caching=') for arg in argv),
            'prefix cache configuration changed')
    return launch

previous.check_launch = check_launch
runner.probe.check_launch = check_launch
runner.SOURCES += [Path(__file__).resolve(), HERE / 'PROTOCOL.md']
score_first_window = previous.score_first_window
first_window = previous.first_window
run_full = previous.run_full
score = previous.score

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'smoke', 'first-window', 'score-window', 'run', 'score'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed')
    parser.add_argument('--server', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': runner.prepare(args.output, args.seed)
    elif args.action in ['first-window', 'score-window']:
        result = first_window(args.output, args.server) if args.action == 'first-window' else score_first_window(args.output)
        print(json.dumps(result), flush=True)
        sys.exit(0 if result['verdict'] == 'PASS' else 1)
    else:
        if args.action == 'run': result = run_full(args.output, args.server)
        elif args.action == 'score': result = score(args.output)
        else: result = runner.smoke(args.output, args.server)
        sys.exit(0 if result else 1)
