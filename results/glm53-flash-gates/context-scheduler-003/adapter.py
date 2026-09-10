"""Run unchanged direct-context checks with the reviewed current scheduler validator."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

scheduler = load('direct_current_scheduler', HERE.parent / 'soak-scheduler-002/candidate.py')
probe = scheduler.runner.probe
check_launch = scheduler.check_launch
prepare = load('direct_current_prepare', HERE.parent / 'context-clear-instruction-001/prepare_inputs.py')
short = load('direct_current_short', HERE.parent / 'context-clear-instruction-001/run_short.py')
prepare.probe = short.probe = probe

SOURCES = [Path(__file__).resolve(), HERE / 'PROTOCOL.md', HERE / 'test_adapter.py',
           ROOT / 'scripts/48_probe_glm53_context.py', ROOT / 'scripts/57_dsv4_context_probe.py',
           HERE.parent / 'soak-scheduler-001/candidate.py', HERE.parent / 'soak-scheduler-002/candidate.py',
           HERE.parent / 'soak-native-001/run.py', ROOT / 'scripts/35_soak.py',
           HERE.parent / 'context-clear-instruction-001/prepare_inputs.py',
           HERE.parent / 'context-clear-instruction-001/run_short.py',
           HERE.parent / 'context-native-profile-002/bind_launch.py']

def verify_freeze(root):
    manifest = probe.read(root / 'manifest.json')
    rows = manifest['files']
    scheduler.require(type(rows) is list, 'frozen file list required')
    bindings = {row['path']: row for row in rows}
    scheduler.require(len(bindings) == len(rows), 'duplicate frozen path')
    for path in SOURCES:
        row = bindings.get(str(path))
        scheduler.require(row is not None and path.stat().st_size == row['size_bytes'] and
                          probe.sha(path) == row['sha256'], 'unbound or changed adapter source: ' + str(path))

def bind_launch(out, server):
    # Same binding statements as the closed binder; only the validator is selected above.
    manifest = probe.verify(out)
    planned = probe.read(out / 'server-launch.json');actual = probe.launch_check(server)
    if planned.get('scope') != 'declared fixture configuration only; no running server observation':
        raise ValueError('expected preregistered fixture configuration')
    if planned['arguments'] != actual['arguments']:
        raise ValueError('actual profile arguments differ from frozen fixture configuration')
    for name in ('prelaunch-manifest.json', 'planned-server-launch.json'):
        if (out / name).exists():raise ValueError('launch binding already exists')
    shutil.copyfile(out / 'manifest.json', out / 'prelaunch-manifest.json')
    shutil.copyfile(out / 'server-launch.json', out / 'planned-server-launch.json')
    shutil.copyfile(server / 'launch.json', out / 'server-launch.json')
    manifest['files']['server-launch.json'] = probe.sha(out / 'server-launch.json')
    manifest['launch_binding'] = {'prelaunch_manifest': {'sha256': probe.sha(out / 'prelaunch-manifest.json')},
                                'planned_configuration': {'sha256': probe.sha(out / 'planned-server-launch.json')},
                                'binding_program': {'path': str(Path(__file__).resolve()), 'sha256': probe.sha(Path(__file__))}}
    probe.write(out / 'manifest.json', manifest);probe.verify(out)
    print(json.dumps({'actual_launch_bound': True, 'requests_and_fixtures_unchanged': True}), flush=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'bind', 'short', 'run', 'score'])
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--server', type=Path)
    parser.add_argument('--seed')
    args = parser.parse_args()
    verify_freeze(args.frozen)
    if args.action == 'prepare':prepare.prepare(args.output, args.seed, args.server)
    elif args.action == 'bind':bind_launch(args.output, args.server)
    elif args.action == 'short':
        probe.launch_check(args.server)
        result = short.main(args.output / 'short-correctness', args.server)
        verify_freeze(args.frozen)
        return result
    elif args.action == 'run':
        probe.run(args.output, args.server)
        verify_freeze(args.frozen)
        return 0 if probe.read(args.output / 'summary.json')['verdict'] == 'PASS' else 1
    else:
        result = probe.score(args.output)
        print(json.dumps(result), flush=True)
        verify_freeze(args.frozen)
        return 0 if result['verdict'] == 'PASS' else 1
    verify_freeze(args.frozen)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
