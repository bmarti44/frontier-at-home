"""Run the frozen short correctness falsifier; never count it as context capacity."""
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path('/home/bmarti44/spark-deepseek-v4-flash')
spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/48_probe_glm53_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
def verify_inputs(out):
    parent = probe.verify(out.parent)
    assert parent['short_correctness'] == {
        'path': 'short-correctness/manifest.json', 'sha256': probe.sha(out / 'manifest.json')}
    manifest = probe.read(out / 'manifest.json')
    assert set(manifest['files']) == {'request.json', 'fixture.json', 'input-token-ids.json'}
    source = Path(__file__).with_name('prepare_inputs.py').resolve()
    expected = {'path': str(source), 'sha256': probe.sha(source)}
    assert manifest['preparation_source'] == expected == parent['request_configuration']['preparation_source']
    for name, digest in manifest['files'].items():
        assert probe.sha(out / name) == digest
    assert len(probe.read(out / 'input-token-ids.json')) == 4224
    return probe.sha(out.parent / 'manifest.json')


def main(out, server):
    initial_binding = verify_inputs(out)
    body = probe.read(out / 'request.json')
    ids = probe.read(out / 'input-token-ids.json')
    fixture = probe.read(out / 'fixture.json')
    headers = {'Authorization': 'Bearer ' + (server / 'api-key').read_text().strip(),
               'Content-Type': 'application/json'}
    request = urllib.request.Request('http://127.0.0.1:8015/v1/chat/completions',
                                     data=json.dumps(body).encode(), headers=headers)
    with (out / 'raw.jsonl').open('x') as raw:
        def event(**data):
            raw.write(json.dumps({'monotonic_ns': time.monotonic_ns(), 'time_unix': time.time(), **data}) + '\n')
            raw.flush()
        event(kind='start')
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                event(kind='http', status=response.status)
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    value = line[5:].strip()
                    if value == b'[DONE]':
                        event(kind='done')
                        break
                    event(kind='chunk', chunk=probe.decode(value))
        except Exception as error:
            event(kind='error', error=str(error), body=error.read().decode() if hasattr(error, 'read') else None)
        event(kind='end')
    try:
        assert verify_inputs(out) == initial_binding
        content, reasoning, usage, finish, first, terminal, rid, tokens = probe.parse_stream(
            [probe.decode(line) for line in (out / 'raw.jsonl').read_text().splitlines()], ids)
        result = probe.retrieval.validate_completion(content=content, reasoning_content=reasoning,
            finish_reason=finish, done=True, records=fixture['records'], absent_value=fixture['absent_value'])
        summary = {'scope': 'short instruction correctness only; not context capacity',
                   'verdict': 'PASS' if result['pass'] else 'FAIL', 'usage': usage,
                   'output_tokens_observed': len(tokens), 'retrieval': result}
    except Exception as error:
        summary = {'scope': 'short instruction correctness only', 'verdict': 'FAIL', 'error': repr(error)}
    probe.write(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)
    return 0 if summary['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main(*map(Path, sys.argv[1:])))
