"""Prepare exact direct inputs and one short instruction falsifier, before load."""
import hashlib
import importlib.util
import re
import sys
import time
from pathlib import Path

ROOT = Path('/home/bmarti44/spark-deepseek-v4-flash')
spec = importlib.util.spec_from_file_location('probe', ROOT / 'scripts/48_probe_glm53_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
INSTRUCTION = (
    '\n\nReturn exactly one line containing four comma-separated fields. The first '
    'three fields must be the exact identifier values after the colons on AUDIT '
    'RECORD lines labeled 1, 2, 3, in that label order. The fourth field must be the '
    'literal NO_EXTRA_RECORD. Do not include labels, explanations, or other text. '
    'Do not invent any other record.'
)


def prepare(out, seed, server):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    assert re.fullmatch('[0-9a-f]{64}', seed)
    launch = probe.launch_check(server)
    out.mkdir(parents=True, exist_ok=False)
    probe.write(out / 'server-launch.json', launch)
    tokenizer = Tokenizer.from_file(str(probe.MODEL / 'tokenizer.json'))
    assert probe.sha(probe.MODEL / 'tokenizer.json') == probe.read(probe.BINDING)['tokenizer']['sha256']
    template = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=['jinja2.ext.loopcontrols']
    ).from_string((probe.MODEL / 'chat_template.jinja').read_text())

    def case(case_seed, total):
        target = total - 128
        for _ in range(8):
            # The existing fixture builder, insertions and negative control stay intact.
            art = probe.retrieval.build_request_artifacts(tokenizer, target=target, seed_sha256=case_seed)
            fixture = art['fixture']
            body = art['payload']
            body['messages'][0]['content'] = fixture['text'] + INSTRUCTION
            body.update(model='glm-5.3-flash', max_tokens=2048, return_token_ids=True,
                        chat_template_kwargs={'reasoning_effort': 'low', 'clear_thinking': True})
            rendered = template.render(messages=body['messages'], tools=[], add_generation_prompt=True,
                                       reasoning_effort='low', clear_thinking=True)
            ids = tokenizer.encode(rendered, add_special_tokens=False).ids
            if len(ids) == total:
                break
            target += total - len(ids)
        else:
            raise RuntimeError('exact prompt length did not converge')
        assert body['messages'][0]['content'] == fixture['text'] + INSTRUCTION
        assert fixture['absent_value'] not in body['messages'][0]['content']
        meta = {k: v for k, v in fixture.items() if k != 'text'}
        meta.update(seed=case_seed, input_tokens=len(ids))
        return body, meta, ids

    inputs = []
    for slot in range(4):
        case_seed = hashlib.sha256(f'{seed}:slot:{slot}'.encode()).hexdigest()
        body, meta, ids = case(case_seed, 250128)
        meta['slot'] = slot
        for name, value in [('request', body), ('fixture', meta), ('input-token-ids', ids)]:
            probe.write(out / f'{slot}-{name}.json', value)
        inputs.append({'slot': slot, 'input_tokens': len(ids)})
    files = [out / 'server-launch.json'] + [
        out / f'{s}-{name}.json' for s in range(4) for name in ('request', 'fixture', 'input-token-ids')
    ]
    manifest = {
        'scope': 'direct aggregate context with explicit output instruction; unchanged scorer',
        'prepared_unix': time.time(), 'seed': seed, 'inputs': inputs,
        'files': {p.name: probe.sha(p) for p in files},
        'sources': {str(p): probe.sha(p) for p in [
            ROOT / 'scripts/48_probe_glm53_context.py', probe.DS, probe.BINDING,
            probe.MODEL / 'tokenizer.json', probe.MODEL / 'chat_template.jinja', ROOT / 'fixtures/ctx-32k.txt'
        ]},
        'request_configuration': {
            'max_tokens': 2048, 'change': 'explicit output instruction only; unchanged model/template/scorer',
            'preparation_source': {'path': str(Path(__file__).resolve()), 'sha256': probe.sha(Path(__file__))}
        }
    }
    probe.write(out / 'manifest.json', manifest)
    probe.verify(out)
    smoke = out / 'short-correctness'
    smoke.mkdir()
    body, meta, ids = case(hashlib.sha256(f'{seed}:short-instruction-smoke'.encode()).hexdigest(), 4224)
    for name, value in [('request', body), ('fixture', meta), ('input-token-ids', ids)]:
        probe.write(smoke / f'{name}.json', value)
    probe.write(smoke / 'manifest.json', {
        'scope': 'short instruction correctness falsifier; not a context result',
        'seed': seed, 'files': {p.name: probe.sha(p) for p in sorted(smoke.glob('*.json'))},
        'preparation_source': {'path': str(Path(__file__).resolve()), 'sha256': probe.sha(Path(__file__))}
    })
    manifest['short_correctness'] = {'path': 'short-correctness/manifest.json', 'sha256': probe.sha(smoke / 'manifest.json')}
    probe.write(out / 'manifest.json', manifest)
    probe.verify(out)
    print(inputs, flush=True)


if __name__ == '__main__':
    out, seed, server = sys.argv[1:]
    prepare(Path(out), seed, Path(server))
