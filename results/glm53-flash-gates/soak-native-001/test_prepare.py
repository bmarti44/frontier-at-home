"""Real pinned-tokenizer regression for the failed zero-seed preparation."""
import hashlib
import importlib.util
import tempfile
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

spec = importlib.util.spec_from_file_location('prepare_target', Path(__file__).with_name('run.py'))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)
seed = '0' * 64
with tempfile.TemporaryDirectory(prefix='glm53-soak-real-prepare-') as directory:
    out = Path(directory) / 'inputs'
    api.prepare(out, seed)
    tokenizer = Tokenizer.from_file(str(api.probe.MODEL / 'tokenizer.json'))
    template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
        extensions=['jinja2.ext.loopcontrols']).from_string((api.probe.MODEL / 'chat_template.jinja').read_text())
    for worker in range(4):
        body = api.read(out / f'{worker}-request.json')
        fixture = api.read(out / f'{worker}-fixture.json')
        ids = api.read(out / f'{worker}-input-token-ids.json')
        text = body['messages'][0]['content']
        assert text.endswith(api.preparation.INSTRUCTION)
        assert body['temperature'] == 0 and body['max_tokens'] == 2048
        assert body['return_token_ids'] is True
        assert body['chat_template_kwargs'] == {'reasoning_effort': 'low', 'clear_thinking': True}
        assert fixture['seed'] == hashlib.sha256(f'{seed}:soak-worker:{worker}'.encode()).hexdigest()
        assert fixture['absent_value'] not in text
        for i, record in enumerate(fixture['records']):
            assert f"AUDIT RECORD {i + 1}: {record['value']}" in text
            assert hashlib.sha256(record['value'].encode()).hexdigest() == record['expected_sha256']
        rendered = template.render(messages=body['messages'], tools=[], add_generation_prompt=True,
            reasoning_effort='low', clear_thinking=True)
        actual = tokenizer.encode(rendered, add_special_tokens=False).ids
        assert actual == ids and len(actual) == 4224
    api.verify(out)
    print('PASS: four real zero-seed requests, exact 4224-token renderings, unchanged instruction/settings and valid markers')
