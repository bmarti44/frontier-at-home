"""Run the actual nested short-fixture builder with the failed public seed."""
import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
R=Path('/home/bmarti44/spark-deepseek-v4-flash')
SOURCE=R/'results/glm53-flash-gates/context-clear-instruction-001/prepare_inputs.py'
spec=importlib.util.spec_from_file_location('actual_direct_prepare',SOURCE);prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
seed=json.loads(gzip.decompress(Path(__file__).with_name('failed-beacon.json.gz').read_bytes()))['randomness']
class ShortPaddingTests(unittest.TestCase):
 def test_actual_failed_seed_exact_short_rendering_and_markers(self):
  tokenizer=Tokenizer.from_file(str(prep.probe.MODEL/'tokenizer.json'))
  self.assertEqual(prep.probe.sha(prep.probe.MODEL/'tokenizer.json'),prep.probe.read(prep.probe.BINDING)['tokenizer']['sha256'])
  template=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True,extensions=['jinja2.ext.loopcontrols']).from_string((prep.probe.MODEL/'chat_template.jinja').read_text())
  tree=ast.parse(SOURCE.read_text());outer=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare');case=next(n for n in outer.body if isinstance(n,ast.FunctionDef) and n.name=='case')
  namespace={**vars(prep),'tokenizer':tokenizer,'template':template};exec(compile(ast.Module(body=[case],type_ignores=[]),str(SOURCE),'exec'),namespace)
  body,meta,ids=namespace['case'](hashlib.sha256(f'{seed}:short-instruction-smoke'.encode()).hexdigest(),4224)
  rendered=template.render(messages=body['messages'],tools=[],add_generation_prompt=True,reasoning_effort='low',clear_thinking=True)
  actual=tokenizer.encode(rendered,add_special_tokens=False).ids
  self.assertEqual(actual,ids);self.assertEqual(len(actual),4224)
  content=body['messages'][0]['content'];self.assertTrue(content.endswith(prep.INSTRUCTION));self.assertNotIn(meta['absent_value'],content)
  text=content[:-len(prep.INSTRUCTION)];self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),meta['fixture_sha256'])
  for record in meta['records']:self.assertIn(record['value'],text)
  self.assertEqual(body['max_tokens'],2048);self.assertEqual(body['chat_template_kwargs'],{'reasoning_effort':'low','clear_thinking':True})
if __name__=='__main__':unittest.main()
