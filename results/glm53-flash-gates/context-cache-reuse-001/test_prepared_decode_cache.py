import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path('/home/bmarti44/spark-deepseek-v4-flash')
spec=importlib.util.spec_from_file_location('launcher',ROOT/'scripts/47_run_glm53_dev.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class PreparedDecodeCacheTests(unittest.TestCase):
 def test_observed_decode_cache_is_copied_and_relocated(self):
  manifest=json.loads((ROOT/'results/glm53-flash-gates/context-direct-004/manifest.json').read_text())
  expected=[r for r in manifest['generated_cache']['files'] if r['path'].startswith('triton/')]
  self.assertEqual(len(expected),8)
  with tempfile.TemporaryDirectory() as directory:
   state=Path(directory)/'state';launcher.reuse_prepared_kernels(state)
   for row in expected:
    path=state/row['path'];self.assertTrue(path.is_file(),'observed decode cache missing: '+row['path'])
    if path.name.startswith('__grp__'):
     for child in json.loads(path.read_text())['child_paths'].values():
      self.assertTrue(Path(child).is_relative_to(state));self.assertTrue(Path(child).is_file())
    else:self.assertEqual(launcher.sha(path),row['sha256'])
if __name__=='__main__':unittest.main()
