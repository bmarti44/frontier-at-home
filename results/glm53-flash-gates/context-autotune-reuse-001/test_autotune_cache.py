"""Require the native startup autotune receipt to be reused without retuning."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path('/home/bmarti44/spark-deepseek-v4-flash')
spec=importlib.util.spec_from_file_location('launcher',ROOT/'scripts/47_run_glm53_dev.py');launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
class NativeAutotuneCache(unittest.TestCase):
 def test_observed_native_autotune_bytes_are_prepared(self):
  manifest=json.loads((ROOT/'results/glm53-flash-gates/profile-launch-001/manifest.json').read_text())
  name='.cache/exllamav3/autotune/coop_autotune_v1.bin'
  row=next(r for r in manifest['files'] if r['path']=='server/state/'+name)
  with tempfile.TemporaryDirectory() as directory:
   state=Path(directory)/'state';launcher.reuse_prepared_kernels(state)
   self.assertTrue((state/name).is_file(),'native cooperative autotune cache omitted from preparation')
   self.assertEqual((state/name).stat().st_size,row['size_bytes'])
   self.assertEqual(launcher.sha(state/name),row['sha256'])
if __name__=='__main__':unittest.main()
