"""Fixed acceptance for optional named GLM profiles; no model is loaded."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/lib'))
import profile_resolver as resolver

class ExperimentalProfiles(unittest.TestCase):
    def test_profiles_render_exact_geometry_and_containment(self):
        for name, cap, batch, kv in [('agent-fast', 65536, 512, 4294967296),
                                     ('1m-experimental', 262144, 128, 9565304320)]:
            p = resolver.load_profile('glm-5.3-flash', 'cuda-spark-128g-' + name + '.json')
            host = resolver.load_host(ROOT / 'configs/hosts/spark-aba1.json')
            d = resolver.resolve(p, resolver.load_model('glm-5.3-flash'), host, run_root='/tmp/glm-test')
            self.assertEqual(d['status'], 'estimated')
            self.assertIsNone(d['switch_alias'])
            self.assertEqual(d['context_cap'], cap * 4)
            for flag, value in [('--max-model-len', cap), ('--max-num-seqs', 4),
                                ('--max-num-batched-tokens', batch), ('--kv-cache-memory-bytes', kv)]:
                self.assertEqual(d['argv'][d['argv'].index(flag)+1], str(value))
            self.assertEqual(d['port'], 8015)
            self.assertEqual(d['env']['HOME'], '/tmp/glm-test/state')
            self.assertNotIn('CUDA_LAUNCH_BLOCKING', d['env'])
            self.assertEqual(d['systemd']['properties']['MemoryMax'], '94G')
            self.assertEqual(d['systemd']['properties']['MemorySwapMax'], '0')
            self.assertEqual(d['safety']['minimum_start_gib'], 110)
            self.assertEqual(d['safety']['kill_floor_gib'], 18)
            self.assertGreaterEqual(len(d['digest_checks']), 3)

    def test_profile_entry_rejects_production_and_cli_overrides(self):
        api = importlib.import_module('glm53_profile')
        with self.assertRaisesRegex(ValueError, 'experimental'):
            api.resolve_profile('glm-5.3-flash/cuda-spark-128g-1m', None, Path('/tmp/test'))
        with self.assertRaisesRegex(ValueError, 'override'):
            api.reject_overrides(['--profile', 'test', '--prefill-batch', '2048'])

    def test_frozen_inventory_rejects_changed_or_unlisted_files(self):
        api = importlib.import_module('glm53_profile')
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'runtime'; root.mkdir()
            (root/'a').write_bytes(b'a')
            manifest = Path(tmp)/'inventory.json'
            manifest.write_text(json.dumps({'files':[{'path':'a','size_bytes':1,'sha256':hashlib.sha256(b'a').hexdigest(),'identity':[]}]}))
            api.verify_frozen_tree(root, manifest)
            (root/'a').write_bytes(b'b')
            with self.assertRaises(ValueError): api.verify_frozen_tree(root, manifest)
            (root/'a').write_bytes(b'a'); (root/'extra').write_text('extra')
            with self.assertRaises(ValueError): api.verify_frozen_tree(root, manifest)

    def test_stale_process_record_cannot_stop_a_process(self):
        api = importlib.import_module('glm53_profile')
        import os
        with self.assertRaisesRegex(ValueError, 'identity'):
            api.open_identity({'pid':os.getpid(),'start_ticks':-1,'command':['not this process']})

if __name__ == '__main__': unittest.main()
