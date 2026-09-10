"""Execute the freezer's actual inventory block against the installed profile runtime."""
import ast,gzip,sys,tempfile,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[3];B=Path.home()/'.cache/glm53-flash'
sys.path.insert(0,str(R/'scripts/lib'))
from glm53_contract import strict_json,verify_inventory,sha256_file
from glm53_probe_capture import inference_lock

class RuntimeInventory(unittest.TestCase):
    def test_freezer_verifies_existing_profile_runtime(self):
        source=Path(__file__).with_name('freeze.py');tree=ast.parse(source.read_text());body=next(n for n in tree.body if isinstance(n,ast.With)).body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='runtime' for t in n.targets))
        end=next(i+1 for i,n in enumerate(body[start:],start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='verified' for t in n.targets))
        with tempfile.TemporaryDirectory() as d,inference_lock():
            out=Path(d);(out/'metadata').mkdir()
            def save(p,v):p.write_text(__import__('json').dumps(v))
            scope=dict(R=R,B=B,O=out,Path=Path,gzip=gzip,json=__import__('json'),save=save,strict_json=strict_json,verify_inventory=verify_inventory,sha256_file=sha256_file)
            exec(compile(ast.Module(body=body[start:end],type_ignores=[]),str(source),'exec'),scope)
            profile=strict_json(R/'configs/profiles/glm-5.3-flash/cuda-spark-128g-agent-fast.json')
            build=strict_json(R/'configs/build-manifests/glm53-flash-local.json')
            expected=build['inventories'][profile['artifact_roles']['runtime_inventory']].copy();expected['path']=expected['path'].replace('{repo}',str(R))
            self.assertEqual(sha256_file(Path(expected['path'])),expected['sha256'])
            current=__import__('json').loads(gzip.decompress(Path(expected['path']).read_bytes()))
            self.assertEqual(len(scope['verified']),len(current['files']))

if __name__=='__main__':unittest.main()
