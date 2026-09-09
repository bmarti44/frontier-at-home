"""Full-size metadata and independent, chunk-invariant synthetic fixture bytes."""
import importlib
import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'lib'))


class LoadFixtureTests(unittest.TestCase):
    def setUp(self):self.api=importlib.import_module('glm53_load_fixture')

    def test_full_geometry_and_total_checkpoint_bytes(self):
        expected={'moe':(3456,1822559616),'kda':(15,53026828),'mla':(8,4214792),'ordinary':(1,1268776960)}
        for case,(count,size) in expected.items():
            specs=self.api.tensor_specs(case)
            self.assertEqual(len(specs),count);self.assertEqual(len({s['name'] for s in specs}),count)
            self.assertEqual(sum(self.api.byte_count(s) for s in specs),size)
        with self.assertRaises(ValueError):self.api.tensor_specs('full-model')

    def test_specs_match_pinned_checkpoint_and_overlay_shapes(self):
        base=Path('/home/bmarti44/.cache/glm53-flash/model-layout-001/k2')
        headers={}
        for path in base.glob('*.header.json'):
            headers.update({k:v for k,v in json.loads(path.read_text()).items() if k!='__metadata__'})
        self.assertTrue(headers)
        plan=json.loads((Path(__file__).resolve().parents[2]/'results/glm53-flash-gates/model-layout-001/overlay-plan.json').read_text())['plan']
        for e in plan:
            shapes={'trellis':('I16',[e['in']//16,e['out']//16,16*e['k']]),'suh':('F16',[e['in']]),
                    'svh':('F16',[e['out']]),e['marker']:('I32',[])}
            for suffix,(dtype,shape) in shapes.items():headers[e['base']+'.'+suffix]={'dtype':dtype,'shape':shape}
        for case in ('moe','kda','mla','ordinary'):
            for spec in self.api.tensor_specs(case):
                source=headers[spec['name']]
                self.assertEqual((spec['dtype'],spec['shape']),(source['dtype'],source['shape']))

    def test_chunk_boundaries_seed_and_identity_bind_every_byte(self):
        spec={'name':'synthetic.packed','dtype':'I16','shape':[4101]}
        whole=self.api.fixture_bytes(123,spec,0,8202)
        chunks=b''.join(self.api.fixture_bytes(123,spec,i,min(777,8202-i)) for i in range(0,8202,777))
        self.assertEqual(whole,chunks)
        self.assertNotEqual(whole,self.api.fixture_bytes(124,spec,0,8202))
        self.assertNotEqual(whole,self.api.fixture_bytes(123,dict(spec,name='synthetic.other'),0,8202))
        self.assertNotEqual(whole[:4096],whole[4096:8192])
        for start,count in ((-1,1),(8202,1),(0,-1)):
            with self.assertRaises(ValueError):self.api.fixture_bytes(123,spec,start,count)
        for seed in (True,-1,2**64):
            with self.assertRaises(ValueError):self.api.fixture_bytes(seed,spec,0,1)

    def test_finite_patterns_and_exact_codebook_markers(self):
        for dtype in ('F16','BF16'):
            spec={'name':'synthetic.weight','dtype':dtype,'shape':[4096]}
            raw=self.api.fixture_bytes(123,spec,0,8192)
            words=np.frombuffer(raw,dtype='<u2')
            values=words.view('<f2') if dtype=='F16' else (words.astype('<u4')<<16).view('<f4')
            self.assertTrue(np.isfinite(values).all());self.assertTrue((values!=0).all())
        for suffix,value in (('mcg',-877912083),('mul1',-2082680531)):
            spec={'name':'synthetic.'+suffix,'dtype':'I32','shape':[]}
            raw=self.api.fixture_bytes(123,spec,0,4)
            self.assertEqual(int.from_bytes(raw,'little',signed=True),value)


if __name__=='__main__':unittest.main()
