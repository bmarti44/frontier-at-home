"""Load probe byte coverage and storage metadata; synthetic CPU fixtures only."""
import importlib.util
from pathlib import Path
import unittest
import copy


class LoadProbeTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('load_probe_test',Path(__file__).resolve().parents[1]/'42_probe_glm53_load.py')
        self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)

    def test_constructor_parameter_formulas(self):
        expected={'moe':1822563072,'kda':53707440,'mla':4214800,'ordinary':1268776960}
        for case,total in expected.items():
            self.assertEqual(self.api.layout_bytes(self.api.parameter_layout(case)),total)
        with self.assertRaises(ValueError):self.api.parameter_layout('full-model')

    def test_storage_description_keeps_alias_and_offset(self):
        import torch
        base=torch.arange(30,dtype=torch.int16).reshape(5,6);view=base[:,2:4]
        a,b=self.api.describe(base),self.api.describe(view)
        self.assertEqual(a['storage_pointer'],b['storage_pointer'])
        self.assertEqual(b['data_pointer']-a['data_pointer'],4)
        self.assertEqual(b['shape'],[5,2]);self.assertEqual(b['stride'],[6,1])
        self.assertEqual(b['storage_bytes'],60)

    def test_byte_coverage_rejects_missing_duplicate_mismatched_or_bad_transfer(self):
        names={'tensor-a':'a'*64,'tensor-b':'b'*64}
        rows=[]
        for name,digest in names.items():
            rows.extend([{'event':'loaded','name':name,'sha256':digest},
                         {'event':'transfer','name':name,'source_sha256':digest,'device_sha256':digest},
                         {'event':'final_bytes','name':name,'sha256':digest}])
        self.api.validate_byte_coverage(rows,names)
        variants=[rows[:-1],rows+[rows[0]],copy.deepcopy(rows),copy.deepcopy(rows)]
        variants[2][0]['sha256']='0'*64;variants[3][1]['device_sha256']='0'*64
        for changed in variants:
            with self.assertRaises(ValueError):self.api.validate_byte_coverage(changed,names)


if __name__=='__main__':unittest.main()

class LoadScoreTests(unittest.TestCase):
    setUp = LoadProbeTests.setUp
    def test_reject_retained_nested_temporary(self):
        import torch
        layer=torch.nn.Module(); temp=torch.zeros(8); layer.weight=torch.nn.Parameter(torch.ones(8))
        self.api.reject_retained_temporary(layer,temp)
        layer.extra={'nested':[temp[2:]]}
        with self.assertRaisesRegex(ValueError,'retained'): self.api.reject_retained_temporary(layer,temp)

    def test_complete_ordinary_record_and_mutations(self):
        import tempfile, json
        from unittest.mock import patch
        api=self.api; spec=api.tensor_specs('ordinary')[0]; spec={**spec, 'shape':[2,4]}; size=16; name=spec['name']
        digest=api.fixture_digest(7,spec)
        storage={'device':'cuda:0','dtype':'torch.bfloat16','shape':[2,4],'stride':[4,1],
                 'storage_pointer':4096,'storage_bytes':size,'data_pointer':4096,'storage_offset':0}
        memory={'process_kib':{'Rss':1,'Pss':1,'Pss_Anon':1,'Pss_File':0},'cuda_allocated':size,
                'cuda_reserved':size,'cuda_peak_allocated':size,'cuda_peak_reserved':size,
                'pinned_allocator':{f'{group}.{kind}':api.CAPACITY for group in
                    ('allocated_bytes','active_bytes','allocations','active_requests') for kind in ('current','peak','allocated','freed')}}
        chunks=(size+api.CAPACITY-1)//api.CAPACITY
        rows=[{'event':'configured','case':'ordinary','selection':'persistent_pinned_stream','triton':'all_specializations_rejected',
               'retuning':'rejected','minimal_MoE_context':False,'pinned_capacity':api.CAPACITY},
              {'event':'memory','phase':'before_constructor',**memory},
              {'event':'constructed','parameters':{'weight':storage},'memory':memory},
              {'event':'loaded','name':name,'sha256':digest,'storage':storage},
              {'event':'transfer','name':name,'bytes':size,'source_sha256':digest,'device_sha256':digest,
               'staging_pointer':8192,'staging_bytes':api.CAPACITY,'pinned':True,'upload_chunks':chunks,
               'completed_reuses':chunks*2,'temporary_bytes':size},
              {'event':'memory','phase':'after_transfers_and_verification',**memory},
              {'event':'memory','phase':'after_finalization',**memory},
              {'event':'final_bytes','name':name,'sha256':digest,'storage':storage},
              {'event':'retained','pointer_tables':{},'shared_scratch':[],'tensor_cache':[],'handles':[],'memory':memory}]
        for i,row in enumerate(rows):row['time_unix']=i+1
        with tempfile.TemporaryDirectory() as td, patch.object(api,'tensor_specs',return_value=[spec]), \
                patch.object(api,'parameter_layout',return_value={'weight':{'dtype':'torch.bfloat16','shape':[2,4]}}), \
                patch.object(api,'EXCLUDED',[]):
            root=Path(td); fixture=api.write_fixture(root/'fixture','ordinary',7)
            for filename in ('manifest.json','summary.json','raw.jsonl','traceback.log'): (root/filename).touch()
            (root/'fixture.json').write_text(json.dumps(fixture))
            self.assertEqual(api.score_capture(root,rows,'ordinary',7)['bytes_checked_per_stage'],size)
            mutations=[(4,'pinned',False),(4,'completed_reuses',chunks*2-1),(4,'device_sha256','b'*64),
                       (4,'temporary_bytes',0),(4,'staging_bytes',1),(4,'upload_chunks',1),
                       (0,'minimal_MoE_context',True),(8,'handles',[{}]),(8,'pointer_tables',{'fake':{}})]
            for index,key,value in mutations:
                changed=copy.deepcopy(rows);changed[index][key]=value
                with self.subTest(key=key), self.assertRaises(ValueError):api.score_capture(root,changed,'ordinary',7)
            for index,key,value in [(7,'storage_pointer',8192),(7,'storage_offset',1),(3,'stride',[5,1]),(3,'dtype',[]),
                                    (7,'storage_bytes',size-1)]:
                changed=copy.deepcopy(rows);changed[index]['storage'][key]=value
                with self.subTest(key=key), self.assertRaises(ValueError):api.score_capture(root,changed,'ordinary',7)
            for changed in (rows[:-1], rows+[rows[-1]], rows[:3]+rows[4:]):
                with self.assertRaises(ValueError):api.score_capture(root,changed,'ordinary',7)
            changed=copy.deepcopy(rows);changed[8]['memory']['cuda_allocated']=float('nan')
            with self.assertRaises(ValueError):api.score_capture(root,changed,'ordinary',7)
            changed=copy.deepcopy(rows)
            for row in changed:
                mem=row.get('memory',row)
                for key in ('cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved'):
                    if key in mem:mem[key]=0
            with self.subTest(finding='H1'), self.assertRaises(ValueError):api.score_capture(root,changed,'ordinary',7)
            path=root/'fixture/weights.safetensors';path.chmod(0o644);path.write_bytes(b'invalid retained payload')
            fixture['inventory']=api.file_inventory(root/'fixture');(root/'fixture.json').write_text(json.dumps(fixture))
            with self.subTest(finding='H2'), self.assertRaises(ValueError):api.score_capture(root,rows,'ordinary',7)
