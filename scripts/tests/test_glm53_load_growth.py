"""Incremental storage union cannot duplicate shared memory or alias weights. CPU only."""
import copy
import importlib.util
from pathlib import Path
import unittest

class GrowthTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('growth_probe',Path(__file__).resolve().parents[1]/'43_probe_glm53_growth.py')
        self.api=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.api)

    def test_required_storage_union_and_overlap(self):
        self.assertEqual(self.api.minimum_bytes(1),2124585984)
        self.assertEqual(self.api.minimum_bytes(2),3947169792)
        self.assertEqual(self.api.minimum_bytes(2)-self.api.minimum_bytes(1),1822583808)
        for bad in (0,3,True):
            with self.assertRaises(ValueError):self.api.minimum_bytes(bad)

    def test_shared_backing_identity_must_match(self):
        first={'shared_scratch':[{'storage_pointer':100,'storage_bytes':64}],
               'tensor_cache':[{'storage_pointer':200,'storage_bytes':16}]}
        second=copy.deepcopy(first)
        self.api.require_shared_identity(first,second)
        for key in ('shared_scratch','tensor_cache'):
            changed=copy.deepcopy(second);changed[key][0]['storage_pointer']+=1
            with self.assertRaises(ValueError):self.api.require_shared_identity(first,changed)
        changed=copy.deepcopy(second);changed['shared_scratch']=[]
        with self.assertRaises(ValueError):self.api.require_shared_identity(first,changed)

    def test_parameter_and_table_storage_are_independent(self):
        self.api.require_disjoint([{'storage_pointer':100,'storage_bytes':64}],
                                  [{'storage_pointer':200,'storage_bytes':64}])
        for pointer in (100,110,150):
            with self.assertRaises(ValueError):self.api.require_disjoint(
                [{'storage_pointer':100,'storage_bytes':64}], [{'storage_pointer':pointer,'storage_bytes':64}])

    def test_full_geometry_growth_evidence_mutations(self):
        # Full metadata geometry; canonical-byte generation is delegated to the
        # already qualified component API and mocked only in this CPU census test.
        import hashlib, json, math, tempfile
        from unittest.mock import patch
        api=self.api; c=api.api; specs=c.tensor_specs('moe'); names=[s['name'] for s in specs]; digest='a'*64
        pointer=1<<40
        def storage(shape,dtype):
            nonlocal pointer
            size=math.prod(shape)*({**c.SIZES,'torch.int64':8}[dtype]);address=pointer;pointer+=size+4096
            return {'device':'cuda:0','dtype':dtype,'shape':shape,'stride':c.contiguous_stride(shape),'storage_offset':0,
                    'storage_bytes':size,'storage_pointer':address,'data_pointer':address}
        def memory(current,peak=None):
            peak=current if peak is None else peak
            return {'process_kib':{'Rss':1,'Pss':1,'Pss_Anon':1,'Pss_File':0},'cuda_allocated':current,
                    'cuda_reserved':peak+4096,'cuda_peak_allocated':peak,'cuda_peak_reserved':peak+4096,
                    'pinned_allocator':{f'{group}.{kind}':c.CAPACITY*2 for group in
                        ('allocated_bytes','active_bytes','allocations','active_requests') for kind in ('current','peak','allocated','freed')}}
        shared={'shared_scratch':[storage(shape,'torch.float16') for shape in [[6,2048,4096]]*2+[[6,2048,2048]]*2],
                'tensor_cache':[storage(shape,'torch.float16') for shape in [[1,2048],[1,4096]]]}
        rows=[{'event':'configured','selection':'persistent_pinned_growth','layers':2,'multiprocessors':48,
               'triton':'all_specializations_rejected','retuning':'rejected'}]; retained=[];finals=[];blocks=[]
        for index in range(2):
            before=api.minimum_bytes(1) if index else 0
            params={name:storage(meta['shape'],meta['dtype']) for name,meta in c.parameter_layout('moe').items()}
            block=[{'event':'memory','phase':'before_constructor',**memory(before)},
                   {'event':'constructed','parameters':params,'memory':memory(before+api.PARAMETERS)}]
            final={}
            for j,spec in enumerate(specs):
                view=c.expected_loaded_storage('moe',spec,params);size=c.byte_count(spec)
                block += [{'event':'loaded','name':spec['name'],'sha256':digest,'storage':view},
                          {'event':'transfer','name':spec['name'],'bytes':size,'source_sha256':digest,'device_sha256':digest,
                           'staging_pointer':100,'staging_bytes':c.CAPACITY,'pinned':True,'upload_chunks':1,
                           'completed_reuses':2*(j+1),'temporary_bytes':size}]
                final[spec['name']]={'event':'final_bytes','name':spec['name'],'sha256':digest,'storage':view}
            block += [{'event':'memory','phase':'after_transfers',**memory(before+api.PARAMETERS,before+api.PARAMETERS+2097152)},
                      {'event':'memory','phase':'after_finalization',**memory(api.minimum_bytes(index+1))},*final.values()]
            tables={}
            for projection,suffix in [('gate','gate_proj'),('up','up_proj'),('down','down_proj')]:
                for part in ('trellis','suh','svh'):
                    tables[projection+'_'+part]={'storage':storage([288],'torch.int64'),
                        'alias':projection+'_'+('t' if part=='trellis' else part)+'_ptrs',
                        'values':[final[f'model.language_model.layers.3.mlp.experts.{e}.{suffix}.{part}']['storage']['data_pointer'] for e in range(288)]}
            value={**shared,'pointer_tables':tables,'handles':c.expected_handles('moe'),'concurrency':6,
                   'module_object':9999+index*100000,'owned_parameters':sorted(params),
                   'python_handles':list(range(1+index*10000,865+index*10000)),
                   'native_handles':list(range(1000+index*10000,1864+index*10000)),
                   'scratch_keys':[['cuda:0',4096,2048,6]],
                   'tensor_cache_keys':['cuda:0/(1, 2048)/torch.float16/','cuda:0/(1, 4096)/torch.float16/']}
            block += [{'event':'retained',**value,'memory':memory(api.minimum_bytes(index+1))}]
            for row in block:row['layer_index']=index
            rows+=block;retained.append(value);finals.append(final);blocks.append(block)
        rows += [{'event':'first_layer_recheck','name':name,'sha256':digest,'storage':finals[0][name]['storage']} for name in names]
        rows += [{'event':'both_live','layers':2,'first_retained':retained[0],'memory':memory(api.minimum_bytes(2))}]
        for i,row in enumerate(rows):row['time_unix']=i+1
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'fixture').mkdir();(root/'fixture/weights.safetensors').write_bytes(b'CPU census input only')
            for filename in ('manifest.json','summary.json','raw.jsonl','traceback.log'):(root/filename).touch()
            inventory=c.file_inventory(root/'fixture');digests={s['name']:digest for s in [*specs,*c.EXCLUDED]}
            fixture={'schema_version':1,'qualification':'generated_synthetic_input_only','case':'moe','seed':7,
                     'generator_sha256':c.sha256_file(c.ROOT/'scripts/lib/glm53_load_fixture.py'),'inventory':inventory,
                     'selection':{s['name']:{'file':'weights.safetensors','dtype':c.DTYPES[s['dtype']],'shape':s['shape'],'sha256':digest} for s in specs},
                     'excluded':{s['name']:{'dtype':c.DTYPES[s['dtype']],'shape':s['shape'],'sha256':digest} for s in c.EXCLUDED}}
            (root/'fixture.json').write_text(json.dumps(fixture))
            with patch.object(c,'canonical_fixture',return_value=(digests,inventory)) as canonical:
                self.assertEqual(api.score_capture(root,rows,7)['known_retained_bytes'],3947169792)
                canonical.assert_called_once_with('moe',7)
                mutations=[(1+len(blocks[0])+1,('memory','cuda_allocated'),0),
                           (1+len(blocks[0]),('cuda_allocated',),0),
                           (-1,('layers',),1),(-2,('sha256',),'b'*64),
                           (len(blocks[0]),('concurrency',),0),
                           (len(blocks[0])+len(blocks[1]),('tensor_cache_keys',),[])]
                for index,path,value in mutations:
                    changed=copy.deepcopy(rows);target=changed[index]
                    for key in path[:-1]:target=target[key]
                    target[path[-1]]=value
                    with self.subTest(path=path),self.assertRaises(ValueError):api.score_capture(root,changed,7)
                changed=copy.deepcopy(rows);changed[-1]['first_retained']['pointer_tables']['gate_trellis']['values'][0]+=2
                with self.assertRaises(ValueError):api.score_capture(root,changed,7)
                with self.assertRaises(ValueError):api.score_capture(root,rows[:-1],7)
                changed=copy.deepcopy(rows);changed[3]['layer_index']=1
                with self.assertRaises(ValueError):api.score_capture(root,changed,7)
            with patch.object(c,'canonical_fixture',side_effect=ValueError('canonical verification required')):
                with self.assertRaisesRegex(ValueError,'canonical verification required'):api.score_capture(root,rows,7)

if __name__=='__main__':unittest.main()
