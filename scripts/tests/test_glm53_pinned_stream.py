"""Actual safetensors selection and staged-byte contract; GPU calls mocked here."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'lib'))


class PinnedStreamTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('glm53_pinned_stream')
        import torch
        from safetensors.torch import save_file
        from glm53_mla_replay import file_inventory
        self.torch=torch
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.values={'model.layers.3.packed':torch.arange(17,dtype=torch.int16),
                     'model.layers.3.marker':torch.tensor(-877912083,dtype=torch.int32),
                     'model.layers.3.signs':torch.tensor([1.,-1.,1.],dtype=torch.float16)}
        save_file({**self.values,'stale.weight':torch.ones(2),'model.layers.45.weight':torch.ones(3)},self.root/'a.safetensors')
        self.inventory=file_inventory(self.root)
        self.selection={name:{'file':'a.safetensors','dtype':str(t.dtype),'shape':list(t.shape),
            'sha256':hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
            for name,t in self.values.items()}

    def test_default_off_does_not_import_or_access_inputs(self):
        with patch('builtins.__import__',side_effect=AssertionError('disabled import')):
            self.assertIsNone(self.api.stream_selected_weights(None,None,None,None,None,enabled=False))
        with self.assertRaises(ValueError):self.api.stream_selected_weights(None,None,None,None,None,enabled=1)

    def test_stock_lazy_iterator_filters_before_get_tensor_without_global_patch(self):
        from vllm.model_executor.model_loader import weight_utils as weights
        original=weights.safe_open; skip=weights.should_skip_weight; touched=[]
        class Reader:
            def __init__(inner,*a,**k):inner.context=original(*a,**k)
            def __enter__(inner):inner.reader=inner.context.__enter__();return inner
            def __exit__(inner,*a):return inner.context.__exit__(*a)
            def keys(inner):return inner.reader.keys()
            def get_tensor(inner,name):touched.append(name);return inner.reader.get_tensor(name)
        with patch.object(weights,'safe_open',Reader),patch.object(weights,'_prefetch_all_checkpoints',side_effect=AssertionError('prefetch')):
            result=dict(self.api.selected_weights(self.root,self.inventory,self.selection))
        self.assertEqual(set(touched),set(self.values));self.assertEqual(len(touched),3)
        self.assertIs(weights.should_skip_weight,skip)
        for name,value in result.items():self.assertTrue(self.torch.equal(value,self.values[name]))

    def test_selection_rejects_missing_wrong_shape_dtype_and_source_changes(self):
        for field,value in [('shape',[99]),('dtype','torch.float32'),('file','../a.safetensors')]:
            selection=json.loads(json.dumps(self.selection));selection[next(iter(selection))][field]=value
            with self.assertRaises(ValueError):list(self.api.selected_weights(self.root,self.inventory,selection))
        selection=dict(self.selection);selection['missing']=selection[next(iter(selection))]
        with self.assertRaises(ValueError):list(self.api.selected_weights(self.root,self.inventory,selection))
        (self.root/'a.safetensors').write_bytes(b'changed')
        with self.assertRaises(ValueError):self.api.selected_weights(self.root,self.inventory,self.selection)

    def test_scalar_short_chunks_reuse_and_copy_corruption_rejected(self):
        torch=self.torch; original_empty=torch.empty; events=[]; allocations=[]
        def empty(*a,**k):
            allocations.append(dict(k));k.pop('pin_memory',None);k['device']='cpu';return original_empty(*a,**k)
        event=MagicMock();event.record.side_effect=lambda:events.append('record');event.synchronize.side_effect=lambda:events.append('complete')
        consumed={};rows=[]
        def consume(name,tensor):
            self.assertEqual(events[-1],'complete');consumed[name]=tensor.clone()
        with patch.object(torch,'empty',side_effect=empty),patch.object(torch.Tensor,'is_pinned',return_value=True), \
             patch.object(torch.cuda,'Event',return_value=event),patch.object(torch.cuda,'synchronize'):
            self.api.stream_selected_weights(self.root,self.inventory,self.selection,consume,rows.append,enabled=True,pinned_capacity=7)
        self.assertEqual(set(consumed),set(self.values))
        for name,value in consumed.items():self.assertTrue(torch.equal(value,self.values[name]))
        self.assertEqual(sum(bool(a.get('pin_memory')) for a in allocations),1)
        self.assertEqual(len({r['staging_pointer'] for r in rows}),1)
        self.assertTrue(all(r['source_sha256']==r['device_sha256']==self.selection[r['name']]['sha256'] for r in rows))
        changed=json.loads(json.dumps(self.selection));changed[next(iter(changed))]['sha256']='0'*64
        with patch.object(torch,'empty',side_effect=empty),patch.object(torch.Tensor,'is_pinned',return_value=True), \
             patch.object(torch.cuda,'Event',return_value=event),patch.object(torch.cuda,'synchronize'):
            with self.assertRaises(ValueError):
                self.api.stream_selected_weights(self.root,self.inventory,changed,consume,rows.append,enabled=True,pinned_capacity=7)


if __name__=='__main__':unittest.main()
