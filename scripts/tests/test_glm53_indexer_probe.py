"""Reject malformed indexer captures before any native preparation run."""
import copy
import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest


class IndexerProbeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('indexer_probe_test', Path(__file__).resolve().parents[1] / '45_probe_glm53_indexer.py')
        self.api = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.api)
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root = Path(self.temp.name)

    def test_chunk_geometry_and_exact_valid_bounds(self):
        a = self.api
        self.assertEqual(a.call_specs('prefill-1'), [{'kind':'prefill','start':0,'stop':2048,'columns':65536}])
        self.assertEqual(a.call_specs('prefill-4'), [{'kind':'prefill','start':0,'stop':1024,'columns':131072}, {'kind':'prefill','start':1024,'stop':2048,'columns':131072}])
        starts, ends = a.logit_bounds(7,'prefill-4',0)
        self.assertEqual((int(starts[0]),int(ends[0])),(0,65408))
        self.assertEqual((int(starts[512]),int(ends[512])),(65536,130944))
        self.assertEqual(int(ends[-1]),131072)
        starts, ends = a.logit_bounds(7,'decode-4',0)
        self.assertEqual(starts.tolist(),[0]*4)
        self.assertEqual(ends.tolist(),[65535,65535,65535,65536])
        for bad in (-1,1,True):
            with self.assertRaises(ValueError): a.logit_bounds(7,'decode-4',bad)

    def logit_file(self, mutation=None):
        a = self.api; cfg = a.fixture.case_config(7,'decode-4'); payload = []
        for r,n,p in zip(cfg['row_requests'],cfg['history_counts'],cfg['positions']):
            payload.append(a.fixture.valid_logits(7,int(r),int(n),(int(p)+1)//4).tobytes())
        if mutation: mutation(payload)
        path=self.root/'logits-decode-4-0.f32.gz'; path.write_bytes(gzip.compress(b''.join(payload)))
        return path,a.sha256_file(path)

    def test_every_valid_logit_byte_and_closed_payload(self):
        a=self.api; path,digest=self.logit_file()
        self.assertEqual(a.score_logits(path,digest,7,'decode-4',0)['valid_logit_elements'],262141)
        for mutate in (lambda rows: rows.__setitem__(0,b'\x00'*4+rows[0][4:]),
                       lambda rows: rows.__setitem__(1,rows[0]),
                       lambda rows: rows.__setitem__(3,rows[3]+b'\x00'*4),
                       lambda rows: rows.__setitem__(0,b'\x00\x00\xc0\x7f'+rows[0][4:])):
            path,digest=self.logit_file(mutate)
            with self.assertRaises(ValueError): a.score_logits(path,digest,7,'decode-4',0)
        path,digest=self.logit_file()
        with self.assertRaises(ValueError): a.score_logits(path,'0'*64,7,'decode-4',0)
        target=path.with_suffix('.link'); target.symlink_to(path)
        with self.assertRaises(ValueError): a.score_logits(target,digest,7,'decode-4',0)

    def test_decode_tensor_capture_and_corruption(self):
        import numpy as np
        a=self.api; cfg=a.fixture.case_config(7,'decode-4')
        indices=np.full((4,2048),-1,dtype='<i4')
        for i,(r,n,p) in enumerate(zip(cfg['row_requests'],cfg['history_counts'],cfg['positions'])):
            pools=a.fixture.top_history(7,int(r),int(n))[:511]
            indices[i,:2044]=(pools[:,None]*4+np.arange(4)).reshape(-1)
            count=(int(p)+1)%4; indices[i,2044:2044+count]=np.arange(int(p)//4*4,int(p)//4*4+count)
        cache,tail=a.fixture.cache_and_tail(7,'decode-4',final=True)
        blobs=[indices.tobytes(),cache.tobytes(),tail.tobytes()]
        paths=[self.root/name for name in a.tensor_names('decode-4')]
        for path,blob in zip(paths,blobs): path.write_bytes(gzip.compress(blob))
        hashes=[a.sha256_file(p) for p in paths]
        self.assertEqual(a.score_tensors(paths,hashes,7,'decode-4')['invalid_indices'],0)
        for i in range(3):
            changed=bytearray(blobs[i]); changed[0]^=1
            paths[i].write_bytes(gzip.compress(changed)); hashes[i]=a.sha256_file(paths[i])
            with self.assertRaises(ValueError): a.score_tensors(paths,hashes,7,'decode-4')
            paths[i].write_bytes(gzip.compress(blobs[i])); hashes[i]=a.sha256_file(paths[i])

    def test_memory_minima_and_malformed_receipts(self):
        a=self.api
        good={'cuda_allocated':2000000000,'cuda_reserved':4000000000,'cuda_peak_allocated':3500000000,
              'device_free':110000000000,'device_total':120000000000,'workspace_bytes':1385168896}
        a.validate_memory(good,'prefill-4')
        for key,value in [('cuda_allocated',0),('cuda_peak_allocated',2000000000),('cuda_reserved',True),('device_free',float('nan')),('workspace_bytes',1)]:
            row=copy.deepcopy(good); row[key]=value
            with self.assertRaises(ValueError): a.validate_memory(row,'prefill-4')
        with self.assertRaises(ValueError): a.score_capture(self.root,[],7)


if __name__=='__main__': unittest.main()
