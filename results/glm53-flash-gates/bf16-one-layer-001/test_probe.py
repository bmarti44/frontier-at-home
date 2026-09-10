"""Reject corrupt payload/header combinations before native weight use."""
import copy,hashlib,importlib.util,json,struct,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('one_layer_probe',Path(__file__).with_name('probe.py'));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class Shards(unittest.TestCase):
    def fixture(self,header=None,payload=b'\x00\x3f\x80\x3f'):
        if header is None:header={'tensor':{'dtype':'BF16','shape':[2],'data_offsets':[0,4]}}
        text=json.dumps(header).encode();b=bytearray(struct.pack('<Q',len(text))+text+payload)
        return b,{'size':len(b),'lfs':{'sha256':hashlib.sha256(b).hexdigest()}},header
    def test_valid(self):
        b,r,h=self.fixture();offset,observed=p.validate_shard(b,r,h);self.assertEqual(observed,h);self.assertEqual(bytes(b[offset:]),b'\x00\x3f\x80\x3f')
    def test_corruption_and_truncation(self):
        b,r,h=self.fixture();b[-1]^=1
        with self.assertRaisesRegex(ValueError,'digest'):p.validate_shard(b,r,h)
        with self.assertRaisesRegex(ValueError,'byte count'):p.validate_shard(b[:-1],r,h)
    def test_pinned_header_and_geometry(self):
        b,r,h=self.fixture();bad=copy.deepcopy(h);bad['tensor']['shape']=[1]
        with self.assertRaisesRegex(ValueError,'pinned'):p.validate_shard(b,r,bad)
        b,r,h=self.fixture(bad)
        with self.assertRaisesRegex(ValueError,'geometry'):p.validate_shard(b,r,h)
    def test_unclaimed_bytes_and_offsets(self):
        b,r,h=self.fixture(payload=b'123456')
        with self.assertRaisesRegex(ValueError,'unclaimed'):p.validate_shard(b,r,h)
        h={'tensor':{'dtype':'BF16','shape':[2],'data_offsets':[1,5]}};b,r,h=self.fixture(h,payload=b'12345')
        with self.assertRaisesRegex(ValueError,'offsets'):p.validate_shard(b,r,h)
    def test_duplicate_and_nonfinite_json(self):
        for raw in ['{"x":1,"x":2}','{"x":NaN}','{"x":Infinity}']:
            with self.assertRaises(ValueError):p.strict(raw)

# These deliberately synthetic files exercise the evidence validator only;
# they are never native layer observations or reference data.
import gzip,tempfile
class Evidence(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.out=self.root/'checks';self.out.mkdir();(self.root/'metadata').mkdir()
        self.digest=hashlib.sha256(b'synthetic-validator-fixture').hexdigest()
        self.shards=[{'rfilename':str(i),'size':1,'lfs':{'sha256':self.digest}} for i in range(3)]
        self.dump('manifest.json',{'shards':self.shards});self.dump('randomness.json',{'seed':0,'randomness':'0'*64})
        spec={'shape':[1],'dtype':'torch.bfloat16','bytes':2};self.dump('metadata/layer-plan.json',{'tensors':{'test':spec},'converted_layer_bytes':2})
        self.verified=[{'path':x['rfilename'],'size_bytes':1,'sha256':self.digest} for x in self.shards]
        self.dump('checks/staging.json',{'weights':[{'name':'test',**spec,'pinned':True,'sha256':self.digest}],'persistent_until_forward_completion':True,'converted_layer_bytes':2,'native_source_tensors':892,'downloaded_shards':self.verified})
        self.rows=[]
        for shard in self.verified:self.rows.extend([{'kind':'download_start','shard':shard['path']},{'kind':'verified_shard','shard':shard}])
        self.rows.extend([{'kind':'verified_gpu_tensor','name':'test','sha256':self.digest},{'kind':'weights_on_gpu'},{'kind':'forward_complete'}])
        self.raw();self.payload=bytes(516*4*4096*2);self.output(self.payload)
    def tearDown(self):self.tmp.cleanup()
    def dump(self,name,value):(self.root/name).write_text(json.dumps(value))
    def raw(self):
        for i,row in enumerate(self.rows):row['time_unix']=float(i+1)
        (self.out/'raw.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in self.rows))
    def output(self,b):
        (self.out/'output.bf16.gz').write_bytes(gzip.compress(b,mtime=0));self.dump('checks/output.json',{'shape':[1,516,4,4096],'dtype':'torch.bfloat16','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'seed':0,'input':{'shape':[1,516,4,4096],'dtype':'torch.bfloat16','sha256':self.digest}})
    def test_valid_shape_only_control(self):self.assertEqual(p.score(self.root)['verdict'],'PASS')
    def test_missing_gpu_tensor(self):
        self.rows=[x for x in self.rows if x['kind']!='verified_gpu_tensor'];self.raw()
        with self.assertRaisesRegex(ValueError,'GPU parameter coverage'):p.score(self.root)
    def test_duplicate_shard(self):
        self.rows.insert(1,self.rows[1].copy());self.raw()
        with self.assertRaisesRegex(ValueError,'shard coverage'):p.score(self.root)
    def test_changed_gpu_digest(self):
        next(x for x in self.rows if x['kind']=='verified_gpu_tensor')['sha256']='bad';self.raw()
        with self.assertRaisesRegex(ValueError,'digest'):p.score(self.root)
    def test_nonfinite_payload(self):
        self.output(b'\x80\x7f'+self.payload[2:])
        with self.assertRaisesRegex(ValueError,'nonfinite output'):p.score(self.root)
    def test_short_payload(self):
        self.output(self.payload[:-2])
        with self.assertRaisesRegex(ValueError,'geometry'):p.score(self.root)
    def test_wrong_seed(self):
        self.dump('randomness.json',{'seed':1,'randomness':'0'*64})
        with self.assertRaisesRegex(ValueError,'seed'):p.score(self.root)

if __name__=='__main__':unittest.main()
