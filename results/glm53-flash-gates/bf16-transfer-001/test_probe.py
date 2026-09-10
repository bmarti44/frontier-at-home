"""Reject framing errors before reading an unexpectedly large response."""
import importlib.util,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('transport',Path(__file__).with_name('probe.py'));p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
class Headers(unittest.TestCase):
    def test_exact_partial_response(self):p.check_headers(206,{'Content-Range':['bytes 10-19/100'],'Content-Length':['10']},10,19,100)
    def test_ignore_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'206'):p.check_headers(200,{'Content-Length':'100'},10,19,100)
    def test_wrong_framing_rejected(self):
        for h in [{'Content-Range':'bytes 10-20/100','Content-Length':'10'},{'Content-Range':'bytes 10-19/101','Content-Length':'10'},{'Content-Range':'bytes 10-19/100','Content-Length':'100'},{'Content-Range':'bytes 10-19/100','Content-Length':'10','Content-Encoding':'gzip'},{}]:
            with self.subTest(h=h),self.assertRaises(ValueError):p.check_headers(206,{k:[v] for k,v in h.items()},10,19,100)
    def test_missing_arms_rejected(self):
        with self.assertRaisesRegex(ValueError,'four'):p.score([])

class RawHeaders(unittest.TestCase):
    def response(self,headers,body):
        import io,http.client
        class Socket:
            def makefile(self,*args):return io.BytesIO(b'HTTP/1.1 206 Partial Content\r\nContent-Range: bytes 0-9/100\r\n'+headers+b'\r\n'+body)
        r=http.client.HTTPResponse(Socket());r.begin();return r
    def test_conflicting_lengths_rejected(self):
        r=self.response(b'Content-Length: 10\r\nContent-Length: 999\r\n',b'0123456789')
        with self.assertRaises(ValueError):p.check_headers(r.status,r.headers,0,9,100)
    def test_chunked_and_length_rejected(self):
        r=self.response(b'Transfer-Encoding: chunked\r\nContent-Length: 10\r\n',b'a\r\n0123456789\r\n0\r\n\r\n')
        with self.assertRaises(ValueError):p.check_headers(r.status,r.headers,0,9,100)
class Evidence(unittest.TestCase):
    def rows(self):
        records=[]
        for i,arm in enumerate('ABBA'):
            workers=1 if arm=='A' else 4;parts=[]
            for n in range(workers):
                left=n*p.SIZE//workers;right=left+p.SIZE//workers-1
                parts.append({'left':left,'right':right,'received':right-left+1,'status':206,'headers':{'Content-Range':[f'bytes {left}-{right}/{p.TOTAL}'],'Content-Length':[str(right-left+1)]}})
            host={'available_kib':115*1024**2,'pswpin':0,'pswpout':0,'used_swap_kib':0}
            records.append({'index':i,'arm':arm,'workers':workers,'status':'COMPLETE','bytes':p.SIZE,'sha256':'0'*64,'elapsed_seconds':1.,'host_before':host.copy(),'host_after':host.copy(),'ranges':parts})
        return records
    def test_valid_validator_fixture(self):self.assertEqual(p.score(self.rows())['verdict'],'PASS')
    def test_payload_mismatch(self):
        rows=self.rows();rows[1]['sha256']='1'*64
        with self.assertRaisesRegex(ValueError,'unequal'):p.score(rows)
    def test_gap_short_swap_and_nan(self):
        for name in ['gap','short','swap','nan']:
            rows=self.rows()
            if name=='gap':rows[1]['ranges'][0]['left']=1
            if name=='short':rows[1]['ranges'][0]['received']-=1
            if name=='swap':rows[1]['host_after']['pswpout']=1
            if name=='nan':rows[1]['elapsed_seconds']=float('nan')
            with self.subTest(name=name),self.assertRaises(ValueError):p.score(rows)

class OutputOwnership(unittest.TestCase):
    def test_probe_leaves_terminal_summary_to_controller(self):
        import contextlib,hashlib,io,json,tempfile,types
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            manifest={'probe':{'sha256':hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()},'frozen_at_unix':1595431051.,'node':'synthetic'}
            randomness={'round':2,'publication_unix':1595431080,'frozen_at_unix':1595431051.,'seed':0,'randomness':'0'*64,'signature':'fixture','previous_signature':'fixture'}
            (root/'manifest.json').write_text(json.dumps(manifest));(root/'randomness.json').write_text(json.dumps(randomness))
            with patch.object(p,'SIZE',16),patch.object(p,'host',return_value={'available_kib':115*1024**2,'pswpin':0,'pswpout':0,'used_swap_kib':0}),patch.object(p.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='DRAND_BLS_RECEIPT_OK\n')),patch.object(p.urllib.request,'urlopen',side_effect=ValueError('synthetic failed request')),contextlib.redirect_stdout(io.StringIO()):
                self.assertFalse(p.run(root))
            self.assertFalse((root/'summary.json').exists(),'terminal summary belongs to controller')
            self.assertEqual(json.loads((root/'transport-summary.json').read_text())['verdict'],'FAIL')

if __name__=='__main__':unittest.main()
