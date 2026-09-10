"""Exercise the native probe's actual download block with synthetic HTTP bodies."""
import ast,importlib.util,io,json,types,unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

SOURCE=Path(__file__).with_name('probe.py')
spec=importlib.util.spec_from_file_location('native_transfer_probe',SOURCE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Transfer(unittest.TestCase):
    def exercise(self,mode=None,small_chunks=False):
        payload=bytes(range(251))*4+bytes(range(3));buffer=bytearray(len(payload));requests=[];events=[]
        class Response(io.BytesIO):
            def __init__(self,request):
                requested=request.get_header('Range');requests.append(requested)
                left,right=map(int,requested.removeprefix('bytes=').split('-')) if requested else (0,len(payload)-1)
                super().__init__(payload[left:right+1]);self.status=206 if requested else 200;self.url='https://synthetic.invalid/shard'
                self.headers=Message();self.headers['Content-Length']=str(right-left+1)
                if requested:self.headers['Content-Range']=f'bytes {left}-{right}/{len(payload)}'
                if mode=='ignored':self.status=200
                if mode=='duplicate':self.headers['Content-Length']='999999'
                if mode=='short':self.truncate(max(0,right-left))
                if mode=='encoding':self.headers['Content-Encoding']='gzip'
                if mode=='chunked':self.headers['Transfer-Encoding']='chunked'
                if mode=='wrong_range':self.headers.replace_header('Content-Range','bytes 0-0/1')
                if mode=='oversized':self.seek(0,2);self.write(b'x');self.seek(0)
        function=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='run')
        loop=next(x for x in ast.walk(function) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='record')
        first=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='url' for t in x.targets))
        end=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='validate_shard')
        scope={**probe.__dict__,'__file__':str(SOURCE),'manifest':{'model_revision':'synthetic'},'name':'synthetic.safetensors','record':{'size':len(payload)},'buffer':buffer,'position':0,'event':lambda **row:events.append(row),'check_host':lambda:{},'save':lambda *args:None}
        if small_chunks:
            spec=importlib.util.spec_from_file_location('small_range_transport',SOURCE.with_name('range_download.py'))
            transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
            transport.CHUNK=5;transport.PROGRESS_STEP=17;scope['download_shard']=transport.download
        with patch('urllib.request.urlopen',side_effect=lambda request,**kw:Response(request)):
            exec(compile(ast.Module(body=loop.body[first:end],type_ignores=[]),str(SOURCE),'exec'),scope)
        return payload,buffer,requests,events

    def test_actual_probe_downloads_four_disjoint_ranges(self):
        payload,buffer,requests,events=self.exercise()
        self.assertEqual(bytes(buffer),payload)
        self.assertEqual(len(requests),4)
        ranges=sorted(tuple(map(int,x.removeprefix('bytes=').split('-'))) for x in requests)
        self.assertEqual(ranges,[(i*len(payload)//4,(i+1)*len(payload)//4-1) for i in range(4)])

    def test_malformed_responses_fail_before_tensor_use(self):
        for mode in ['ignored','duplicate','short','encoding','chunked','wrong_range','oversized']:
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.exercise(mode)

    def test_threaded_aggregate_progress_is_monotonic_and_complete(self):
        payload,buffer,requests,events=self.exercise(small_chunks=True)
        self.assertEqual(bytes(buffer),payload)
        counts=[x['bytes'] for x in events]
        self.assertGreater(len(counts),4)
        self.assertEqual(counts[-1],len(payload))
        self.assertTrue(all(a<b for a,b in zip([0]+counts,counts)))

if __name__=='__main__':unittest.main()
