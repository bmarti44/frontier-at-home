"""Exercise the native probe's actual download block with synthetic HTTP bodies."""
import ast,importlib.util,io,json,types,unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

SOURCE=Path(__file__).with_name('probe.py')
spec=importlib.util.spec_from_file_location('native_transfer_probe',SOURCE)
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class Transfer(unittest.TestCase):
    def exercise(self,mode=None):
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
        function=next(x for x in ast.parse(SOURCE.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='run')
        loop=next(x for x in ast.walk(function) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='record')
        first=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='url' for t in x.targets))
        end=next(i for i,x in enumerate(loop.body) if isinstance(x,ast.Assign) and isinstance(x.value,ast.Call) and isinstance(x.value.func,ast.Name) and x.value.func.id=='validate_shard')
        scope={**probe.__dict__,'__file__':str(SOURCE),'manifest':{'model_revision':'synthetic'},'name':'synthetic.safetensors','record':{'size':len(payload)},'buffer':buffer,'position':0,'event':lambda **row:events.append(row),'check_host':lambda:{},'save':lambda *args:None}
        with patch('urllib.request.urlopen',side_effect=lambda request,**kw:Response(request)):
            exec(compile(ast.Module(body=loop.body[first:end],type_ignores=[]),str(SOURCE),'exec'),scope)
        return payload,buffer,requests,events

    def test_actual_probe_downloads_four_disjoint_ranges(self):
        payload,buffer,requests,events=self.exercise()
        self.assertEqual(bytes(buffer),payload)
        self.assertEqual(len(requests),4)
        ranges=sorted(tuple(map(int,x.removeprefix('bytes=').split('-'))) for x in requests)
        self.assertEqual(ranges,[(i*len(payload)//4,(i+1)*len(payload)//4-1) for i in range(4)])

if __name__=='__main__':unittest.main()
