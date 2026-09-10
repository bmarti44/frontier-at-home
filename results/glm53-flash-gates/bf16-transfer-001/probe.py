"""Bounded HTTP transport diagnostic only; never native weights used for inference."""
import argparse,concurrent.futures,gc,hashlib,json,math,os,subprocess,threading,time,urllib.parse,urllib.request
from pathlib import Path

SIZE=256*1024**2
REVISION='a5b45eb41df6402735dedc900be14a42e8d5e538'
SHARD='model-00115-of-00120.safetensors'
TOTAL=5366143376
BASE='https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/resolve/'+REVISION+'/'+SHARD

def require(ok,message):
    if not ok:raise ValueError(message)

def host():
    mem=dict(x.split(':',1) for x in Path('/proc/meminfo').read_text().splitlines());vm=dict(x.split() for x in Path('/proc/vmstat').read_text().splitlines())
    return {'time_unix':time.time(),'available_kib':int(mem['MemAvailable'].split()[0]),'pswpin':int(vm['pswpin']),'pswpout':int(vm['pswpout']),'used_swap_kib':int(mem['SwapTotal'].split()[0])-int(mem['SwapFree'].split()[0])}

def check_headers(status,headers,left,right,total):
    require(status==206,'range server must return206')
    values={key:(headers.get_all(key,[]) if hasattr(headers,'get_all') else headers.get(key,[])) for key in ['Content-Range','Content-Length','Content-Encoding','Transfer-Encoding']}
    require(all(isinstance(v,list) and len(v)<=1 and all(isinstance(x,str) for x in v) for v in values.values()),'duplicate or malformed framing headers')
    require(values['Transfer-Encoding']==[],'Transfer-Encoding forbidden for exact-length response')
    require(values['Content-Range']==[f'bytes {left}-{right}/{total}'],'exact Content-Range required')
    require(values['Content-Length']==[str(right-left+1)],'exact range Content-Length required')
    require(values['Content-Encoding'] in ([],['identity']),'unexpected content encoding')

def score(rows):
    if isinstance(rows,Path):
        root=rows;rows=[json.loads(x) for x in (root/'raw.jsonl').read_text().splitlines()];r=json.loads((root/'randomness.json').read_text());require(''.join(x['arm'] for x in rows)==('ABBA' if r['seed']%2==0 else 'BAAB'),'seeded transport order')
    require(len(rows)==4,'all four transport arms required')
    require([x['arm'] for x in rows] in (list('ABBA'),list('BAAB')),'matched transport order')
    digests=[]
    for row in rows:
        require(row['status']=='COMPLETE' and row['bytes']==SIZE,'complete transport payload required')
        require(row['workers']==(1 if row['arm']=='A' else 4),'fixed transport workers')
        require(type(row['elapsed_seconds']) in (int,float) and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds']>0,'elapsed time')
        require(row['host_before']['available_kib']>=110*1024**2 and row['host_after']['available_kib']>=110*1024**2,'transport memory floor')
        require(all(row['host_before'][k]==row['host_after'][k] for k in ['pswpin','pswpout','used_swap_kib']),'transport host swap changed')
        parts=sorted(row['ranges'],key=lambda x:x['left']);require(len(parts)==row['workers'],'range coverage');cursor=0
        for part in parts:
            require(part['left']==cursor and part['right']==cursor+SIZE//row['workers']-1 and part['received']==part['right']-part['left']+1,'range bounds/received count')
            check_headers(part['status'],part['headers'],part['left'],part['right'],TOTAL);cursor=part['right']+1
        require(cursor==SIZE and len(row['sha256'])==64 and all(c in '0123456789abcdef' for c in row['sha256']),'digest/coverage');digests.append(row['sha256'])
    require(len(set(digests))==1,'unequal transport bytes')
    for a,b in zip(rows,rows[1:]):
        require(all(a['host_after'][k]==b['host_before'][k] for k in ['pswpin','pswpout','used_swap_kib']),'between-arm host swap changed')
    a=[x['elapsed_seconds'] for x in rows if x['arm']=='A'];b=[x['elapsed_seconds'] for x in rows if x['arm']=='B']
    return {'verdict':'PASS','scope':'Partial-file HTTP byte equivalence and transport timing diagnostic only; no complete shard or model/quality/speed/context qualification','single_connection_seconds':a,'four_connections_seconds':b,'timing_formula':'Mean single-connection elapsed seconds / mean four-connection elapsed seconds; two observations each, no confidence claim','observed_ratio':sum(a)/sum(b),'paired_payloads_identical':True,'partial_bytes_per_arm':SIZE,'whole_shard_verified':False}

def run(root):
    manifest=json.loads((root/'manifest.json').read_text());r=json.loads((root/'randomness.json').read_text());require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==manifest['probe']['sha256'],'frozen probe changed')
    frozen=manifest['frozen_at_unix'];expected_round=int((frozen-1595431050)//30)+2
    require(r['round']==expected_round and r['publication_unix']==1595431050+(expected_round-1)*30 and r['frozen_at_unix']==frozen,'exact later beacon')
    cmd=[manifest['node'],str(root/'code/scripts/103_verify_drand_receipt_bundle.mjs'),str(r['round']),r['randomness'],r['signature'],r['previous_signature']]
    verified=subprocess.run(cmd,capture_output=True,text=True,timeout=30);require(verified.returncode==0 and verified.stdout=='DRAND_BLS_RECEIPT_OK\n','beacon verification')
    require(r['seed']==int(r['randomness'][:16],16),'seed conversion');order='ABBA' if r['seed']%2==0 else 'BAAB';rows=[]
    for index,arm in enumerate(order):
        workers=1 if arm=='A' else 4;row={'index':index,'arm':arm,'workers':workers,'status':'STARTED','bytes':SIZE,'host_before':host(),'ranges':[]};began=time.monotonic();lock=threading.Lock()
        try:
            require(row['host_before']['available_kib']>=110*1024**2,'transport start memory')
            buffer=bytearray(SIZE)
            def part(number):
                left=number*(SIZE//workers);right=left+SIZE//workers-1
                request=urllib.request.Request(BASE+f'?download=true&range={left}-{right}',headers={'Range':f'bytes={left}-{right}','Accept-Encoding':'identity','User-Agent':'glm53-bounded-transport-diagnostic'})
                item={'left':left,'right':right,'received':0}
                with lock:row['ranges'].append(item)
                with urllib.request.urlopen(request,timeout=30) as response:
                    item.update(status=response.status,headers={k:response.headers.get_all(k,[]) for k in ['Content-Range','Content-Length','Content-Encoding','Transfer-Encoding']},host=urllib.parse.urlsplit(response.url).hostname)
                    check_headers(response.status,item['headers'],left,right,TOTAL)
                    while item['received']<right-left+1:
                        start=left+item['received'];n=response.readinto(memoryview(buffer)[start:min(start+1024**2,right+1)])
                        require(n is not None and n>0,'short range body');item['received']+=n
                    require(response.read(1)==b'','oversized range body')
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(part,range(workers)))
            row.update(elapsed_seconds=time.monotonic()-began,sha256=hashlib.sha256(buffer).hexdigest(),status='COMPLETE');del buffer;gc.collect()
        except BaseException as error:row.update(status='FAIL',error=repr(error),elapsed_seconds=time.monotonic()-began)
        row['host_after']=host();rows.append(row)
        with (root/'raw.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps({'index':index,'arm':arm,'status':row['status'],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
        if row['status']!='COMPLETE':break
    try:summary=score(rows)
    except Exception as error:summary={'verdict':'FAIL','scope':'HTTP transport diagnostic only; no native/model result','error':repr(error)}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    return summary['verdict']=='PASS'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--frozen',type=Path,required=True);raise SystemExit(0 if run(parser.parse_args().frozen) else 1)
