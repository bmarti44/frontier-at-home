"""Four disjoint HTTP ranges into one caller-owned shard buffer; no tensor imports."""
import concurrent.futures,threading,urllib.parse,urllib.request

CHUNK=1024**2
PROGRESS_STEP=256*1024**2

def require(ok,message):
    if not ok:raise ValueError(message)

def check_headers(status,headers,left,right,total):
    require(status==206,'range server must return206')
    values={key:(headers.get_all(key,[]) if hasattr(headers,'get_all') else headers.get(key,[])) for key in ['Content-Range','Content-Length','Content-Encoding','Transfer-Encoding']}
    require(all(isinstance(v,list) and len(v)<=1 and all(isinstance(x,str) for x in v) for v in values.values()),'duplicate or malformed framing headers')
    require(values['Transfer-Encoding']==[],'Transfer-Encoding forbidden for exact-length response')
    require(values['Content-Range']==[f'bytes {left}-{right}/{total}'],'exact Content-Range required')
    require(values['Content-Length']==[str(right-left+1)],'exact range Content-Length required')
    require(values['Content-Encoding'] in ([],['identity']),'unexpected content encoding')

def download(buffer,url,progress):
    size=len(buffer);require(size>=4,'four nonempty ranges required')
    lock=threading.Lock();received=0;next_report=PROGRESS_STEP
    def part(number):
        nonlocal received,next_report
        left=number*size//4;right=(number+1)*size//4-1
        request=urllib.request.Request(url+f'&range={left}-{right}',headers={'Range':f'bytes={left}-{right}','Accept-Encoding':'identity','User-Agent':'glm53-bounded-reference-feasibility'})
        row={'left':left,'right':right,'received':0}
        with urllib.request.urlopen(request,timeout=30) as response:
            row.update(status=response.status,headers={k:response.headers.get_all(k,[]) for k in ['Content-Range','Content-Length','Content-Encoding','Transfer-Encoding']},host=urllib.parse.urlsplit(response.url).hostname)
            check_headers(response.status,row['headers'],left,right,size)
            while row['received']<right-left+1:
                start=left+row['received'];n=response.readinto(memoryview(buffer)[start:min(start+CHUNK,right+1)])
                require(n is not None and n>0,'short range body');row['received']+=n
                with lock:
                    received+=n
                    if received>=next_report or received==size:
                        progress(received);next_report=received+PROGRESS_STEP
            require(response.read(1)==b'','oversized range body')
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(part,range(4)))
    require(received==size,'incomplete full-shard transfer')
    return {'workers':4,'bytes':received,'ranges':rows,'whole_shard_hash_required_before_use':True}
