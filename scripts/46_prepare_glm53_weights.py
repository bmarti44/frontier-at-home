#!/usr/bin/env python3
"""Download pinned GLM weights, preserving selected bytes without requantization."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('/home/bmarti44/.cache/glm53-flash')
CHUNK = 8 << 20


def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n'); temporary.replace(path)


def stream_selected(source, target_fd, segments, expected_size, expected_sha256):
    """Hash the complete source while copying only declared intervals with pwrite."""
    segments = sorted(segments)
    if any(a < 0 or b <= a or b > expected_size or out < 0 for a,b,out in segments):
        raise ValueError('invalid selected source interval')
    h = hashlib.sha256(); offset = 0; pending = 0; active = []; written = 0
    while block := source.read(CHUNK):
        end = offset + len(block)
        if end > expected_size: raise ValueError('source exceeds pinned size')
        h.update(block)
        while pending < len(segments) and segments[pending][0] < end:
            active.append(segments[pending]); pending += 1
        for first, last, output in active:
            a, b = max(first, offset), min(last, end)
            if a < b:
                data = memoryview(block)[a-offset:b-offset]
                position = output + a - first
                while data:
                    n = os.pwrite(target_fd, data, position)
                    if n <= 0: raise OSError('short output write')
                    written += n; position += n; data = data[n:]
        active = [row for row in active if row[1] > end]
        offset = end
    if offset != expected_size or h.hexdigest() != expected_sha256:
        raise ValueError('complete upstream shard size or SHA-256 mismatch')
    if written != sum(b-a for a,b,_ in segments): raise ValueError('selected byte coverage mismatch')
    return written


def selected_digest(fd, segments):
    h = hashlib.sha256()
    for first,last,out in sorted(segments):
        for at in range(0,last-first,CHUNK):
            count=min(CHUNK,last-first-at); data=os.pread(fd,count,out+at)
            if len(data)!=count: raise ValueError('short selected output')
            h.update(data)
    return h.hexdigest()


def range_groups(segments):
    groups=[]
    for a,b,_ in sorted(segments):
        if groups and a-groups[-1][1]<=(8<<20) and max(b,groups[-1][1])-groups[-1][0]<=(128<<20):
            groups[-1][1]=max(b,groups[-1][1])
        else:groups.append([a,b])
    return groups


def download_dense_ranges(url, sink, segments, size):
    """Read pinned selected ranges twice; do not claim a whole-shard hash."""
    ranges=[];written=0
    for first,last in range_groups(segments):
        def fetch(arm):
            request=urllib.request.Request(url+f'&selected_range={first}-{last-1}&verification={arm}',
                headers={'Range':f'bytes={first}-{last-1}'})
            with urllib.request.urlopen(request,timeout=120) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {first}-{last-1}/{size}':
                    raise ValueError('exact pinned byte range required')
                data=response.read(last-first+1)
            if len(data)!=last-first:raise ValueError('short or excessive dense range')
            return data
        data=fetch(1);reference=fetch(2)
        if data!=reference:raise ValueError('independent dense range reads differ')
        source_digest=hashlib.sha256(data).hexdigest();del reference
        selected=[(a-first,b-first,out) for a,b,out in segments if first<=a and b<=last]
        written+=stream_selected(io.BytesIO(data),sink,selected,last-first,source_digest)
        ranges.append({'first':first,'last_exclusive':last,'sha256':source_digest})
    if written!=sum(b-a for a,b,_ in segments):raise ValueError('dense range coverage mismatch')
    return written,ranges


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,choices=(1,2,4),default=4)
    parser.add_argument('--source-cache',type=Path,help='Optional complete upstream shards under k2/; every shard is still hash-verified')
    args=parser.parse_args(); target=args.output.resolve()
    layout=BASE/'model-layout-001'
    sys.path.insert(0,str(ROOT/'scripts/lib'))
    from glm53_contract import verify_inventory
    verify_inventory(layout,json.loads((ROOT/'results/glm53-flash-gates/model-layout-001/inventory.json').read_text()))
    overlay_path=BASE/'build-source-005/vllm-exl3/tools/dense_overlay.py'
    lock=json.loads((ROOT/'configs/build-manifests/glm53-flash-sources.json').read_text())
    overlay_pin=next(row['sha256'] for row in lock['sources']['vllm-exl3']['files'] if row['path']=='tools/dense_overlay.py')
    if digest(overlay_path)!=overlay_pin:
        raise ValueError('overlay implementation changed')
    spec=importlib.util.spec_from_file_location('pinned_overlay',overlay_path)
    overlay=importlib.util.module_from_spec(spec);spec.loader.exec_module(overlay)
    plan=json.loads((layout/'overlay-plan.json').read_text())
    replaced={e['name'] for e in plan['plan']}
    metadata={label:json.loads((layout/label/'api.json').read_text()) for label in ('k2','dense')}
    target.mkdir(parents=True,exist_ok=True); receipts=target/'download-receipts';receipts.mkdir(exist_ok=True)
    jobs=[]; wmap={}; headers={}
    def header_file(name,entries):
        import io
        buffer=io.BytesIO(); header=overlay.write_header(buffer,entries,{'format':'pt'})
        prefix=buffer.getvalue(); size=len(prefix)+max((m['data_offsets'][1] for n,m in header.items() if n!='__metadata__'),default=0)
        headers[name]=(prefix,size); return header,len(prefix)
    for filename in sorted(set(json.loads((layout/'k2/model.safetensors.index.json').read_text())['weight_map'].values())):
        raw=(layout/'k2'/(filename+'.header.json')).read_bytes(); source_header=json.loads(raw)
        chosen=sorted([(n,m) for n,m in source_header.items() if n!='__metadata__' and n not in replaced and not n.startswith('model.language_model.layers.45.')],key=lambda item:item[1]['data_offsets'][0])
        if not chosen: continue
        output_name='selected-'+filename
        header,start=header_file(output_name,[(n,m['dtype'],m['shape']) for n,m in chosen])
        segments=[(8+len(raw)+m['data_offsets'][0],8+len(raw)+m['data_offsets'][1],start+header[n]['data_offsets'][0]) for n,m in chosen]
        jobs.append(('k2',filename,output_name,segments))
        wmap.update({n:output_name for n,_ in chosen})
    dense_outputs=overlay.plan_outputs(plan['plan']); dense_name='dense-overlay.safetensors'
    header,start=header_file(dense_name,[(n,d,s) for n,d,s,_,_ in dense_outputs]); dense_segments={}
    for name,_,_,entry,part in dense_outputs:
        info=entry['parts'][part]; first,last=overlay.tensor_range(info['hlen'],info['meta']);last+=1
        out=start+header[name]['data_offsets'][0]; segments=dense_segments.setdefault(info['file'],[])
        if entry['nblocks']==1 or part in ('suh',entry['marker']): segments.append((first,last,out))
        elif part=='svh':
            width=entry['out']*2; segments.append((first+entry['block']*width,first+(entry['block']+1)*width,out))
        else:
            rows,cols,words=info['meta']['shape']; stride=cols*words*2; width=stride//entry['nblocks']
            for row in range(rows):
                a=first+row*stride+entry['block']*width;segments.append((a,a+width,out+row*width))
        wmap[name]=dense_name
    jobs.extend(('dense',filename,dense_name,segments) for filename,segments in sorted(dense_segments.items()))
    for name,(prefix,size) in headers.items():
        coverage=sorted((out,out+b-a) for _,_,destination,segments in jobs if destination==name for a,b,out in segments)
        end=len(prefix)
        for first,last in coverage:
            if first!=end:raise ValueError('output ranges overlap or leave a gap')
            end=last
        if end!=size:raise ValueError('output ranges do not cover the complete shard')
    selected_bytes=sum(size-len(prefix) for prefix,size in headers.values())
    if selected_bytes!=84696019172 or len(wmap)!=147690: raise ValueError('selection differs from frozen census')
    required=sum(size for _,size in headers.values())
    existing=sum(min((target/name).stat().st_blocks*512,size) for name,(_,size) in headers.items() if (target/name).exists())
    if shutil.disk_usage(target).free < required-existing+(8<<30): raise ValueError('insufficient disk reserve')
    binding={'script_sha256':digest(__file__),'layout_sha256':digest(layout/'overlay-plan.json'),'payload_bytes':selected_bytes,'tensors':len(wmap),'jobs':len(jobs)}
    previous=target/'download-manifest.json'
    if previous.exists():
        prior=json.loads(previous.read_text())
        if {k:v for k,v in prior.items() if k!='script_sha256'}!={k:v for k,v in binding.items() if k!='script_sha256'}:
            raise ValueError('download selection changed')
        if prior!=binding:
            with (target/'download-script-history.jsonl').open('a') as history:
                history.write(json.dumps({'time_unix':time.time(),'previous':prior,'current':binding})+'\n')
    write_json(previous,binding)
    for name,(prefix,size) in headers.items():
        p=target/name
        if p.exists():
            with p.open('rb') as f:
                if f.read(len(prefix))!=prefix or p.stat().st_size!=size: raise ValueError('existing shard layout changed')
        else:
            with p.open('xb') as f: f.write(prefix);f.truncate(size)
    def download(job):
        label,filename,out,segments=job; api=metadata[label]
        entry=next(row for row in api['siblings'] if row['rfilename']==filename)
        record_path=receipts/(label+'-'+filename+'.json')
        with (target/out).open('r+b',buffering=0) as sink:
            if record_path.exists():
                record=json.loads(record_path.read_text())
                if record['upstream_sha256']!=entry['lfs']['sha256'] or record['selected_sha256']!=selected_digest(sink.fileno(),segments):
                    raise ValueError('resumed selected bytes differ from receipt')
                print(json.dumps({'event':'resumed','pack':label,'file':filename}),flush=True);return
            started=time.time();url=f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{filename}?full_verified_download=1&request={time.time_ns()}"
            print(json.dumps({'event':'start','pack':label,'file':filename,'time_unix':started}),flush=True)
            ranges=None
            if label=='dense':written,ranges=download_dense_ranges(url,sink.fileno(),segments,entry['size'])
            elif args.source_cache is not None and (args.source_cache/label/filename).is_file():
                with (args.source_cache/label/filename).open('rb') as source:
                    written=stream_selected(source,sink.fileno(),segments,entry['size'],entry['lfs']['sha256'])
            else:
                with urllib.request.urlopen(url,timeout=120) as response:
                    if response.status!=200: raise ValueError('full shard response required')
                    written=stream_selected(response,sink.fileno(),segments,entry['size'],entry['lfs']['sha256'])
            os.fsync(sink.fileno())
            record={'pack':label,'file':filename,'source_bytes':entry['size'],'selected_bytes':written,'upstream_sha256':entry['lfs']['sha256'],'selected_sha256':selected_digest(sink.fileno(),segments),'start_unix':started,'end_unix':time.time()}
            record['verification']='full_upstream_sha256' if ranges is None else 'pinned_selected_ranges_read_twice; whole upstream hash is metadata only'
            if ranges is not None:record['ranges']=ranges
            write_json(record_path,record);print(json.dumps({'event':'complete',**record}),flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(download,job):job for job in jobs}
        for future in as_completed(futures):
            try: future.result()
            except Exception as error:
                failure={'time_unix':time.time(),'pack':futures[future][0],'file':futures[future][1],'failure':repr(error)}
                failures.append(failure)
                with (target/'download-failures.jsonl').open('a') as log:log.write(json.dumps(failure)+'\n')
                print(json.dumps({'event':'failure',**failure}),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} downloads failed; completed shard receipts can resume')
    for filename in ('config.json','generation_config.json','processor_config.json','tokenizer.json','tokenizer_config.json','chat_template.jinja','quantization_config.json','LICENSE'):
        api=metadata['k2'];entry=next(row for row in api['siblings'] if row['rfilename']==filename)
        with urllib.request.urlopen(f"https://huggingface.co/{api['id']}/resolve/{api['sha']}/{filename}",timeout=120) as response: data=response.read(entry['size']+1)
        if len(data)!=entry['size']:raise ValueError('metadata size mismatch')
        actual=hashlib.sha256(data).hexdigest() if 'lfs' in entry else hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if actual!=entry.get('lfs',{}).get('sha256',entry['blobId']): raise ValueError('metadata digest mismatch')
        (target/filename).write_bytes(data)
    cfg=json.loads((target/'config.json').read_text())
    cfg.setdefault('quantization_config',{})['non_routed_exl3']={'codebook':plan['plan'][0]['marker'],'source':metadata['dense']['id']+'@'+metadata['dense']['sha'],'layers':plan['fork_keys']}
    write_json(target/'config.json',cfg)
    write_json(target/'model.safetensors.index.json',{'metadata':{'total_size':selected_bytes},'weight_map':dict(sorted(wmap.items()))})
    inventory=[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(target.iterdir()) if p.is_file() and p.name!='inventory.json']
    write_json(target/'inventory.json',{'schema_version':1,'files':inventory})
    print(json.dumps({'verdict':'PASS','output':str(target),'payload_bytes':selected_bytes,'tensors':len(wmap)}),flush=True)


if __name__=='__main__':main()
