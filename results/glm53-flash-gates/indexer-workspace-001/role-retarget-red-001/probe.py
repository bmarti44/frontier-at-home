"""Evidence-only post-projection workspace comparison; no model or serving path."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
REPO=Path('/home/bmarti44/spark-deepseek-v4-flash')
PRIOR_FIXTURE=HERE/'prior_fixture.py' if (HERE/'prior_fixture.py').exists() else REPO/'scripts/lib/glm53_indexer_fixture.py'
CASES={'prefill-4':[128]*4,'prefill-1':[512]}
WORKSPACE=1385168896
CACHE_BYTES=4930*8448
TAIL_BYTES=145*2*4*128*2
TOPK_BYTES=512*2048*4
BASE_BYTES=WORKSPACE+CACHE_BYTES+TAIL_BYTES+TOPK_BYTES
SAFETY={'minimum_start_gib':110,'kill_floor_gib':40,'timeout_seconds':600,'memory_high_gib':32,'memory_max_gib':34}

def require(value,message):
    if not value:raise ValueError(message)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read_json(path):
    def num(v):
        x=float(v);require(math.isfinite(x),'nonfinite JSON');return x
    return json.loads(Path(path).read_bytes(),parse_float=num,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def write(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

REQUIRED_CODE={'probe.py','native.py','run.py','freeze.py','test_probe.py','test_review.py','CORRECTION.md','PROTOCOL.md','prior_fixture.py','prior_probe45.py',
    'scripts/38_guard_glm53_probe.py','scripts/103_verify_drand_receipt_bundle.mjs',
    'scripts/lib/glm53_contract.py','scripts/lib/glm53_probe_capture.py','scripts/lib/glm53_host_evidence.py'}

def verify_frozen(root,manifest):
    require(manifest['safety']==SAFETY and manifest['cases']==CASES and manifest['budgets_mib']==[512,64],'frozen scope')
    rows=manifest['files'];require(isinstance(rows,list),'frozen file list')
    paths=[x['path'] for x in rows];require(len(set(paths))==len(paths),'duplicate frozen binding')
    dependencies=manifest.get('external_dependencies')
    require(isinstance(dependencies,dict) and set(dependencies)=={'safe_wrapper','memory_guard','libc','nvcc','cxx'},'external dependency roles')
    runtime=manifest.get('runtime');require(isinstance(runtime,dict),'runtime selection missing')
    selected=[manifest.get(k) for k in ('python','node','wrapper')]+[runtime.get('inventory'),runtime.get('root')]+list(dependencies.values())
    require(all(isinstance(x,str) and Path(x).is_absolute() and str(Path(x))==x for x in selected),'noncanonical external selection')
    required_external=set(selected)-{runtime['root']}
    require(required_external<=set(paths),'selected external dependency omitted')
    runtime_root=Path(runtime['root']);python=Path(manifest['python']);inventory=Path(runtime['inventory'])
    require(python==runtime_root/'bin/python3' and inventory==root/'metadata/runtime-inventory.json','interpreter/runtime inventory selection mismatch')
    environment=manifest.get('environment')
    require(isinstance(environment,dict) and environment.get('DG_JIT_NVCC_COMPILER')==dependencies['nvcc'],'compiler selection differs from frozen dependency')
    require({str(root/'code'/name) for name in REQUIRED_CODE}|{str(root/'metadata/config.json'),str(root/'metadata/runtime-inventory.json'),str(root/'cache-template-inventory.json')}<=set(paths),'required frozen source omitted')
    for folder in ('code','metadata','cache-template'):
        directory=root/folder;require(directory.is_dir() and not directory.is_symlink(),'frozen directory missing')
        actual=set()
        for p in directory.rglob('*'):
            require(not p.is_symlink(),'frozen tree symlink')
            if p.is_file():actual.add(str(p))
        require(actual=={x for x in paths if Path(x).is_relative_to(directory)},'frozen local file coverage')
    for row in rows:
        require(set(row)=={'path','size_bytes','sha256'},'frozen binding schema')
        p=Path(row['path']);require(p.is_file() and p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256'],'frozen file changed: '+str(p))
    # The selected interpreter must be the same object bytes covered by the
    # normalized inventory that the controller passes to the closed verifier.
    normalized=read_json(inventory)
    require(normalized.get('schema_version')==1 and isinstance(normalized.get('files'),list),'runtime inventory schema')
    interpreter_rows=[x for x in normalized['files'] if x.get('path')=='bin/python3']
    require(len(interpreter_rows)==1,'selected interpreter inventory coverage')
    interpreter_binding=next(x for x in rows if x['path']==str(python))
    require(all(interpreter_rows[0].get(k)==interpreter_binding[k] for k in ('sha256','size_bytes')),'interpreter differs from runtime inventory')

def bind_host_identity(root,host_scorer,expected):
    """Bind the exact identity files consumed by the unchanged host scorer."""
    paths={'raw':root/'identity/raw.jsonl','summary':root/'identity/summary.json'}
    before={key:{'sha256':sha(path)} for key,path in paths.items()}
    result=host_scorer(root,expected)
    require(before=={key:{'sha256':sha(path)} for key,path in paths.items()},'identity files changed during host scoring')
    require(result.get('verdict')=='PASS','host identity not verified')
    return {**result,'identity_binding':before}


def join_identity(checks,armroot,host_result,inner):
    """Join already validated inner rows to the exact verified guard interval."""
    bindings=host_result['identity_binding'];raw_path=armroot/'identity/raw.jsonl';summary_path=armroot/'identity/summary.json'
    raw=raw_path.read_bytes();summary=read_json(summary_path)
    require(hashlib.sha256(raw).hexdigest()==bindings['raw']['sha256']==summary.get('raw_sha256') and sha(summary_path)==bindings['summary']['sha256'],'identity proof binding changed')
    records=[read_json_line(line) for line in raw.splitlines()]
    require(len(records)>=3 and records[-1].get('event')=='cleanup' and all(x.get('event')=='identity' for x in records[:-1]),'identity record coverage')
    identities=records[:-1]
    require(summary.get('verdict')=='PASS' and summary.get('identity_samples')==host_result['identity_samples']==len(identities),'verified identity count')
    require(all(x.get('pid')==host_result['identity_pid'] and x.get('start_ticks')==host_result['identity_start_ticks'] for x in identities),'verified identity process mismatch')
    terminal=identities[-1];first=identities[0]['time_unix'];last=terminal['time_unix']
    require(terminal.get('completion_verified') is True and terminal.get('terminal_exec_filter_verified') is True,'verified completion identity missing')
    require(all(type(t) in (int,float) and math.isfinite(t) for t in (first,last)) and first<last,'identity interval')
    inner_raw=(checks/'raw.jsonl').read_bytes();require(inner.get('verdict')=='PASS' and hashlib.sha256(inner_raw).hexdigest()==inner['raw_sha256'],'inner rows changed after scoring')
    times=[read_json_line(line).get('time_unix') for line in inner_raw.splitlines()]
    require(len(times)==4 and all(type(t) in (int,float) and math.isfinite(t) and first<=t<=last for t in times),'inner evidence outside verified identity interval')
    require(all(b>a for a,b in zip(times,times[1:])),'inner chronology changed')
    return {'initial_identity_unix':first,'completion_identity_unix':last,'inner_first_unix':times[0],'inner_last_unix':times[-1]}


def fixture_api():
    # A private module namespace, never the serving module or the closed source.
    module=load_module('workspace_private_fixture',PRIOR_FIXTURE)
    module.CASES={key:list(value) for key,value in CASES.items()}
    return module

fixture=fixture_api()

def arm_schedule(seed):
    fixture.case_order(seed)
    cases=list(CASES);random.Random(seed ^ 0x57534C).shuffle(cases)
    return [{'case':case,'arm':arm,'budget_mib':budget} for arms,budget in [(('baseline-a','baseline-b'),512),(('candidate',),64)] for case in cases for arm in arms]

def execute_schedule(seed,execute,compare):
    """A failing repeated baseline cannot admit either candidate arm."""
    baselines={};results=[]
    for arm in arm_schedule(seed):
        result=execute(arm);results.append(result)
        if arm['arm']=='baseline-a':baselines[arm['case']]=result
        else:compare(baselines[arm['case']],result)
    return results

def call_specs(case,budget):
    require(case in CASES and type(budget)is int and budget in (512,64),'unsupported case/budget')
    if case=='prefill-4':
        if budget==512:return [{'start':0,'stop':512,'columns':262144,'requests':[0,4],'skip_gather':False}]
        return [{'start':i*128,'stop':(i+1)*128,'columns':65536,'requests':[i,i+1],'skip_gather':False} for i in range(4)]
    return [{'start':0,'stop':512,'columns':65536,'requests':[0,1],'skip_gather':False}] if budget==512 else [
        {'start':i*256,'stop':(i+1)*256,'columns':65536,'requests':[0,1],'skip_gather':i>0} for i in range(2)]

def bounds(seed,case,budget,index):
    import numpy as np
    spec=call_specs(case,budget)[index];cfg=fixture.case_config(seed,case)
    row=np.arange(spec['start'],spec['stop']);local_req=row//128-spec['requests'][0] if case=='prefill-4' else np.zeros(len(row),dtype='int64')
    lo=(local_req*65536).astype('<i4');return lo,lo+((cfg['positions'][row]+1)//4).astype('<i4')

def geometry(seed,case,budget):
    return {'seed':seed,'case':case,'budget_mib':budget,'query_lengths':CASES[case],'context_per_request':262144,
            'configured_slots':4,'scheduler_tokens':512,'heads':32,'head_dim':128,'topk_tokens':2048,'pool_tokens':4,
            'gather_rows':10485760,'pinned_staging':'probe_owned_persistent_buffers','actual_model_input_tokens':0}

def expected_metadata(seed,case,budget):
    cfg=fixture.case_config(seed,case);blocks=fixture.physical_blocks(seed);common,_=fixture.tables(seed,case);table=(common[:,::4]//4).tolist()
    slots=[];tails=[]
    for r,p in zip(cfg['row_requests'],cfg['positions']):
        r,p=int(r),int(p);block,offset=divmod(p//4,2176);slots.append(blocks[r][block]*2176+offset if p%4==3 else -1);tails.append(blocks[r][31]*4+p%4)
    return {'num_decodes':0,'num_decode_tokens':0,'num_prefills':len(CASES[case]),'num_prefill_tokens':512,
            'index_slots':slots,'tail_slots':tails,'chunks':[{'start':s['start'],'stop':s['stop'],'total_seq_lens':s['columns'],
                'skip_gather':s['skip_gather'],'block_table':table[s['requests'][0]:s['requests'][1]],
                'ks':bounds(seed,case,budget,i)[0].tolist(),'ke':bounds(seed,case,budget,i)[1].tolist()} for i,s in enumerate(call_specs(case,budget))]}

def expected_inputs(seed,case):
    import numpy as np
    cfg=fixture.case_config(seed,case);cache,tail=fixture.cache_and_tail(seed,case)
    q=np.zeros((512,32,128),dtype='uint8');q[:,:,0]=0x38
    keys=np.empty((512,128),dtype='<u2')
    for i,(r,p) in enumerate(zip(cfg['row_requests'],cfg['positions'])):keys[i]=fixture.bf16_bits(fixture.raw_value(int(r),int(p)%4))
    arrays={'cache':cache,'tail':tail,'keys':keys,'query':q,'weights':np.full((512,32),1/32,dtype='<f4'),
            'gate':np.zeros((512,128),dtype='<u2'),'ape':np.zeros((4,128),dtype='<f4'),'hidden':np.zeros((512,4096),dtype='<u2')}
    return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}

def file_stream(path,size,digest):
    identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
    initial=path.lstat();require(stat.S_ISREG(initial.st_mode) and not path.is_symlink() and initial.st_size<=size+1048576,'invalid artifact')
    require(sha(path)==digest,'artifact digest')
    with gzip.open(path,'rb') as f:
        remaining=size
        while remaining:
            data=f.read(min(1024**2,remaining));require(data,'truncated artifact');remaining-=len(data);yield data
        require(not f.read(1),'extra artifact bytes')
    require(identity(path.lstat())==identity(initial) and sha(path)==digest,'artifact changed')

def compare_files(left,right,size):
    a=file_stream(left,size,sha(left));b=file_stream(right,size,sha(right))
    from itertools import zip_longest
    for x,y in zip_longest(a,b):require(x==y,'ordered artifact bytes differ')

def valid_logit_bytes(seed,case):
    cfg=fixture.case_config(seed,case)
    return int(((cfg['positions']+1)//4).sum())*4

def artifact_sizes(seed,case):
    return {'indices.i32.gz':TOPK_BYTES,'cache.u8.gz':CACHE_BYTES,'tail.bf16.gz':TAIL_BYTES,'logits.f32.gz':valid_logit_bytes(seed,case)}

def validate_memory(value):
    require(set(value)=={'cuda_allocated','cuda_reserved','cuda_peak_allocated','cuda_peak_reserved','device_free','device_total','workspace_bytes'},'memory fields')
    require(all(type(v)is int and v>0 for v in value.values()),'memory type/value')
    require(value['workspace_bytes']==WORKSPACE and BASE_BYTES<=value['cuda_allocated']<=value['cuda_peak_allocated']<=value['cuda_peak_reserved']<=value['device_total'],'memory lower/peak/device bound')
    require(value['cuda_allocated']<=value['cuda_reserved']<=value['cuda_peak_reserved'] and value['device_free']<=value['device_total'],'memory cross counters')
    require(value['cuda_peak_reserved']<=34*1024**3,'probe containment allocation cap')

def validate_rows(rows,seed,case,budget):
    require(len(rows)==4,'raw row coverage')
    times=[r.get('time_unix') for r in rows]
    require(all(type(t) in (int,float) and math.isfinite(t) and t>0 for t in times) and all(b>a for a,b in zip(times,times[1:])),'raw chronology')
    require(rows[0]=={'time_unix':times[0],'event':'configured',**geometry(seed,case,budget)},'configuration receipt')
    profile=rows[1];require(set(profile)=={'time_unix','event','workspace_bytes','cuda_peak_allocated','cuda_reserved','device_total'} and profile['event']=='profiled','profile schema')
    require(all(type(profile[k])is int for k in ('workspace_bytes','cuda_peak_allocated','cuda_reserved','device_total')) and profile['workspace_bytes']==WORKSPACE and BASE_BYTES+budget*1024**2<=profile['cuda_peak_allocated']<=profile['cuda_reserved']<=profile['device_total'],'profile capacity')
    require(rows[2]=={'event':'start','case':case,'time_unix':times[2]},'start receipt')
    row=rows[3];require(set(row)=={'event','time_unix','case','budget_mib','artifacts','calls','metadata','memory','cuda_elapsed_ms','cache_stride','tail_stride','output_alias','input_digests'},'output schema')
    require(row['event']=='output' and row['case']==case and type(row['budget_mib'])is int and row['budget_mib']==budget,'output selection')
    require(row['metadata']==expected_metadata(seed,case,budget) and row['input_digests']==expected_inputs(seed,case),'metadata/input identity')
    require(row['cache_stride']==[8448,132,1] and row['tail_stride']==[1024,512,128,1] and row['output_alias'] is True,'storage layout')
    require(type(row['cuda_elapsed_ms']) in (int,float) and math.isfinite(row['cuda_elapsed_ms']) and row['cuda_elapsed_ms']>0,'CUDA observation')
    validate_memory(row['memory']);specs=call_specs(case,budget);require(len(row['calls'])==len(specs),'call coverage')
    previous=0
    for call,spec in zip(row['calls'],specs):
        require(set(call)==set(spec)|{'cuda_allocated_at_return'} and all(call[k]==v for k,v in spec.items()),'call geometry')
        size=(spec['stop']-spec['start'])*spec['columns']*4
        require(type(call['cuda_allocated_at_return'])is int and BASE_BYTES+previous+size<=call['cuda_allocated_at_return']<=row['memory']['cuda_peak_allocated'],'native overlapping allocation receipt')
        previous=size
    require(set(row['artifacts'])==set(artifact_sizes(seed,case)) and all(isinstance(v,str) and len(v)==64 for v in row['artifacts'].values()),'artifact coverage')
    return row

def score_arm(root,seed,case,budget):
    import numpy as np
    rows=[read_json_line(x) for x in (root/'raw.jsonl').read_bytes().splitlines()];row=validate_rows(rows,seed,case,budget)
    require({p.name for p in root.iterdir()}=={'raw.jsonl','traceback.log','summary.json','manifest.json',*artifact_sizes(seed,case)},'arm file coverage')
    blobs={}
    for name,size in artifact_sizes(seed,case).items():
        if name!='logits.f32.gz':blobs[name]=b''.join(file_stream(root/name,size,row['artifacts'][name]))
    fixture.score_indices(np.frombuffer(blobs['indices.i32.gz'],dtype='<i4').reshape(512,2048),seed,case)
    cache,tail=fixture.cache_and_tail(seed,case,final=True)
    require(blobs['cache.u8.gz']==cache.tobytes() and blobs['tail.bf16.gz']==tail.tobytes(),'native cache/tail bytes')
    cfg=fixture.case_config(seed,case);path=root/'logits.f32.gz';require(sha(path)==row['artifacts'][path.name],'logit hash')
    # Validate a canonical global-row stream independently of native chunk boundaries.
    with gzip.open(path,'rb') as f:
        for request,position,old in zip(cfg['row_requests'],cfg['positions'],cfg['history_counts']):
            expected=fixture.valid_logits(seed,int(request),int(old),(int(position)+1)//4).tobytes();require(f.read(len(expected))==expected,'valid logit bytes')
        require(not f.read(1),'extra logits')
    # Exhaust generic identity/length validator too, including every compressed byte.
    for _ in file_stream(path,valid_logit_bytes(seed,case),row['artifacts'][path.name]):pass
    require(not (root/'traceback.log').read_bytes(),'inner traceback')
    return {'verdict':'PASS','scope':'synthetic native arm only','case':case,'budget_mib':budget,'raw_sha256':sha(root/'raw.jsonl'),'artifacts':row['artifacts'],'memory':row['memory']}

def read_json_line(line):
    return json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def compare_arms(left,right,seed,case):
    for name,size in artifact_sizes(seed,case).items():compare_files(left/name,right/name,size)

def main():
    require(not sys.flags.optimize and sys.dont_write_bytecode,'use unoptimized Python -I -B')
    parser=argparse.ArgumentParser();parser.add_argument('--frozen',type=Path,required=True);parser.add_argument('--arm-index',type=int,required=True);args=parser.parse_args()
    frozen=args.frozen.resolve();m=read_json(frozen/'manifest.json');r=read_json(frozen/'randomness.json');seed=r['seed']
    require(Path(__file__).resolve()==frozen/'code/probe.py' and m['safety']==SAFETY,'frozen probe/safety')
    verify_frozen(frozen,m)
    schedule=arm_schedule(seed);require(0<=args.arm_index<len(schedule),'arm index');arm=schedule[args.arm_index]
    require(os.environ.get('VLLM_SPARSE_INDEXER_MAX_LOGITS_MB')==str(arm['budget_mib']),'startup budget selection')
    output=frozen/'arms'/str(args.arm_index)/'checks';output.mkdir(parents=True,exist_ok=False)
    write(output/'manifest.json',{'seed':seed,**arm,'source_sha256':sha(Path(__file__)),'frozen_manifest_sha256':sha(frozen/'manifest.json'),'randomness_sha256':sha(frozen/'randomness.json')})
    failure=None
    with (output/'raw.jsonl').open('x') as raw,(output/'traceback.log').open('x') as errors:
        def record(row):raw.write(json.dumps({'time_unix':time.time(),**row},allow_nan=False)+'\n');raw.flush()
        try:
            native=load_module('workspace_native',HERE/'native.py')
            native.__dict__.update({'fixture':fixture,'require':require,'WORKSPACE':WORKSPACE,'geometry':lambda s:geometry(s,arm['case'],arm['budget_mib']),
                'expected_metadata':lambda s,c:expected_metadata(s,c,arm['budget_mib']), 'call_specs':lambda c:call_specs(c,arm['budget_mib']),
                'bounds':lambda s,c,i:bounds(s,c,arm['budget_mib'],i),'sha256_file':sha,'expected_inputs':expected_inputs})
            native.run_native(frozen/'metadata',output,seed,arm['case'],arm['budget_mib'],record)
        except Exception as error:failure=repr(error);traceback.print_exc(file=errors);record({'event':'failure','failure':failure})
    write(output/'summary.json',{})
    try:
        if failure:raise ValueError(failure)
        summary=score_arm(output,seed,arm['case'],arm['budget_mib'])
    except Exception as error:summary={'verdict':'FAIL','failure':repr(error),'raw_sha256':sha(output/'raw.jsonl')}
    write(output/'summary.json',summary);print(json.dumps(summary),flush=True);return 0 if summary['verdict']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
