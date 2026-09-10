"""Capture limited HTTP diagnostics under existing lock and fresh user cgroup."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,time
O=Path(sys.argv[1]).resolve();assert Path(__file__).resolve()==O/'code/run.py'
sys.path.insert(0,str(O/'code/scripts/lib'))
from glm53_probe_capture import inference_lock
m=json.loads((O/'manifest.json').read_text())
spec=importlib.util.spec_from_file_location('transport',O/'code/probe.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bindings():
    for row in m['files']:
        path=Path(row['path']);p.require(path.stat().st_size==row['size_bytes'] and digest(path)==row['sha256'],'frozen file changed')
def observe(unit):
    args=['systemctl','--user','show',unit,'-p','LoadState','-p','ActiveState','-p','MainPID','-p','ControlGroup','-p','MemoryHigh','-p','MemoryMax','-p','MemorySwapMax','-p','OOMPolicy','-p','KillMode','-p','RuntimeMaxUSec'];v=subprocess.run(args,capture_output=True,text=True,timeout=5);fields=dict(x.split('=',1) for x in v.stdout.splitlines() if '=' in x)
    row={'time_unix':time.time(),'host':p.host(),'unit':{'returncode':v.returncode,'stdout':v.stdout,'stderr':v.stderr},'fields':fields};pid=int(fields.get('MainPID','0'))
    if pid:
        try:row['process']={'pid':pid,'stat':Path(f'/proc/{pid}/stat').read_text(),'cmdline':Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0'),'executable':os.readlink(f'/proc/{pid}/exe')}
        except OSError as error:row['process_error']=repr(error)
    if fields.get('ControlGroup'):
        cg=Path('/sys/fs/cgroup')/fields['ControlGroup'].lstrip('/')
        try:row['cgroup']={name:(cg/name).read_text() for name in ['memory.current','memory.peak','memory.swap.current','memory.events','cgroup.procs']}
        except OSError as error:row['cgroup_error']=repr(error)
    return row
with inference_lock():
    failure=None;samples=[];process=None;before=p.host();unit=m['unit']+'.service';document_digests={name:digest(O/name) for name in ['manifest.json','randomness.json']}
    try:
        bindings();p.require(before['available_kib']>=110*1024**2,'unloaded start memory');p.require(':8015' not in subprocess.check_output(['ss','-ltn'],text=True),'GLM listener active')
        command=['systemd-run','--user','--wait','--pipe','--collect','--quiet','--unit='+m['unit'],'-p','MemoryHigh=512M','-p','MemoryMax=768M','-p','MemorySwapMax=0','-p','OOMPolicy=kill','-p','KillMode=control-group','-p','RuntimeMaxSec=180s','-p','CPUQuota=200%','/usr/bin/env','-i','HOME='+str(O/'state'),'LANG=C.UTF-8','PATH=/usr/bin:/bin',m['python'],'-I','-B',str(O/'code/probe.py'),'--frozen',str(O)]
        save(O/'invocation.json',{'command':command,'time_unix':time.time(),'host_before':before,'documents':{k:{'sha256':v} for k,v in document_digests.items()}})
        with (O/'stdout').open('w') as out,(O/'stderr').open('w') as err:
            process=subprocess.Popen(command,stdout=out,stderr=err)
            while process.poll() is None:
                row=observe(unit);samples.append(row)
                with (O/'host.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                time.sleep(.5)
            p.require(process.wait()==0,'transport process failed')
        rows=[json.loads(x) for x in (O/'raw.jsonl').read_text().splitlines()];scored=p.score(rows);r=json.loads((O/'randomness.json').read_text());p.require(''.join(x['arm'] for x in rows)==('ABBA' if r['seed']%2==0 else 'BAAB'),'seeded transport order')
        live=[s for s in samples if s['fields'].get('ActiveState')=='active'];p.require(live,'no live containment observation')
        for s in live:
            p.require(s['fields']['MemoryMax']==str(768*1024**2) and s['fields']['MemoryHigh']==str(512*1024**2) and s['fields']['MemorySwapMax']=='0' and s['fields']['OOMPolicy']=='kill' and s['fields']['KillMode']=='control-group','containment differs')
            p.require(s.get('process',{}).get('executable')==m['python'] and s['process']['cmdline']==[m['python'],'-I','-B',str(O/'code/probe.py'),'--frozen',str(O),''],'process identity missing or changed')
            cg=s.get('cgroup',{});p.require(cg and int(cg['memory.swap.current'])==0 and all(int(line.split()[1])==0 for line in cg['memory.events'].splitlines()),'cgroup swap/limit event or missing sample')
        for s in samples:
            h=s['host'];p.require(h['available_kib']>=110*1024**2 and all(h[k]==before[k] for k in ['pswpin','pswpout','used_swap_kib']),'observed host memory/swap failure')
        save(O/'transport-score.json',scored)
    except BaseException as error:failure=repr(error)
    finally:
        if process is not None and process.poll() is None:
            subprocess.run(['systemctl','--user','stop',unit],timeout=30,capture_output=True);process.wait(timeout=10)
        terminal=observe(unit);save(O/'terminal.json',terminal)
        try:
            bindings();p.require(all(digest(O/name)==sha for name,sha in document_digests.items()),'launch document changed')
            p.require(terminal['fields'].get('MainPID')=='0' and terminal['fields'].get('ActiveState') in ['inactive','failed'] and not terminal['fields'].get('ControlGroup'),'surviving transport cgroup')
            last=p.host();p.require(all(last[k]==before[k] for k in ['pswpin','pswpout','used_swap_kib']),'terminal host swap changed')
        except BaseException as error:failure=failure or repr(error)
        save(O/'controller-summary.json',{'verdict':'FAIL' if failure else 'PASS','scope':'Partial-file transport diagnostic only; never model qualification','failure':failure,'observed_samples':len(samples),'exit_code':None if process is None else process.returncode,'time_unix':time.time()})
print(json.dumps(json.loads((O/'controller-summary.json').read_text())),flush=True)
