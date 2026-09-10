"""Optional GLM profiles: declarative configuration and identity-bound lifecycle.

No production switch/default state is read or written here.
"""
from contextlib import contextmanager
import fcntl
import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

import profile_resolver as resolver
from glm53_contract import strict_json, verify_inventory, sha256_file

NAMES = {'glm-5.3-flash/cuda-spark-128g-agent-fast',
         'glm-5.3-flash/cuda-spark-128g-1m-experimental'}


def reject_overrides(argv):
    allowed = {'--profile', '--host', '--output', '--api-key-file'}
    switches = {'--start', '--status', '--stop'}
    i = 0
    while i < len(argv):
        flag = argv[i].split('=')[0]
        if flag in allowed: i += 1 if '=' in argv[i] else 2
        elif flag in switches: i += 1
        else: raise ValueError('named profile does not allow launch overrides: ' + flag)


def resolve_profile(name, host_path, output):
    name = name.removesuffix('.json')
    if name not in NAMES: raise ValueError('only named experimental GLM profiles are executable')
    slug, file = name.split('/')
    host = resolver.load_host(host_path)
    if host['host_id'] != 'spark-aba1' or os.getuid() != 1000:
        raise ValueError('this experimental lifecycle requires spark-aba1 owner uid 1000')
    p = resolver.load_profile(slug, file + '.json')
    d = resolver.resolve(p, resolver.load_model(slug), host, run_root=str(output))
    props = d['systemd']['properties']
    required = {'User':'bmarti44','MemoryHigh':'92G','MemoryMax':'94G',
                'MemorySwapMax':'0','OOMPolicy':'kill','KillMode':'control-group',
                'RuntimeMaxSec':'9060s','Type':'exec','Delegate':'no'}
    if props != required:
        raise ValueError('unsupported containment; hardened wrapper requires exact measured envelope')
    if (d['safety']['minimum_start_gib'], d['safety']['kill_floor_gib']) != (110,18):
        raise ValueError('unsupported containment safety floor')
    if d['systemd']['flock'] != '/run/lock/frontier-at-home/inference.lock':
        raise ValueError('unsupported inference lock')
    if d['port'] != 8015 or d['argv'][:4] != ['-I','-B','-m','vllm.entrypoints.openai.api_server']:
        raise ValueError('unsupported native entry point or development port')
    if d['binary'] != '/home/bmarti44/.cache/glm53-flash/native-runtime-002/runtime/bin/python3':
        raise ValueError('unsupported runtime path for this installed launcher')
    return d


def verify_frozen_tree(root, manifest):
    if str(manifest).endswith('.gz'):
        data = json.loads(gzip.decompress(Path(manifest).read_bytes()))
    else: data = strict_json(Path(manifest))
    normalized = {'schema_version':1,'files':[
        {k:row[k] for k in ('path','size_bytes','sha256')} for row in data['files']]}
    return verify_inventory(root, normalized)


def verify_artifacts(snapshot):
    memory = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    if int(memory['MemAvailable'].split()[0]) < 110 * 1024**2:
        raise ValueError('GLM start requires 110 GiB available; stop the active model first')
    # Digest-bind the inventories before parsing any trusted file list.
    for row in snapshot['digest_checks']:
        if set(row) != {'path','sha256'} or sha256_file(Path(row['path'])) != row['sha256']:
            raise ValueError('profile artifact digest mismatch: ' + row['path'])
    by_name = {Path(r['path']).name:Path(r['path']) for r in snapshot['digest_checks']}
    runtime = Path(snapshot['binary']).parent.parent
    model = Path(snapshot['argv'][snapshot['argv'].index('--model')+1])
    for name,root in [('runtime',runtime),('model',model)]:
        verify_frozen_tree(root, by_name[name+'-inventory.json.gz'])


def process_identity(pid):
    path = Path('/proc') / str(pid)
    # comm can contain spaces and parentheses; parse after its final ')'.
    fields = (path/'stat').read_text().rsplit(')',1)[1].split()
    return {'pid':pid,'start_ticks':int(fields[19]),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'command':(path/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
            'executable':os.readlink(path/'exe')}


def open_identity(record):
    fd = os.pidfd_open(record['pid'])
    try:
        if process_identity(record['pid']) != record:
            raise ValueError('process identity mismatch')
        return fd
    except BaseException:
        os.close(fd); raise


def unit_properties(unit):
    result = subprocess.run(['systemctl','--user','show',unit,
        '--property=LoadState,InvocationID,ActiveState,ControlGroup'],
        text=True,capture_output=True,timeout=10)
    props = dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    if result.returncode or props.get('ActiveState') not in {'active','activating','deactivating','inactive','failed'}:
        raise ValueError('cannot observe systemd unit state: ' + unit)
    return props


def check_group_empty(record):
    cgroup = record.get('control_group')
    if cgroup:
        events = Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'cgroup.events'
        if events.exists() and 'populated 1' in events.read_text():
            raise ValueError('model descendants survived stop')


def bind_unit(record, props):
    if not props.get('InvocationID'): return False
    if not record.get('invocation_id'):
        # The unique unit name was derived from this still-verified controller.
        fd=open_identity(record['controller'])
        os.close(fd)
        record['invocation_id']=props['InvocationID']
    if props['InvocationID'] != record['invocation_id']:
        raise ValueError('systemd invocation identity mismatch')
    if props.get('ControlGroup'): record['control_group']=props['ControlGroup']
    return True


def stop_unit(record):
    props = unit_properties(record['unit'])
    if props['ActiveState'] not in ('inactive','failed'):
        if not bind_unit(record,props):
            raise ValueError('cannot observe systemd invocation identity; refusing stop')
        subprocess.run(['systemctl','--user','stop',record['unit']],check=True,timeout=60)
        props=unit_properties(record['unit'])
        if props['ActiveState'] not in ('inactive','failed'):
            raise ValueError('model unit survived stop')
    check_group_empty(record)


def save_record(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.chmod(0o600)
    temporary.replace(path)


def lifecycle_path(name):
    return Path.home()/'.local/state/glm53-profiles'/ (name.split('/')[-1]+'.json')


@contextmanager
def preparing_session(snapshot, output):
    path=lifecycle_path(snapshot['profile_id']);path.parent.mkdir(parents=True,exist_ok=True)
    path.parent.chmod(0o700)
    with path.with_suffix('.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('this profile already has an active launcher')
        if path.exists():
            old=strict_json(path)
            if old.get('unit'):
                props=unit_properties(old['unit'])
                if props['ActiveState'] not in ('inactive','failed'):
                    raise ValueError('existing model unit must be stopped before starting this profile')
                check_group_empty(old)
        record={'profile':snapshot['profile_id'],'output':str(output),'unit':None,
                'launcher':process_identity(os.getpid()),'phase':'preparing','ready':False,
                'startup_deadline':time.monotonic()+snapshot.get('safety',{}).get('startup_timeout_seconds',1800)}
        def cancel(signum, frame): raise SystemExit(128+signum)
        previous={s:signal.signal(s,cancel) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            save_record(path,record)
            yield record
        finally:
            try:
                record['phase']='exited';save_record(path,record)
            finally:
                for signum,handler in previous.items(): signal.signal(signum,handler)


def wait_launcher(record, path, timeout=65):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        with path.with_suffix('.lock').open('a') as lock:
            try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: pass
            else:
                # Lock release is the end of all launch/cleanup work. Never
                # confuse a later run's record with the run being stopped.
                current=strict_json(path)
                if current['launcher'] != record['launcher']:
                    raise ValueError('another profile start replaced the stopped run')
                return
        time.sleep(.1)
    raise TimeoutError('profile launcher did not finish cleanup')


def lifecycle_action(name, action):
    name=name.removesuffix('.json')
    if name not in NAMES: raise ValueError('unknown experimental profile')
    path=lifecycle_path(name)
    if not path.exists():
        # A start may hold the lock immediately before atomic registration.
        if path.with_suffix('.lock').exists():
            with path.with_suffix('.lock').open('a') as lock:
                try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError: raise ValueError('profile start is registering; retry lifecycle action')
        print(json.dumps({'profile':name,'state':'stopped','registered_run':False}));return
    record = strict_json(path)
    if record['profile'] != name: raise ValueError('profile identity mismatch')
    active=False
    if record.get('unit'):
        props=unit_properties(record['unit'])
        active=props['ActiveState'] not in ('inactive','failed')
        if active and record.get('invocation_id'): bind_unit(record,props)
        if not active: check_group_empty(record)
    launcher_live=False
    try: fd=open_identity(record['launcher']);launcher_live=True
    except (ProcessLookupError,FileNotFoundError): fd=None
    try:
        if action=='stop':
            if active and record.get('invocation_id'): stop_unit(record)
            if launcher_live: signal.pidfd_send_signal(fd,signal.SIGTERM)
            wait_launcher(record,path)
            # A preparing controller may have created its unit before cancelling.
            final=strict_json(path)
            if final.get('unit'): stop_unit(final)
            active=launcher_live=False
    finally:
        if fd is not None: os.close(fd)
    state=('running' if record.get('ready') else 'starting') if active or launcher_live else 'stopped'
    print(json.dumps({'profile':name,'state':state,'output':record['output'],'port':8015,'qualified':False}))


def authenticated_ready(snapshot, output):
    key = (output/'api-key').read_text().strip()
    base = 'http://127.0.0.1:' + str(snapshot['port'])
    def request(path, body=None, auth=True):
        headers={'Content-Type':'application/json'}
        if auth: headers['Authorization']='Bearer '+key
        req=urllib.request.Request(base+path,headers=headers,data=None if body is None else json.dumps(body).encode())
        with urllib.request.urlopen(req,timeout=30) as response:
            value=response.read();return json.loads(value) if value else None
    request('/health')
    try: request('/v1/models',auth=False)
    except urllib.error.HTTPError as error:
        if error.code != 401: raise ValueError('unauthenticated endpoint did not reject with 401') from error
    else: raise ValueError('unauthenticated endpoint accepted request')
    models=request('/v1/models')
    if 'glm-5.3-flash' not in [r['id'] for r in models['data']]: raise ValueError('wrong model at profile port')
    result=request('/v1/chat/completions',{'model':'glm-5.3-flash',
        'messages':[{'role':'user','content':'Reply with exactly READY'}],
        'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}})
    choice=result['choices'][0]
    if choice['finish_reason']!='stop' or choice['message'].get('content','').strip()!='READY':
        raise ValueError('profile readiness semantics failed')
    return {'health':True,'auth_rejection':401,'model':'glm-5.3-flash','semantic_completion':True}


def wait_controller(child, record, path):
    # Keep the lifecycle lock and ownership when the service bus temporarily
    # fails. The hardened unit has its own independent wall-clock timeout.
    while True:
        try:
            child.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            try:
                stop_unit(record)
            except (ValueError, subprocess.SubprocessError) as error:
                record['cleanup_failure']=type(error).__name__+': '+str(error)
                save_record(path,record)


def run_contained(command, control, output, snapshot, session=None):
    """Keep controller ownership through readiness, cancellation and cleanup."""
    if session is None:
        with preparing_session(snapshot,output) as record:
            return run_contained(command,control,output,snapshot,record)
    path=lifecycle_path(snapshot['profile_id']);record=session
    if time.monotonic()>record['startup_deadline']: raise TimeoutError('profile preparation timed out')
    cancelled=[]
    def cancel(signum, frame): cancelled.append(signum)
    previous={s:signal.signal(s,cancel) for s in (signal.SIGTERM,signal.SIGINT)}
    child=None
    try:
        with (output/'wrapper.log').open('w') as log:
            if cancelled: raise SystemExit(128+cancelled[0])
            child=subprocess.Popen(command,env=control,stdout=log,stderr=subprocess.STDOUT)
            record.update(unit='glm52-'+command[command.index('--tag')+1]+'-'+str(child.pid)+'.service',
                          controller=process_identity(child.pid),invocation_id=None,phase='starting')
            save_record(path,record)
            while child.poll() is None:
                if time.monotonic()>record['startup_deadline'] and not record['ready']:
                    raise TimeoutError('profile startup timed out')
                props=unit_properties(record['unit'])
                bound=bind_unit(record,props)
                if bound: save_record(path,record)
                if cancelled:
                    if bound: stop_unit(record)
                elif bound and not record['ready']:
                    try: readiness=authenticated_ready(snapshot,output)
                    except (ConnectionError,urllib.error.URLError,TimeoutError): pass
                    else:
                        record.update(ready=True,readiness=readiness,phase='running');save_record(path,record)
                        print(json.dumps({'event':'ready','profile':snapshot['profile_id'],'port':snapshot['port'],'output':str(output)}),flush=True)
                time.sleep(1)
            return child.returncode
    except BaseException as error:
        record['failure']=type(error).__name__+': '+str(error);save_record(path,record)
        raise
    finally:
        try:
            try:
                if record.get('unit'): stop_unit(record)
            except BaseException as error:
                record['cleanup_failure']=type(error).__name__+': '+str(error);save_record(path,record)
                raise
            finally:
                if child is not None: wait_controller(child,record,path)
        finally:
            for signum,handler in previous.items(): signal.signal(signum,handler)
