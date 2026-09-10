"""Optional GLM profiles: declarative configuration and identity-bound lifecycle.

No production switch/default state is read or written here.
"""
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
        '--property=InvocationID,ActiveState,ControlGroup'],text=True,capture_output=True)
    if result.returncode: return {}
    return dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)


def stop_unit(record):
    props = unit_properties(record['unit'])
    if props.get('ActiveState') in (None,'inactive','failed'): return
    if props.get('InvocationID') != record.get('invocation_id'):
        raise ValueError('systemd invocation identity mismatch; refusing stop')
    subprocess.run(['systemctl','--user','stop',record['unit']],check=True,timeout=60)
    if unit_properties(record['unit']).get('ActiveState') not in (None,'inactive','failed'):
        raise ValueError('model unit survived stop')
    cgroup = record.get('control_group')
    if cgroup:
        events = Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'cgroup.events'
        if events.exists() and 'populated 1' in events.read_text():
            raise ValueError('model descendants survived stop')


def save_record(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.chmod(0o600)
    temporary.replace(path)


def lifecycle_path(name):
    return Path.home()/'.local/state/glm53-profiles'/ (name.split('/')[-1]+'.json')


def lifecycle_action(name, action):
    if name.removesuffix('.json') not in NAMES: raise ValueError('unknown experimental profile')
    path = lifecycle_path(name.removesuffix('.json'))
    if not path.exists():
        print(json.dumps({'profile':name,'state':'stopped','registered_run':False}));return
    record = strict_json(path)
    if record['profile'] != name.removesuffix('.json'): raise ValueError('profile identity mismatch')
    props = unit_properties(record['unit'])
    active = props.get('ActiveState') not in (None,'inactive','failed')
    if active and props.get('InvocationID') != record['invocation_id']:
        raise ValueError('systemd invocation identity mismatch')
    if action == 'stop': stop_unit(record);active=False
    print(json.dumps({'profile':name,'state':('running' if record.get('ready') else 'starting') if active else 'stopped',
                      'output':record['output'],'port':8015,'qualified':False}))


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


def run_contained(command, control, output, snapshot):
    """Foreground launcher with unit-bound shutdown, including Ctrl-C/TERM."""
    path=lifecycle_path(snapshot['profile_id']);path.parent.mkdir(parents=True,exist_ok=True)
    path.parent.chmod(0o700)
    with path.with_suffix('.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise ValueError('this profile already has an active launcher')
        if path.exists():
            old=strict_json(path)
            if unit_properties(old['unit']).get('ActiveState') not in (None,'inactive','failed'):
                raise ValueError('existing model unit must be stopped before starting this profile')
        cancelled=[]
        def cancel(signum, frame): cancelled.append(signum)
        previous={s:signal.signal(s,cancel) for s in (signal.SIGTERM,signal.SIGINT)}
        record=None
        with (output/'wrapper.log').open('w') as log:
            child=subprocess.Popen(command,env=control,stdout=log,stderr=subprocess.STDOUT)
            unit='glm52-'+command[command.index('--tag')+1]+'-'+str(child.pid)+'.service'
            try:
                deadline=time.monotonic()+snapshot['safety']['startup_timeout_seconds']
                while child.poll() is None:
                    props=unit_properties(unit)
                    if props.get('InvocationID') and record is None:
                        record={'profile':snapshot['profile_id'],'output':str(output),'unit':unit,
                                'invocation_id':props['InvocationID'],'control_group':props.get('ControlGroup'),
                                'launcher':process_identity(os.getpid()),'ready':False}
                        save_record(path,record)
                    if cancelled:
                        if record: stop_unit(record)
                        # Unit creation may still be in flight; do not abandon controller.
                    elif record and not record['ready']:
                        if time.monotonic()>deadline: raise TimeoutError('profile startup timed out')
                        try: readiness=authenticated_ready(snapshot,output)
                        except (ConnectionError,urllib.error.URLError,TimeoutError): pass
                        else:
                            record['ready']=True;record['readiness']=readiness;save_record(path,record)
                            print(json.dumps({'event':'ready','profile':snapshot['profile_id'],'port':snapshot['port'],'output':str(output)}),flush=True)
                    time.sleep(1)
                return child.returncode
            finally:
                if record: stop_unit(record)
                try: child.wait(timeout=60)
                finally:
                    for signum,handler in previous.items(): signal.signal(signum,handler)
