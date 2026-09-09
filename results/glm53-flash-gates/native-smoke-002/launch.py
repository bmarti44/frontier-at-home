import hashlib,json,os,pathlib,subprocess,sys,time
output=pathlib.Path(__file__).parent
manifest=json.loads((output/'manifest.json').read_text());receipt=json.loads((output/'randomness.json').read_text())
sys.path.insert(0,str(output/'code/scripts/lib'))
from glm53_contract import sha256_file,strict_json,verify_inventory
for name,row in manifest['orchestration'].items():
 if sha256_file(output/name)!=row['sha256']:raise ValueError('frozen orchestration changed')
for name,row in manifest['code'].items():
 if sha256_file(output/'code'/name)!=row['sha256']:raise ValueError('frozen code changed')
for name,row in manifest['tools'].items():
 if sha256_file(pathlib.Path(name))!=row['sha256']:raise ValueError('frozen tool changed')
inventory=pathlib.Path(manifest['runtime']['manifest'])
if sha256_file(inventory)!=manifest['runtime']['sha256']:raise ValueError('runtime manifest changed')
verify_inventory(pathlib.Path(manifest['runtime']['root']),strict_json(inventory))
if type(receipt['round']) is not int or receipt['round']<1:raise ValueError('invalid public round')
publication=1595431050+(receipt['round']-1)*30
if publication<=manifest['frozen_at_unix'] or publication!=receipt['publication_unix']:raise ValueError('beacon precedes freeze')
verified=subprocess.run(['/home/bmarti44/.nvm/versions/node/v22.22.2/bin/node',str(output/'code/scripts/103_verify_drand_receipt_bundle.mjs'),str(receipt['round']),receipt['randomness'],receipt['signature'],receipt['previous_signature']],env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'},capture_output=True,text=True,timeout=30)
if verified.returncode or verified.stdout!='DRAND_BLS_RECEIPT_OK\n':raise ValueError('public beacon did not verify at launch')
if receipt['seed']!=int(receipt['randomness'][:16],16):raise ValueError('seed derivation mismatch')
command=['/usr/bin/env','-i',*(k+'='+v for k,v in sorted(manifest['environment'].items())),*manifest['argv_without_seed'],'--seed',str(receipt['seed'])]
wrapper='/home/bmarti44/spark-deepseek-v4-flash/results/glm52-gates/harness/glm_cgroup_run.sh'
env={k:v for k,v in os.environ.items() if not k.startswith(('GLM_SAFE_','GLM_W1_'))}
env.update(GLM_SAFE_RUN_AS_CURRENT_USER='1',GLM_SAFE_MIN_START_GIB='110',GLM_SAFE_KILL_FLOOR_GIB='40',GLM_SAFE_TIMEOUT_S='600')
(output/'invocation.json').write_text(json.dumps({'command':['bash',wrapper,'--tag','glm53-native-smoke-002','--',*command],'start_unix':time.time(),'freeze':{'sha256':sha256_file(output/'manifest.json')},'randomness':{'sha256':sha256_file(output/'randomness.json')}},indent=2)+'\n')
with (output/'wrapper.log').open('wb') as log:r=subprocess.run(['bash',wrapper,'--tag','glm53-native-smoke-002','--',*command],env=env,stdout=log,stderr=subprocess.STDOUT)
postcheck=None
try:verify_inventory(pathlib.Path(manifest['runtime']['root']),strict_json(inventory))
except Exception as error:postcheck=repr(error)
summary={'verdict':'PASS' if r.returncode==0 and postcheck is None else 'FAIL','qualification':'synthetic_native_smoke_only','wrapper_exit_code':r.returncode,'runtime_postcheck_failure':postcheck,'model_loaded':False,'end_unix':time.time(),'wrapper':{'sha256':sha256_file(output/'wrapper.log')}}
(output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary));raise SystemExit(0 if summary['verdict']=='PASS' else 1)
